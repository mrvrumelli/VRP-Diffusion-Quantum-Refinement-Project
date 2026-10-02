"""Expand the equation in a config into a QUBO for one neighborhood."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from vrp_diffusion_quantum.quantum.qubo import AnnealingSettings, QaoaSettings, Qubo

logger = logging.getLogger(__name__)

_FUNCTIONS = frozenset({"abs", "max", "min", "sum", "where", "len", "slack_bits"})
_COMPARISONS = frozenset({"<", ">", "<=", ">=", "==", "!="})
_BLOCKS = frozenset({"sizes", "bits", "definitions", "qaoa", "annealing"})
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOKEN = re.compile(
    r"\s+|#.*?$|\*\*|<=|>=|==|!=|\+|-|\*|/|<|>|\(|\)|\[|\]|,|"
    r"\d+\.\d*|\.\d+|\d+|[A-Za-z_][A-Za-z0-9_]*",
    re.MULTILINE,
)


class QuboBuilder:
    """Read an energy expression from a config and expand it in ``build``.

    Bits are 0/1, so a bit squared is the bit. ``sum(i, count, term)`` binds ``i``
    to ``0 .. count-1``. ``where(condition, a, b)`` keeps one branch.
    """

    def __init__(
        self,
        *,
        config_name: str,
        seed: int,
        equation: str,
        parsed_equation: _Expr,
        sizes: tuple[tuple[str, _Expr], ...],
        bits: tuple[tuple[str, tuple[str | int, ...]], ...],
        definitions: tuple[tuple[str, _Expr], ...],
        numbers: dict[str, float],
        qaoa: QaoaSettings,
        annealing: AnnealingSettings,
    ) -> None:
        self.config_name = config_name
        self.seed = seed
        self.equation = equation
        self._equation = parsed_equation
        self._sizes = sizes
        self._bits = bits
        self._definitions = definitions
        self.numbers = dict(numbers)
        self.values = dict(numbers)
        self.penalty_scale = self.numbers.get("penalty_scale", 0.0)
        self.alpha = self.numbers.get("alpha", 0.0)
        self.qaoa = qaoa
        self.annealing = annealing
        self.qubo: Qubo | None = None
        self.sizes: dict[str, int] = {}
        self.penalties: tuple[tuple[str, float], ...] = ()
        self.slack_bits = 0

    @classmethod
    def load(cls, path: Path) -> QuboBuilder:
        """Load the equation and solver settings. Neighborhood arrays come later."""
        loaded = _load_mapping(path)
        equation = loaded.get("equation")
        if not isinstance(equation, str) or not equation.strip():
            raise ValueError("equation must be text")
        numbers = _numbers(loaded)
        if "penalty_scale" in numbers and numbers["penalty_scale"] <= 0.0:
            raise ValueError("penalty_scale must be positive")
        seed = _require_int(loaded, "seed")
        if seed < 0:
            raise ValueError("seed must be an integer >= 0")
        sizes = _size_specs(_optional_mapping(loaded, "sizes"))
        bits = _bit_specs(_require_mapping(loaded, "bits"))
        definitions = _expressions(_optional_mapping(loaded, "definitions"))
        _unique_names(numbers, sizes, bits, definitions)
        qaoa_block = _require_mapping(loaded, "qaoa")
        annealing_block = _require_mapping(loaded, "annealing")
        builder = cls(
            config_name=path.stem,
            seed=seed,
            equation=equation.strip(),
            parsed_equation=_parse(equation),
            sizes=sizes,
            bits=bits,
            definitions=definitions,
            numbers=numbers,
            qaoa=QaoaSettings(
                depth=_require_int(qaoa_block, "depth"),
                shots=_require_int(qaoa_block, "shots"),
                maxiter=_require_int(qaoa_block, "maxiter"),
                restarts=_require_int(qaoa_block, "restarts"),
                seed=seed,
                noise=_optional_string(qaoa_block, "noise", "noiseless"),
                noise_probability=_optional_float(qaoa_block, "noise_probability", 0.0),
            ),
            annealing=AnnealingSettings(
                num_samples=_require_int(annealing_block, "num_samples"),
                num_sweeps=_require_int(annealing_block, "num_sweeps"),
                initial_temperature=_require_float(annealing_block, "initial_temperature"),
                final_temperature=_require_float(annealing_block, "final_temperature"),
                seed=seed,
            ),
        )
        logger.info(
            "config_name=%s seed=%d equation=%s",
            builder.config_name,
            seed,
            " ".join(builder.equation.split()),
        )
        return builder

    def build(self, **inputs: object) -> Qubo:
        """Bind ``inputs`` and expand the configured equation into a QUBO."""
        scalars = dict(self.numbers)
        arrays: dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            _store_input(name, value, scalars, arrays)
        env = _Env(scalars=scalars, arrays=arrays, bits={}, indices={})
        sizes: dict[str, int] = {}
        for name, expr in self._sizes:
            size = _as_int(_eval(expr, env), name)
            if size < 0:
                raise ValueError(f"{name} must be >= 0")
            sizes[name] = size
            env.scalars[name] = float(size)
        bits: dict[str, tuple[tuple[int, ...], int]] = {}
        count = 0
        for name, dims in self._bits:
            shape = tuple(_resolve_dim(dim, sizes) for dim in dims)
            width = math.prod(shape)
            bits[name] = (shape, count)
            count += width
        if count < 1:
            raise ValueError("equation must include at least one bit")
        env.bits = bits
        self.values = dict(self.numbers)
        defined: list[tuple[str, float]] = []
        for name, expr in self._definitions:
            value = _as_float(_eval(expr, env), name)
            env.scalars[name] = value
            self.values[name] = value
            defined.append((name, value))
        qubo = _qubo_from_poly(_eval(self._equation, env), count)
        self.qubo = qubo
        self.sizes = sizes
        self.penalties = tuple(defined)
        self.slack_bits = sizes.get("slack_bits", 0)
        logger.info(
            "qubo_num_variables=%d qubo_num_terms=%d energy_scale=%.6f penalties=%s",
            qubo.num_variables,
            qubo.num_terms,
            qubo.energy_scale,
            dict(self.penalties),
        )
        return qubo

    def penalty(self, name: str) -> float:
        if name not in self.values:
            raise ValueError(f"unknown value {name}")
        return self.values[name]


@dataclass(frozen=True)
class _Num:
    value: float


@dataclass(frozen=True)
class _Name:
    name: str


@dataclass(frozen=True)
class _Unary:
    op: str
    expr: _Expr


@dataclass(frozen=True)
class _Binary:
    op: str
    left: _Expr
    right: _Expr


@dataclass(frozen=True)
class _Call:
    name: str
    args: tuple[_Expr, ...]


@dataclass(frozen=True)
class _Index:
    name: str
    indexes: tuple[_Expr, ...]


_Expr = _Num | _Name | _Unary | _Binary | _Call | _Index


class _Poly:
    """Multilinear polynomial. A squared bit is the bit, so keys are variable sets."""

    def __init__(self, terms: dict[frozenset[int], float]) -> None:
        self.terms = terms

    @classmethod
    def const(cls, value: float) -> _Poly:
        if value == 0.0:
            return cls({})
        return cls({frozenset(): value})

    @classmethod
    def var(cls, index: int) -> _Poly:
        return cls({frozenset((index,)): 1.0})


@dataclass
class _Env:
    scalars: dict[str, float]
    arrays: dict[str, np.ndarray]
    bits: dict[str, tuple[tuple[int, ...], int]]
    indices: dict[str, int]


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> _Expr:
        if not self.tokens:
            raise ValueError("empty expression")
        expr = self._comparison()
        if self.pos != len(self.tokens):
            raise ValueError(f"unexpected {self.tokens[self.pos]!r}")
        return expr

    def _comparison(self) -> _Expr:
        left = self._sum()
        op = self._peek()
        if op in _COMPARISONS:
            self._advance()
            return _Binary(op, left, self._sum())
        return left

    def _sum(self) -> _Expr:
        left = self._product()
        while self._peek() in {"+", "-"}:
            op = self._advance()
            left = _Binary(op, left, self._product())
        return left

    def _product(self) -> _Expr:
        left = self._power()
        while self._peek() in {"*", "/"}:
            op = self._advance()
            left = _Binary(op, left, self._power())
        return left

    def _power(self) -> _Expr:
        if self._peek() == "-":
            self._advance()
            return _Unary("-", self._power())
        left = self._primary()
        if self._peek() == "**":
            self._advance()
            return _Binary("**", left, self._power())
        return left

    def _primary(self) -> _Expr:
        token = self._advance()
        if token == "(":
            expr = self._comparison()
            self._expect(")")
            return expr
        if token[0].isdigit() or token[0] == ".":
            return _Num(float(token))
        if not _NAME.fullmatch(token):
            raise ValueError(f"unexpected {token!r}")
        if self._peek() == "(":
            self._advance()
            return _Call(token, tuple(self._args(")")))
        if self._peek() == "[":
            self._advance()
            indexes = self._args("]")
            if not indexes:
                raise ValueError(f"{token} needs an index")
            return _Index(token, tuple(indexes))
        return _Name(token)

    def _args(self, closer: str) -> list[_Expr]:
        if self._peek() == closer:
            self._advance()
            return []
        args = [self._comparison()]
        while self._peek() == ",":
            self._advance()
            args.append(self._comparison())
        self._expect(closer)
        return args

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _advance(self) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError("unexpected end of expression")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, token: str) -> None:
        found = self._advance()
        if found != token:
            raise ValueError(f"expected {token!r}, found {found!r}")


def _tokenize(source: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    for match in _TOKEN.finditer(source):
        if match.start() != pos:
            raise ValueError(f"cannot read {source[pos : match.start()]!r}")
        pos = match.end()
        text = match.group()
        if text[0].isspace() or text.startswith("#"):
            continue
        tokens.append(text)
    if pos != len(source):
        raise ValueError(f"cannot read {source[pos:]!r}")
    return tokens


def _parse(source: str) -> _Expr:
    expr = _Parser(_tokenize(source)).parse()
    _check_calls(expr)
    return expr


def _check_calls(expr: _Expr) -> None:
    if isinstance(expr, _Call):
        _check_call(expr)
        for arg in expr.args:
            _check_calls(arg)
        return
    if isinstance(expr, _Unary):
        _check_calls(expr.expr)
        return
    if isinstance(expr, _Binary):
        _check_calls(expr.left)
        _check_calls(expr.right)
        return
    if isinstance(expr, _Index):
        for index in expr.indexes:
            _check_calls(index)


def _check_call(expr: _Call) -> None:
    if expr.name not in _FUNCTIONS:
        raise ValueError(f"unknown function {expr.name}")
    if expr.name == "sum" and (len(expr.args) != 3 or not isinstance(expr.args[0], _Name)):
        raise ValueError("sum() is sum(index, count, term)")
    if expr.name == "where" and len(expr.args) != 3:
        raise ValueError("where() is where(condition, if_true, if_false)")
    if expr.name == "len" and len(expr.args) != 1:
        raise ValueError("len() expects one array")
    if expr.name == "slack_bits" and len(expr.args) != 2:
        raise ValueError("slack_bits() expects capacity and fixed load")
    if expr.name == "abs" and len(expr.args) != 1:
        raise ValueError("abs() expects one value")
    if expr.name in {"max", "min"} and not expr.args:
        raise ValueError(f"{expr.name}() expects a value")


def _eval(expr: _Expr, env: _Env) -> _Poly:
    if isinstance(expr, _Num):
        return _Poly.const(expr.value)
    if isinstance(expr, _Name):
        return _name(expr.name, env)
    if isinstance(expr, _Index):
        return _index(expr, env)
    if isinstance(expr, _Unary):
        return _scale(_eval(expr.expr, env), -1.0)
    if isinstance(expr, _Binary):
        return _binary(expr, env)
    return _call(expr, env)


def _name(name: str, env: _Env) -> _Poly:
    if name in env.indices:
        return _Poly.const(float(env.indices[name]))
    if name in env.scalars:
        return _Poly.const(env.scalars[name])
    if name in env.bits or name in env.arrays:
        raise ValueError(f"index {name}")
    raise ValueError(f"unknown name '{name}'")


def _index(expr: _Index, env: _Env) -> _Poly:
    indexes = [_as_int(_eval(item, env), expr.name) for item in expr.indexes]
    if expr.name in env.bits:
        shape, offset = env.bits[expr.name]
        return _Poly.var(offset + _ravel(shape, indexes, expr.name))
    if expr.name in env.arrays:
        return _Poly.const(_array_at(env.arrays[expr.name], indexes, expr.name))
    raise ValueError(f"unknown name '{expr.name}'")


def _binary(expr: _Binary, env: _Env) -> _Poly:
    if expr.op in _COMPARISONS:
        raise ValueError("a comparison must be the condition of where()")
    left = _eval(expr.left, env)
    right = _eval(expr.right, env)
    if expr.op == "+":
        return _add(left, right)
    if expr.op == "-":
        return _add(left, _scale(right, -1.0))
    if expr.op == "*":
        return _mul(left, right)
    if expr.op == "/":
        divisor = _as_float(right, "divisor")
        if divisor == 0.0:
            raise ValueError("division by zero")
        return _scale(left, 1.0 / divisor)
    if expr.op == "**":
        return _power(left, right)
    raise ValueError(f"unknown operator {expr.op}")


def _power(base: _Poly, exponent: _Poly) -> _Poly:
    power = _as_int(exponent, "exponent")
    if power < 0 or power > 60:
        raise ValueError("exponent must be an integer from 0 to 60")
    result = _Poly.const(1.0)
    for _ in range(power):
        result = _mul(result, base)
    return result


def _call(expr: _Call, env: _Env) -> _Poly:
    if expr.name == "sum":
        return _sum(expr, env)
    if expr.name == "where":
        return _where(expr, env)
    if expr.name == "len":
        return _Poly.const(float(_length(expr, env)))
    if expr.name == "slack_bits":
        capacity = _as_float(_eval(expr.args[0], env), "capacity")
        fixed_load = _as_float(_eval(expr.args[1], env), "fixed load")
        return _Poly.const(float(_slack_width(capacity, fixed_load)))
    values = [_as_float(_eval(arg, env), expr.name) for arg in expr.args]
    if expr.name == "abs":
        return _Poly.const(abs(values[0]))
    if expr.name == "max":
        return _Poly.const(max(values))
    if expr.name == "min":
        return _Poly.const(min(values))
    raise ValueError(f"unknown function {expr.name}")


def _sum(expr: _Call, env: _Env) -> _Poly:
    index_name = expr.args[0]
    if not isinstance(index_name, _Name):
        raise ValueError("sum() is sum(index, count, term)")
    count = _as_int(_eval(expr.args[1], env), "sum count")
    if count < 0:
        raise ValueError("sum count must be >= 0")
    total = _Poly.const(0.0)
    previous = env.indices.get(index_name.name)
    try:
        for index in range(count):
            env.indices[index_name.name] = index
            total = _add(total, _eval(expr.args[2], env))
    finally:
        if previous is None:
            env.indices.pop(index_name.name, None)
        else:
            env.indices[index_name.name] = previous
    return total


def _where(expr: _Call, env: _Env) -> _Poly:
    chosen = expr.args[1] if _truth(expr.args[0], env) else expr.args[2]
    return _eval(chosen, env)


def _truth(expr: _Expr, env: _Env) -> bool:
    if not isinstance(expr, _Binary) or expr.op not in _COMPARISONS:
        raise ValueError("where() condition must be a comparison")
    left = _as_float(_eval(expr.left, env), "where() condition")
    right = _as_float(_eval(expr.right, env), "where() condition")
    if expr.op == "<":
        return left < right
    if expr.op == ">":
        return left > right
    if expr.op == "<=":
        return left <= right
    if expr.op == ">=":
        return left >= right
    if expr.op == "==":
        return left == right
    return left != right


def _length(expr: _Call, env: _Env) -> int:
    target = expr.args[0]
    if not isinstance(target, _Name):
        raise ValueError("len() expects one array")
    if target.name not in env.arrays:
        raise ValueError(f"unknown name '{target.name}'")
    return int(env.arrays[target.name].shape[0])


def _slack_width(capacity: float, fixed_load: float) -> int:
    gap = max(0.0, capacity - min(fixed_load, capacity))
    levels = math.ceil(gap) + 1
    if levels <= 1:
        return 1
    return math.ceil(math.log2(levels))


def _add(left: _Poly, right: _Poly) -> _Poly:
    terms = dict(left.terms)
    for variables, coeff in right.terms.items():
        terms[variables] = terms.get(variables, 0.0) + coeff
    return _Poly(_drop_zeros(terms))


def _scale(poly: _Poly, factor: float) -> _Poly:
    if factor == 0.0:
        return _Poly({})
    terms = {variables: coeff * factor for variables, coeff in poly.terms.items()}
    return _Poly(_drop_zeros(terms))


def _mul(left: _Poly, right: _Poly) -> _Poly:
    terms: dict[frozenset[int], float] = {}
    for left_vars, left_coeff in left.terms.items():
        for right_vars, right_coeff in right.terms.items():
            variables = left_vars | right_vars
            if len(variables) > 2:
                raise ValueError("equation produces a term with more than two variables")
            coeff = left_coeff * right_coeff
            terms[variables] = terms.get(variables, 0.0) + coeff
    return _Poly(_drop_zeros(terms))


def _drop_zeros(terms: dict[frozenset[int], float]) -> dict[frozenset[int], float]:
    return {variables: coeff for variables, coeff in terms.items() if coeff != 0.0}


def _as_float(poly: _Poly, label: str) -> float:
    for variables in poly.terms:
        if variables:
            raise ValueError(f"{label} must be a number")
    return poly.terms.get(frozenset(), 0.0)


def _as_int(poly: _Poly, label: str) -> int:
    value = _as_float(poly, label)
    rounded = round(value)
    if abs(value - rounded) > 1e-8:
        raise ValueError(f"{label} must be an integer")
    return rounded


def _ravel(shape: tuple[int, ...], indexes: list[int], name: str) -> int:
    if len(indexes) != len(shape):
        raise ValueError(f"{name} expects {len(shape)} indexes")
    flat = 0
    for index, size in zip(indexes, shape, strict=True):
        if index < 0 or index >= size:
            raise ValueError(f"{name} index {index} is outside 0..{size - 1}")
        flat = flat * size + index
    return flat


def _array_at(array: np.ndarray, indexes: list[int], name: str) -> float:
    if len(indexes) != array.ndim:
        raise ValueError(f"{name} expects {array.ndim} indexes")
    for index, size in zip(indexes, array.shape, strict=True):
        if index < 0 or index >= int(size):
            raise ValueError(f"{name} index {index} is outside 0..{int(size) - 1}")
    return float(array[tuple(indexes)])


def _resolve_dim(dim: str | int, sizes: dict[str, int]) -> int:
    size = dim if isinstance(dim, int) else sizes.get(dim)
    if size is None:
        raise ValueError(f"unknown size {dim}")
    if size < 1:
        raise ValueError("bit dimension must be >= 1")
    return size


def _qubo_from_poly(poly: _Poly, count: int) -> Qubo:
    linear = np.zeros(count, dtype=np.float64)
    pairs: dict[tuple[int, int], float] = {}
    offset = 0.0
    for variables, coeff in poly.terms.items():
        if not math.isfinite(coeff):
            raise ValueError("equation produced a non-finite coefficient")
        if not variables:
            offset += coeff
            continue
        if len(variables) == 1:
            linear[next(iter(variables))] += coeff
            continue
        i, j = sorted(variables)
        pairs[(i, j)] = pairs.get((i, j), 0.0) + coeff
    quadratic = tuple((i, j, coeff) for (i, j), coeff in sorted(pairs.items()) if coeff != 0.0)
    return Qubo(
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        energy_scale=_coefficient_scale(linear, quadratic, offset),
    )


def _coefficient_scale(
    linear: np.ndarray,
    quadratic: tuple[tuple[int, int, float], ...],
    offset: float,
) -> float:
    peak = abs(offset)
    if linear.size:
        peak = max(peak, float(np.max(np.abs(linear))))
    for _i, _j, coeff in quadratic:
        peak = max(peak, abs(coeff))
    return max(peak, 1.0)


def _store_input(
    name: str,
    value: object,
    scalars: dict[str, float],
    arrays: dict[str, np.ndarray],
) -> None:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a number or an array")
    if isinstance(value, int | float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        scalars[name] = number
        return
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        number = float(array)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        scalars[name] = number
        return
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    arrays[name] = array


def _numbers(mapping: Mapping[str, object]) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for key, value in mapping.items():
        if key in _BLOCKS or key in {"seed", "equation"}:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite")
        numbers[key] = number
    return numbers


def _unique_names(
    numbers: Mapping[str, float],
    sizes: tuple[tuple[str, _Expr], ...],
    bits: tuple[tuple[str, tuple[str | int, ...]], ...],
    definitions: tuple[tuple[str, _Expr], ...],
) -> None:
    used = set(numbers)
    for name, _expr in (*sizes, *definitions):
        if name in used:
            raise ValueError(f"{name} is already defined")
        used.add(name)
    for name, _shape in bits:
        if name in used or name in _FUNCTIONS:
            raise ValueError(f"{name} is already defined")
        used.add(name)


def _size_specs(mapping: Mapping[str, object]) -> tuple[tuple[str, _Expr], ...]:
    items: list[tuple[str, _Expr]] = []
    for name, value in mapping.items():
        _require_identifier(name)
        if isinstance(value, str):
            expr: _Expr = _parse(value)
        elif isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an expression")
        else:
            expr = _Num(float(value))
        items.append((name, expr))
    return tuple(items)


def _bit_specs(
    mapping: Mapping[str, object],
) -> tuple[tuple[str, tuple[str | int, ...]], ...]:
    if not mapping:
        raise ValueError("bits must declare at least one variable")
    items: list[tuple[str, tuple[str | int, ...]]] = []
    for name, value in mapping.items():
        _require_identifier(name)
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} shape must be a non-empty list")
        dims: list[str | int] = []
        for dim in value:
            if isinstance(dim, bool) or not isinstance(dim, str | int):
                raise ValueError(f"{name} shape entries must be sizes or integers")
            if isinstance(dim, str):
                _require_identifier(dim)
            elif dim < 1:
                raise ValueError(f"{name} dimension must be >= 1")
            dims.append(dim)
        items.append((name, tuple(dims)))
    return tuple(items)


def _expressions(mapping: Mapping[str, object]) -> tuple[tuple[str, _Expr], ...]:
    items: list[tuple[str, _Expr]] = []
    for name, value in mapping.items():
        _require_identifier(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be an expression")
        items.append((name, _parse(value)))
    return tuple(items)


def _require_identifier(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a name")


def _optional_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    if key not in mapping:
        return {}
    return _require_mapping(mapping, key)


def _load_mapping(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must be a mapping")
    return {str(key): value for key, value in loaded.items()}


def _require_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = _require(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return {str(item_key): item for item_key, item in value.items()}


def _require(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"missing {key}")
    return mapping[key]


def _optional_string(mapping: Mapping[str, object], key: str, default: str) -> str:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a string")
    return value


def _optional_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    if key not in mapping:
        return default
    return _require_float(mapping, key)


def _require_int(mapping: Mapping[str, object], key: str) -> int:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number
