"""Evaluate a reproducible CVRPLIB CVRP subset.

The default subset is the committed Augerat ``B-n31-k5`` smoke instance from
CVRPLIB.  It is intentionally small, has a matching best-known solution file,
and reaches the best-known cost with a fixed PyVRP iteration budget on the
project development environment::

    python eval/evaluate_cvrplib.py --max-iterations 250 --seed 42

The evaluator reads CVRPLIB ``.vrp`` files, optionally pairs same-stem ``.sol``
files for reference costs, solves with PyVRP, and writes per-instance metrics
plus a summary JSON next to the CSV output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from pyvrp import read as read_vrplib_problem
from pyvrp import solve as solve_vrplib_problem
from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria, StoppingCriterion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET_DIR = ROOT / "data" / "raw" / "cvrplib" / "augerat-b-smoke"
DEFAULT_OUTPUT = ROOT / "eval" / "cvrplib_results.csv"
DEFAULT_ROUND_FUNC = "round"
DEFAULT_MAX_ITERATIONS = 250
DEFAULT_SEED = 42

_SECTION_NAMES = {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}
_COST_RE = re.compile(r"^\s*Cost\s*:?\s+([+-]?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_K_RE = re.compile(r"(?:^|-)k(\d+)(?:$|-)", re.IGNORECASE)
_TRUCKS_RE = re.compile(r"No\s+of\s+trucks\s*:\s*(\d+)", re.IGNORECASE)
_OPT_RE = re.compile(r"Optimal\s+value\s*:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class CVRPLIBInstance:
    """Parsed CVRPLIB metadata and arrays for one CVRP instance."""

    name: str
    path: Path
    dimension: int
    capacity: float
    edge_weight_type: str
    coords: np.ndarray
    demands: np.ndarray
    depot_index: int
    vehicles: int | None
    comment_reference_cost: float | None
    sha256: str

    @property
    def n_customers(self) -> int:
        return self.dimension - 1


@dataclass(frozen=True)
class CVRPLIBSolution:
    """Parsed CVRPLIB solution routes and cost."""

    path: Path
    routes: list[list[int]]
    cost: float
    sha256: str


@dataclass(frozen=True)
class CVRPLIBRecord:
    """One evaluable CVRPLIB instance with an optional reference solution."""

    instance: CVRPLIBInstance
    solution: CVRPLIBSolution | None
    subset: str

    @property
    def reference_cost(self) -> float | None:
        if self.solution is not None:
            return self.solution.cost
        return self.instance.comment_reference_cost


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_header_key(key: str) -> str:
    return key.strip().upper().replace(" ", "_")


def _split_header(line: str) -> tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
    else:
        key, value = line.split(maxsplit=1)
    return _normalise_header_key(key), value.strip()


def _parse_float(value: str, *, field: str, path: Path) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{field} in {path} must be numeric, got {value!r}") from error


def _parse_int(value: str, *, field: str, path: Path) -> int:
    parsed = _parse_float(value, field=field, path=path)
    if not parsed.is_integer():
        raise ValueError(f"{field} in {path} must be an integer, got {value!r}")
    return int(parsed)


def _parse_sections(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    headers: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.rstrip(":").strip().upper()
        if upper == "EOF":
            break
        if upper in _SECTION_NAMES:
            current_section = upper
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            key, value = _split_header(line)
            headers[key] = value
        else:
            sections[current_section].append(line)

    return headers, sections


def _parse_indexed_vector(
    rows: list[str],
    *,
    dimension: int,
    width: int,
    field: str,
    path: Path,
) -> np.ndarray:
    values = np.full((dimension, width), np.nan, dtype=np.float64)
    for row in rows:
        parts = row.split()
        if len(parts) != width + 1:
            raise ValueError(f"{field} row in {path} has wrong width: {row!r}")
        node_id = _parse_int(parts[0], field=f"{field} node id", path=path)
        if not 1 <= node_id <= dimension:
            raise ValueError(f"{field} node id {node_id} is outside 1..{dimension} in {path}")
        values[node_id - 1] = [_parse_float(part, field=field, path=path) for part in parts[1:]]
    if np.isnan(values).any():
        missing = [str(index + 1) for index in np.flatnonzero(np.isnan(values).any(axis=1))]
        raise ValueError(f"{field} in {path} is missing node ids: {', '.join(missing)}")
    return values


def _parse_depot_index(rows: list[str], *, dimension: int, path: Path) -> int:
    depot_ids: list[int] = []
    for row in rows:
        for token in row.split():
            node_id = _parse_int(token, field="depot id", path=path)
            if node_id == -1:
                break
            depot_ids.append(node_id)
        if row.split() and row.split()[-1] == "-1":
            break
    if len(depot_ids) != 1:
        raise ValueError(f"{path} must define exactly one depot, got {depot_ids}")
    depot_id = depot_ids[0]
    if not 1 <= depot_id <= dimension:
        raise ValueError(f"depot id {depot_id} is outside 1..{dimension} in {path}")
    return depot_id - 1


def _parse_vehicles(name: str, headers: dict[str, str]) -> int | None:
    if "VEHICLES" in headers:
        return int(float(headers["VEHICLES"]))
    comment = headers.get("COMMENT", "")
    if match := _TRUCKS_RE.search(comment):
        return int(match.group(1))
    if match := _K_RE.search(name):
        return int(match.group(1))
    return None


def _parse_comment_reference(headers: dict[str, str]) -> float | None:
    if match := _OPT_RE.search(headers.get("COMMENT", "")):
        return float(match.group(1))
    return None


def parse_cvrplib_instance(path: str | Path) -> CVRPLIBInstance:
    """Parse a CVRPLIB ``.vrp`` CVRP instance."""
    source = Path(path)
    headers, sections = _parse_sections(source)
    name = headers.get("NAME", source.stem)
    instance_type = headers.get("TYPE", "").upper()
    if instance_type != "CVRP":
        raise ValueError(f"{source} must be TYPE CVRP, got {instance_type or 'missing'}")
    if "DIMENSION" not in headers:
        raise ValueError(f"{source} is missing DIMENSION")
    if "CAPACITY" not in headers:
        raise ValueError(f"{source} is missing CAPACITY")
    for section in _SECTION_NAMES:
        if section not in sections:
            raise ValueError(f"{source} is missing {section}")

    dimension = _parse_int(headers["DIMENSION"], field="DIMENSION", path=source)
    capacity = _parse_float(headers["CAPACITY"], field="CAPACITY", path=source)
    if dimension <= 1:
        raise ValueError(f"DIMENSION in {source} must be greater than 1")
    if capacity <= 0:
        raise ValueError(f"CAPACITY in {source} must be positive")

    coords = _parse_indexed_vector(
        sections["NODE_COORD_SECTION"],
        dimension=dimension,
        width=2,
        field="NODE_COORD_SECTION",
        path=source,
    )
    demand_rows = _parse_indexed_vector(
        sections["DEMAND_SECTION"],
        dimension=dimension,
        width=1,
        field="DEMAND_SECTION",
        path=source,
    )
    demands = demand_rows[:, 0]
    depot_index = _parse_depot_index(sections["DEPOT_SECTION"], dimension=dimension, path=source)
    if demands[depot_index] != 0:
        raise ValueError(f"depot demand must be 0 in {source}")
    customer_demands = np.delete(demands, depot_index)
    if np.any(customer_demands <= 0):
        raise ValueError(f"all customer demands must be positive in {source}")
    if float(customer_demands.max()) > capacity:
        raise ValueError(f"customer demand exceeds CAPACITY in {source}")

    return CVRPLIBInstance(
        name=name,
        path=source,
        dimension=dimension,
        capacity=capacity,
        edge_weight_type=headers.get("EDGE_WEIGHT_TYPE", ""),
        coords=coords,
        demands=demands,
        depot_index=depot_index,
        vehicles=_parse_vehicles(name, headers),
        comment_reference_cost=_parse_comment_reference(headers),
        sha256=file_sha256(source),
    )


def parse_cvrplib_solution(path: str | Path) -> CVRPLIBSolution:
    """Parse a CVRPLIB ``.sol`` file with routes and reference cost."""
    source = Path(path)
    routes: list[list[int]] = []
    cost: float | None = None

    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("route"):
            _, _, payload = stripped.partition(":")
            routes.append([int(token) for token in payload.split()])
            continue
        if match := _COST_RE.match(stripped):
            cost = float(match.group(1))

    if cost is None:
        raise ValueError(f"{source} is missing a Cost line")
    return CVRPLIBSolution(path=source, routes=routes, cost=cost, sha256=file_sha256(source))


def _solution_path_for(instance_path: Path) -> Path | None:
    candidates = (
        instance_path.with_suffix(".sol"),
        instance_path.parent / "solutions" / f"{instance_path.stem}.sol",
        instance_path.parent / "bks" / f"{instance_path.stem}.sol",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def discover_records(
    subset_dir: str | Path,
    *,
    names: list[str] | None = None,
) -> list[CVRPLIBRecord]:
    """Discover CVRPLIB records in a subset directory."""
    root = Path(subset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"CVRPLIB subset directory not found: {root}")
    requested = None if names is None else set(names)
    records: list[CVRPLIBRecord] = []
    for instance_path in sorted(root.rglob("*.vrp"), key=lambda path: path.stem):
        if requested is not None and instance_path.stem not in requested:
            continue
        instance = parse_cvrplib_instance(instance_path)
        solution_path = _solution_path_for(instance_path)
        solution = None if solution_path is None else parse_cvrplib_solution(solution_path)
        records.append(CVRPLIBRecord(instance=instance, solution=solution, subset=root.name))

    if requested is not None:
        found_names = {record.instance.name for record in records}
        found_stems = {record.instance.path.stem for record in records}
        found = found_names | found_stems
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"requested CVRPLIB instances not found in {root}: {missing}")
    if not records:
        raise FileNotFoundError(f"no .vrp files found in CVRPLIB subset directory: {root}")
    return records


def select_records(
    records: list[CVRPLIBRecord],
    *,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[CVRPLIBRecord]:
    """Return all records, or a deterministic seeded sample."""
    if sample_size is None:
        return list(records)
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if sample_size > len(records):
        raise ValueError(f"sample_size {sample_size} exceeds available records {len(records)}")
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(records), size=sample_size, replace=False)
    indices = sorted(int(index) for index in choices)
    return [records[index] for index in indices]


def subset_hash(records: list[CVRPLIBRecord]) -> str:
    """Return a stable hash over selected instances and optional solution files."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.instance.name):
        digest.update(record.instance.name.encode())
        digest.update(record.instance.sha256.encode())
        if record.solution is not None:
            digest.update(record.solution.sha256.encode())
    return digest.hexdigest()


def _build_stop(
    *,
    max_iterations: int | None = DEFAULT_MAX_ITERATIONS,
    time_limit: float | None = None,
) -> StoppingCriterion:
    criteria: list[StoppingCriterion] = []
    if max_iterations is not None:
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        criteria.append(MaxIterations(max_iterations))
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError(f"time_limit must be positive, got {time_limit}")
        criteria.append(MaxRuntime(time_limit))
    if not criteria:
        raise ValueError("set max_iterations and/or time_limit")
    if len(criteria) == 1:
        return criteria[0]
    return MultipleCriteria(criteria)


def evaluate_record(
    record: CVRPLIBRecord,
    *,
    solver: str = "pyvrp",
    seed: int = DEFAULT_SEED,
    max_iterations: int | None = DEFAULT_MAX_ITERATIONS,
    time_limit: float | None = None,
    round_func: str = DEFAULT_ROUND_FUNC,
    run_index: int = 0,
    selected_subset_hash: str | None = None,
) -> dict[str, Any]:
    """Solve one CVRPLIB record and return a metric row."""
    if solver != "pyvrp":
        raise ValueError(f"unsupported CVRPLIB solver: {solver!r}")

    instance_seed = int(
        np.random.SeedSequence([seed, record.instance.dimension, run_index]).generate_state(1)[0]
    )
    problem = read_vrplib_problem(record.instance.path, round_func=round_func)
    result = solve_vrplib_problem(
        problem,
        _build_stop(max_iterations=max_iterations, time_limit=time_limit),
        seed=instance_seed,
        display=False,
    )
    reference_cost = record.reference_cost
    cost = float(result.cost())
    gap = None
    if reference_cost is not None:
        if reference_cost <= 0:
            raise ValueError(f"reference cost must be positive, got {reference_cost}")
        gap = 100.0 * (cost - reference_cost) / reference_cost

    return {
        "subset": record.subset,
        "subset_hash": selected_subset_hash,
        "instance": record.instance.name,
        "size": record.instance.n_customers,
        "dimension": record.instance.dimension,
        "capacity": record.instance.capacity,
        "declared_vehicles": record.instance.vehicles,
        "cost": cost,
        "reference_cost": reference_cost,
        "gap": gap,
        "runtime": float(result.runtime),
        "feasibility": bool(result.is_feasible()),
        "number_of_vehicles": int(result.best.num_routes()),
        "iterations": int(result.num_iterations),
        "solver": solver,
        "seed": instance_seed,
        "base_seed": seed,
        "max_iterations": max_iterations,
        "time_limit": time_limit,
        "round_func": round_func,
        "instance_sha256": record.instance.sha256,
        "solution_sha256": None if record.solution is None else record.solution.sha256,
    }


def evaluate_subset(
    subset_dir: str | Path = DEFAULT_SUBSET_DIR,
    *,
    names: list[str] | None = None,
    sample_size: int | None = None,
    solver: str = "pyvrp",
    seed: int = DEFAULT_SEED,
    max_iterations: int | None = DEFAULT_MAX_ITERATIONS,
    time_limit: float | None = None,
    round_func: str = DEFAULT_ROUND_FUNC,
) -> list[dict[str, Any]]:
    """Evaluate a CVRPLIB subset directory reproducibly."""
    records = select_records(
        discover_records(subset_dir, names=names),
        sample_size=sample_size,
        seed=seed,
    )
    selected_hash = subset_hash(records)
    return [
        evaluate_record(
            record,
            solver=solver,
            seed=seed,
            max_iterations=max_iterations,
            time_limit=time_limit,
            round_func=round_func,
            run_index=index,
            selected_subset_hash=selected_hash,
        )
        for index, record in enumerate(records)
    ]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate CVRPLIB rows by subset."""
    summary: list[dict[str, Any]] = []
    for subset in sorted({str(row["subset"]) for row in rows}):
        subset_rows = [row for row in rows if row["subset"] == subset]
        gaps = [float(row["gap"]) for row in subset_rows if row["gap"] is not None]
        summary.append(
            {
                "subset": subset,
                "subset_hash": subset_rows[0]["subset_hash"],
                "instances": len(subset_rows),
                "cost": fmean(float(row["cost"]) for row in subset_rows),
                "gap": None if not gaps else fmean(gaps),
                "runtime": fmean(float(row["runtime"]) for row in subset_rows),
                "feasibility": fmean(1.0 if row["feasibility"] else 0.0 for row in subset_rows),
                "number_of_vehicles": fmean(
                    float(row["number_of_vehicles"]) for row in subset_rows
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CVRPLIB evaluation result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--instances", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--solver", choices=("pyvrp",), default="pyvrp")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument(
        "--round-func",
        choices=("round", "none", "trunc", "dimacs", "exact"),
        default=DEFAULT_ROUND_FUNC,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = evaluate_subset(
        args.subset_dir,
        names=args.instances,
        sample_size=args.sample_size,
        solver=args.solver,
        seed=args.seed,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        round_func=args.round_func,
    )
    _write_csv(args.output, rows)
    summary = summarize(rows)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    for row in summary:
        print(
            f"{row['subset']}: instances={row['instances']} cost={row['cost']:.4f} "
            f"gap={row['gap']} runtime={row['runtime']:.4f}s "
            f"feasibility={row['feasibility']:.4f} "
            f"number_of_vehicles={row['number_of_vehicles']:.2f} "
            f"subset_hash={row['subset_hash']}"
        )


if __name__ == "__main__":
    main()
