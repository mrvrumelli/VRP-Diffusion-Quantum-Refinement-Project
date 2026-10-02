import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vrp_diffusion_quantum.quantum.annealing_solver import solve_annealing
from vrp_diffusion_quantum.quantum.qaoa_solver import MAX_QAOA_VARIABLES, solve_qaoa
from vrp_diffusion_quantum.quantum.qubo import (
    AnnealingSettings,
    QaoaSettings,
    Qubo,
    QuboSolveResult,
    assignment_bits,
    decode_reorder,
)
from vrp_diffusion_quantum.quantum.qubo_loader import QuboBuilder

ROOT = Path(__file__).resolve().parents[1]
CONFIG_A = ROOT / "configs" / "quantum" / "qubo_a_reorder.yaml"
CONFIG_B = ROOT / "configs" / "quantum" / "qubo_b_routes.yaml"
DISTANCE = np.array(
    [
        [0.0, 1.0, 5.0, 0.0],
        [1.0, 0.0, 1.0, 5.0],
        [5.0, 1.0, 0.0, 1.0],
        [0.0, 5.0, 1.0, 0.0],
    ]
)


def test_reorder_qubo_matches_formula_and_selects_short_order() -> None:
    model = QuboBuilder.load(CONFIG_A)
    qubo = model.build(segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE)
    bits = np.array([1, 0, 0, 1], dtype=np.int8)

    assert qubo.energy(bits) == pytest.approx(_reorder_formula(bits, model))
    assert decode_reorder(bits, (1, 2)) == (1, 2)
    assert qubo.num_variables == 4
    assert qubo.energy(np.zeros(4, dtype=np.int8)) > qubo.energy(bits)

    minimum = int(np.argmin(qubo.energies()))
    minimum_bits = ((minimum >> np.arange(qubo.num_variables)) & 1).astype(np.int8)
    assert decode_reorder(minimum_bits, (1, 2)) == (1, 2)


def test_two_route_qubo_matches_formula_and_selects_compatible_split() -> None:
    model = QuboBuilder.load(CONFIG_B)
    qubo = model.build(**_route_kwargs())
    bits = _route_bits((1, 0), slack=2, slack_bits=model.slack_bits)

    assert qubo.energy(bits) == pytest.approx(_two_route_formula(bits, model))
    assert assignment_bits(bits, 2) == (1, 0)
    assert qubo.energy(bits) == pytest.approx(-1.8)

    minimum = int(np.argmin(qubo.energies()))
    minimum_bits = ((minimum >> np.arange(qubo.num_variables)) & 1).astype(np.int8)
    assert assignment_bits(minimum_bits, 2) == (1, 0)


def test_penalties_and_slack_follow_the_input() -> None:
    reorder = QuboBuilder.load(CONFIG_A)
    reorder.build(segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE)
    small_penalty = reorder.penalty("lambda_customer")
    reorder.build(segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE * 10.0)
    assert reorder.penalty("lambda_customer") == pytest.approx(10.0 * small_penalty)

    routes = QuboBuilder.load(CONFIG_B)
    narrow = routes.build(**_route_kwargs(capacity=5.0))
    narrow_bits = routes.slack_bits
    wide = routes.build(**_route_kwargs(capacity=40.0))
    assert routes.slack_bits > narrow_bits
    assert wide.num_variables > narrow.num_variables


def test_main_scenarios_from_both_configs() -> None:
    reorder_model = QuboBuilder.load(CONFIG_A)
    reorder = reorder_model.build(segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE)
    reorder_optimum = float(np.min(reorder.energies()))
    annealed = solve_annealing(reorder, reorder_model.annealing)
    assert annealed.energy == pytest.approx(reorder_optimum)
    assert decode_reorder(annealed.bitstring, (1, 2)) == (1, 2)

    ideal = solve_qaoa(reorder, reorder_model.qaoa)
    _assert_sample_energies(reorder, ideal)
    assert ideal.energy == pytest.approx(reorder_optimum)
    assert decode_reorder(ideal.bitstring, (1, 2)) == (1, 2)
    noisy_settings = replace(reorder_model.qaoa, noise="depolarizing", noise_probability=0.05)
    noisy = solve_qaoa(reorder, noisy_settings)
    _assert_sample_energies(reorder, noisy)
    assert noisy.energy == pytest.approx(reorder_optimum)
    assert decode_reorder(noisy.bitstring, (1, 2)) == (1, 2)
    assert noisy.expectation is not None and ideal.expectation is not None
    assert noisy.expectation > ideal.expectation
    assert np.array_equal(noisy.samples, solve_qaoa(reorder, noisy_settings).samples)

    route_model = QuboBuilder.load(CONFIG_B)
    routes = route_model.build(**_route_kwargs())
    route_optimum = float(np.min(routes.energies()))
    routed = solve_annealing(routes, route_model.annealing)
    assert routed.energy == pytest.approx(route_optimum)
    assert assignment_bits(routed.bitstring, 2) == (1, 0)

    route_ideal = solve_qaoa(routes, route_model.qaoa)
    route_noisy_settings = replace(route_model.qaoa, noise="depolarizing", noise_probability=0.05)
    route_noisy = solve_qaoa(routes, route_noisy_settings)
    _assert_sample_energies(routes, route_ideal)
    _assert_sample_energies(routes, route_noisy)
    assert route_noisy.expectation is not None and route_ideal.expectation is not None
    assert route_noisy.expectation > route_ideal.expectation
    assert np.array_equal(route_noisy.samples, solve_qaoa(routes, route_noisy_settings).samples)


def test_annealing_reaches_both_qubo_optima() -> None:
    reorder = QuboBuilder.load(CONFIG_A)
    reorder_qubo = reorder.build(segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE)
    reorder_result = solve_annealing(reorder_qubo, reorder.annealing)
    assert reorder_result.energy == pytest.approx(float(np.min(reorder_qubo.energies())))
    assert decode_reorder(reorder_result.bitstring, (1, 2)) == (1, 2)
    assert reorder_result.solver_name == "simulated_annealing"

    routes = QuboBuilder.load(CONFIG_B)
    route_qubo = routes.build(**_route_kwargs())
    route_result = solve_annealing(route_qubo, routes.annealing)
    assert route_result.energy == pytest.approx(float(np.min(route_qubo.energies())))
    assert assignment_bits(route_result.bitstring, 2) == (1, 0)


def test_annealing_repeat_uses_the_same_seed() -> None:
    model = QuboBuilder.load(CONFIG_A)
    qubo = model.build(segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE)
    first = solve_annealing(qubo, model.annealing)
    second = solve_annealing(qubo, model.annealing)
    assert np.array_equal(first.samples, second.samples)
    assert first.sample_energies == pytest.approx(second.sample_energies)


def test_qaoa_samples_match_qubo_energy_for_both_kinds(caplog: pytest.LogCaptureFixture) -> None:
    settings = QaoaSettings(depth=1, shots=8, maxiter=20, restarts=1, seed=11)
    with caplog.at_level(logging.INFO):
        qubos = (
            QuboBuilder.load(CONFIG_A).build(
                segment=(1, 2), predecessor=0, successor=3, distance=DISTANCE
            ),
            QuboBuilder.load(CONFIG_B).build(**_route_kwargs()),
        )
        for qubo in qubos:
            result = solve_qaoa(qubo, settings)
            for sample, energy in zip(result.samples, result.sample_energies, strict=True):
                assert qubo.energy(sample) == pytest.approx(float(energy))
            assert result.energy == pytest.approx(float(np.min(result.sample_energies)))
            assert result.qaoa_depth == 1
            assert result.expectation is not None
    assert "raw_energy=" in caplog.text
    assert "energy_scale=" in caplog.text


def test_qaoa_beats_uniform_expectation_on_a_two_bit_qubo() -> None:
    qubo = Qubo(linear=np.array([1.0, 2.0]), quadratic=(), offset=0.0)
    settings = QaoaSettings(depth=1, shots=32, maxiter=40, restarts=2, seed=3)
    result = solve_qaoa(qubo, settings)
    assert result.expectation is not None
    assert result.expectation < float(np.mean(qubo.energies()))
    assert result.energy == pytest.approx(0.0)


def test_depolarizing_qaoa_keeps_qubo_energies_and_changes_the_expectation() -> None:
    qubo = Qubo(linear=np.array([1.0, 2.0]), quadratic=(), offset=0.0)
    shared = {"depth": 1, "shots": 16, "maxiter": 8, "restarts": 1, "seed": 3}
    noiseless = solve_qaoa(qubo, QaoaSettings(**shared))
    noisy = solve_qaoa(
        qubo,
        QaoaSettings(**shared, noise="depolarizing", noise_probability=0.25),
    )
    repeated = solve_qaoa(
        qubo,
        QaoaSettings(**shared, noise="depolarizing", noise_probability=0.25),
    )
    for sample, energy in zip(noisy.samples, noisy.sample_energies, strict=True):
        assert qubo.energy(sample) == pytest.approx(float(energy))
    assert noiseless.expectation is not None
    assert noisy.expectation is not None
    assert noisy.expectation != pytest.approx(noiseless.expectation)
    assert np.array_equal(noisy.samples, repeated.samples)
    assert noisy.sample_energies == pytest.approx(repeated.sample_energies)


def test_qaoa_rejects_unknown_noise() -> None:
    with pytest.raises(ValueError, match=r"qaoa\.noise"):
        QaoaSettings(depth=1, shots=1, maxiter=1, restarts=1, seed=0, noise="readout")


def test_qaoa_rejects_a_qubo_above_the_variable_cap() -> None:
    qubo = Qubo(linear=np.zeros(MAX_QAOA_VARIABLES + 1), quadratic=())
    settings = QaoaSettings(depth=1, shots=1, maxiter=1, restarts=1, seed=0)
    with pytest.raises(ValueError, match="at most"):
        solve_qaoa(qubo, settings)


def test_one_builder_uses_the_config_equation() -> None:
    reorder = QuboBuilder.load(CONFIG_A)
    routes = QuboBuilder.load(CONFIG_B)
    assert isinstance(reorder, QuboBuilder)
    assert isinstance(routes, QuboBuilder)
    assert "x[i, p]" in reorder.equation
    assert "(1 - y[i])" in routes.equation
    assert reorder.penalty_scale == pytest.approx(2.0)
    assert routes.alpha == pytest.approx(1.0)
    assert reorder.qaoa.noise == "noiseless"
    assert routes.qaoa.noise_probability == pytest.approx(0.0)
    assert reorder.qubo is None
    with pytest.raises(ValueError, match="segment"):
        reorder.build(**_route_kwargs())


def test_any_equation_expands_to_its_qubo(tmp_path: Path) -> None:
    path = _write_equation(tmp_path / "custom.yaml", "(1 - x[0] - x[1]) ** 2 + 5 * x[0]")
    qubo = QuboBuilder.load(path).build()
    assert qubo.num_variables == 2
    assert qubo.energy(np.array([1, 0])) == pytest.approx(5.0)
    assert qubo.energy(np.array([1, 1])) == pytest.approx(6.0)
    assert qubo.energy(np.array([0, 0])) == pytest.approx(1.0)


def test_equation_rejects_a_product_of_three_bits(tmp_path: Path) -> None:
    path = _write_equation(tmp_path / "cubic.yaml", "x[0] * x[1] * x[2]", count=3)
    with pytest.raises(ValueError, match="two variables"):
        QuboBuilder.load(path).build()


def test_annealing_rejects_an_increasing_temperature() -> None:
    with pytest.raises(ValueError, match="initial_temperature"):
        AnnealingSettings(
            num_samples=1,
            num_sweeps=1,
            initial_temperature=0.1,
            final_temperature=1.0,
            seed=1,
        )


def _assert_sample_energies(qubo: Qubo, result: QuboSolveResult) -> None:
    samples = result.samples
    energies = result.sample_energies
    for sample, energy in zip(samples, energies, strict=True):
        assert qubo.energy(sample) == pytest.approx(float(energy))
    assert result.energy == pytest.approx(float(np.min(energies)))


def _route_kwargs(capacity: float = 5.0) -> dict[str, object]:
    return {
        "demands": np.array([3.0, 3.0]),
        "fixed_load_a": 0.0,
        "fixed_load_b": 0.0,
        "capacity": capacity,
        "compatibility_a": np.array([0.9, 0.1]),
        "compatibility_b": np.array([0.1, 0.9]),
        "pair_prior": np.array([[0.0, 0.2], [0.2, 0.0]]),
    }


def _route_bits(assignment: tuple[int, int], slack: int, slack_bits: int) -> np.ndarray:
    bits = list(assignment)
    for _route in range(2):
        for bit in range(slack_bits):
            bits.append((slack >> bit) & 1)
    return np.asarray(bits, dtype=np.int8)


def _reorder_formula(bitstring: np.ndarray, model: QuboBuilder) -> float:
    penalty = model.penalty("lambda_customer")
    segment = (1, 2)
    k = len(segment)
    assignment = np.asarray(bitstring, dtype=np.float64).reshape(k, k)
    customer_penalty = sum((1.0 - float(row.sum())) ** 2 for row in assignment)
    position_penalty = sum((1.0 - float(column.sum())) ** 2 for column in assignment.T)
    travel = 0.0
    for index, customer in enumerate(segment):
        travel += float(DISTANCE[0, customer]) * assignment[index, 0]
        travel += float(DISTANCE[customer, 3]) * assignment[index, k - 1]
    for slot in range(k - 1):
        for left, left_customer in enumerate(segment):
            for right, right_customer in enumerate(segment):
                if left == right:
                    continue
                travel += (
                    float(DISTANCE[left_customer, right_customer])
                    * assignment[left, slot]
                    * assignment[right, slot + 1]
                )
    return travel + penalty * customer_penalty + penalty * position_penalty


def _two_route_formula(bitstring: np.ndarray, model: QuboBuilder) -> float:
    demands = np.array([3.0, 3.0])
    scores_a = np.array([0.9, 0.1])
    scores_b = np.array([0.1, 0.9])
    prior = np.array([[0.0, 0.2], [0.2, 0.0]])
    m = demands.shape[0]
    y = np.asarray(bitstring[:m], dtype=np.float64)
    slack_a = _slack_value(bitstring[m : m + model.slack_bits])
    slack_b = _slack_value(bitstring[m + model.slack_bits :])
    load_a = float(demands @ y)
    load_b = float(demands @ (1.0 - y))
    capacity = model.penalty("lambda_capacity") * (
        (load_a + slack_a - 5.0) ** 2 + (load_b + slack_b - 5.0) ** 2
    )
    cmd = 0.0
    for index in range(m):
        cmd += float(scores_a[index]) * y[index] + float(scores_b[index]) * (1.0 - y[index])
    pair = 0.0
    for left in range(m):
        for right in range(left + 1, m):
            same = y[left] * y[right] + (1.0 - y[left]) * (1.0 - y[right])
            pair += float(prior[left, right]) * same
    return capacity - model.penalty("alpha") * cmd - model.penalty("beta") * pair


def _slack_value(bits: np.ndarray) -> float:
    return float(sum(int(bit) * (2**index) for index, bit in enumerate(bits)))


def _write_equation(path: Path, equation: str, count: int = 2) -> Path:
    path.write_text(
        f"seed: 1\n"
        f"bits:\n"
        f"  x: [{count}]\n"
        f'equation: "{equation}"\n'
        "qaoa:\n"
        "  depth: 1\n"
        "  shots: 1\n"
        "  maxiter: 1\n"
        "  restarts: 1\n"
        "annealing:\n"
        "  num_samples: 1\n"
        "  num_sweeps: 1\n"
        "  initial_temperature: 1.0\n"
        "  final_temperature: 0.1\n",
        encoding="utf-8",
    )
    return path
