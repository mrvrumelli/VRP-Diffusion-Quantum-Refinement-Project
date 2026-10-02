"""QAOA on the Qiskit statevector, or the same circuit with Aer depolarizing noise."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize

from vrp_diffusion_quantum.quantum.qubo import QaoaSettings, Qubo, QuboSolveResult, make_result

logger = logging.getLogger(__name__)

MAX_QAOA_VARIABLES = 12
SOLVER_NAME = "qaoa"
_BASIS_GATES = ["rz", "sx", "cx"]


def solve_qaoa(qubo: Qubo, settings: QaoaSettings) -> QuboSolveResult:
    """Minimize `qubo` with depth-p QAOA and return shot samples plus their energies."""
    n = qubo.num_variables
    if n > MAX_QAOA_VARIABLES:
        raise ValueError(f"QAOA simulator supports at most {MAX_QAOA_VARIABLES} variables, got {n}")
    # Qiskit publishes no stubs, so the operator and circuit stay object at the boundary.
    hamiltonian = cast(Any, _hamiltonian(qubo)) / qubo.energy_scale
    ansatz = _ansatz(hamiltonian, settings.depth)
    simulator = _simulator(settings)
    rng = np.random.default_rng(settings.seed)
    best_params = np.zeros(2 * settings.depth, dtype=np.float64)
    best_expectation = float("inf")

    for restart in range(settings.restarts):
        initial = _initial_params(rng, settings.depth)
        optimized = minimize(
            _expectation,
            initial,
            args=(ansatz, hamiltonian, simulator),
            method="COBYLA",
            options={"maxiter": settings.maxiter, "rhobeg": 0.2},
        )
        params = np.asarray(optimized.x, dtype=np.float64)
        expectation = float(optimized.fun)
        logger.info(
            "solver_name=%s restart=%d expectation=%.6f seed=%d noise=%s noise_probability=%.6f",
            SOLVER_NAME,
            restart,
            expectation * qubo.energy_scale,
            settings.seed,
            settings.noise,
            settings.noise_probability,
        )
        if expectation < best_expectation:
            best_expectation = expectation
            best_params = params

    probabilities = _probabilities(ansatz, best_params, simulator)
    drawn = rng.choice(probabilities.shape[0], size=settings.shots, replace=True, p=probabilities)
    mode = int(np.argmax(probabilities))
    indices = drawn if mode in drawn else np.concatenate([drawn, np.array([mode], dtype=np.int64)])
    samples = _bitstrings(indices, n)
    sample_energies = qubo.energies()[indices]
    return make_result(
        solver_name=SOLVER_NAME,
        qubo=qubo,
        samples=samples,
        sample_energies=sample_energies,
        seed=settings.seed,
        qaoa_depth=settings.depth,
        expectation=best_expectation * qubo.energy_scale,
    )


def _initial_params(rng: np.random.Generator, depth: int) -> np.ndarray:
    # The QAOA circuit binds every beta, then every gamma.
    betas = rng.uniform(0.0, np.pi, size=depth)
    gammas = rng.uniform(0.0, 0.2, size=depth)
    return np.concatenate([betas, gammas])


def _expectation(
    params: np.ndarray,
    ansatz: object,
    hamiltonian: object,
    simulator: object | None,
) -> float:
    circuit = _circuit(ansatz, params)
    if simulator is None:
        from qiskit.quantum_info import Statevector

        value = Statevector.from_instruction(circuit).expectation_value(hamiltonian)
    else:
        from qiskit.quantum_info import DensityMatrix

        value = DensityMatrix(_density_matrix(circuit, simulator)).expectation_value(hamiltonian)
    return float(np.real(value))


def _probabilities(ansatz: object, params: np.ndarray, simulator: object | None) -> np.ndarray:
    circuit = _circuit(ansatz, params)
    if simulator is None:
        from qiskit.quantum_info import Statevector

        raw = Statevector.from_instruction(circuit).probabilities()
    else:
        from qiskit.quantum_info import DensityMatrix

        raw = DensityMatrix(_density_matrix(circuit, simulator)).probabilities()
    probabilities = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("QAOA returned an empty distribution")
    probabilities /= total
    return cast(np.ndarray, probabilities)


def _circuit(ansatz: object, params: np.ndarray) -> object:
    from qiskit import transpile

    bound = cast(Any, ansatz).assign_parameters(np.asarray(params, dtype=np.float64))
    # QAOA and Pauli evolution are composite instructions. Expand them before
    # transpiling onto the basis that carries the noise model.
    expanded = bound.decompose(reps=4)
    return cast(object, transpile(expanded, basis_gates=_BASIS_GATES, optimization_level=1))


def _ansatz(hamiltonian: object, depth: int) -> object:
    from qiskit.circuit.library import qaoa_ansatz

    return cast(object, qaoa_ansatz(hamiltonian, reps=depth))


def _simulator(settings: QaoaSettings) -> object | None:
    if settings.noise == "noiseless":
        return None
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise = NoiseModel()
    probability = settings.noise_probability
    noise.add_all_qubit_quantum_error(depolarizing_error(probability, 1), ["rz", "sx"])
    noise.add_all_qubit_quantum_error(depolarizing_error(probability, 2), ["cx"])
    return cast(object, AerSimulator(method="density_matrix", noise_model=noise))


def _density_matrix(circuit: object, simulator: object) -> object:
    measured = cast(Any, circuit).copy()
    measured.save_density_matrix()
    result = cast(Any, simulator).run(measured).result()
    return cast(object, result.data(0)["density_matrix"])


def _hamiltonian(qubo: Qubo) -> object:
    from qiskit.quantum_info import SparsePauliOp

    n = qubo.num_variables
    coeffs: dict[str, float] = {_z_label(n): float(qubo.offset)}
    for qubit, coeff in enumerate(np.asarray(qubo.linear, dtype=np.float64)):
        weight = float(coeff)
        if weight == 0.0:
            continue
        _add_coeff(coeffs, _z_label(n), weight / 2.0)
        _add_coeff(coeffs, _z_label(n, qubit), -weight / 2.0)
    for i, j, coeff in qubo.quadratic:
        weight = float(coeff)
        if weight == 0.0:
            continue
        _add_coeff(coeffs, _z_label(n), weight / 4.0)
        _add_coeff(coeffs, _z_label(n, i), -weight / 4.0)
        _add_coeff(coeffs, _z_label(n, j), -weight / 4.0)
        _add_coeff(coeffs, _z_label(n, i, j), weight / 4.0)
    terms = [(label, weight) for label, weight in coeffs.items() if weight != 0.0]
    if not terms:
        terms = [(_z_label(n), 0.0)]
    return cast(object, SparsePauliOp.from_list(terms).simplify())


def _add_coeff(coeffs: dict[str, float], label: str, weight: float) -> None:
    coeffs[label] = coeffs.get(label, 0.0) + weight


def _z_label(n: int, *qubits: int) -> str:
    chars = ["I"] * n
    for qubit in qubits:
        chars[n - 1 - qubit] = "Z"
    return "".join(chars)


def _bitstrings(indices: np.ndarray, n: int) -> np.ndarray:
    return ((indices[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int8)
