"""QAOA screening of small refinement QUBOs in statevector simulation.

Runs in the isolated quantum environment (``outputs/quantum_venv``, Qiskit stack only); it does
not import the project package. Input is a JSON file written by ``export_qubo_instances.py``: one
entry per QUBO with its upper-triangular matrix, offset, labels and the exact minimum energy.

For every QUBO and every depth ``p``: the QUBO is converted to an Ising Hamiltonian with
qiskit-optimization and a ``QAOAAnsatz`` circuit is built. Its parameters are optimised with
COBYLA on the exact expectation (several seeded restarts) using a fast NumPy statevector
simulation of the same circuit; once per QUBO that simulation is checked against Qiskit's
``Statevector`` of the ansatz and the run aborts on any disagreement above 1e-8. Finally
``--shots`` bitstrings are sampled from the optimised state. The output records the
optimal-state probability, the best sampled energy relative to the exact minimum, and the sample
counts for decoding in the main environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from qiskit_optimization import QuadraticProgram
from scipy.optimize import minimize


def to_program(matrix: np.ndarray, offset: float, labels: list[str]) -> QuadraticProgram:
    program = QuadraticProgram("refinement_qubo")
    names = [f"x{i}" for i in range(len(labels))]
    for name in names:
        program.binary_var(name)
    linear = {names[i]: float(matrix[i, i]) for i in range(len(names)) if matrix[i, i] != 0.0}
    quadratic = {
        (names[i], names[j]): float(matrix[i, j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if matrix[i, j] != 0.0
    }
    program.minimize(constant=float(offset), linear=linear, quadratic=quadratic)
    return program


def bit_energies(matrix: np.ndarray, offset: float) -> np.ndarray:
    """Energy of every computational basis state, indexed like Statevector probabilities."""
    n = matrix.shape[0]
    codes = np.arange(1 << n, dtype=np.int64)
    # Qiskit orders qubit 0 as the least significant bit of the basis index.
    bits = ((codes[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    return np.einsum("si,ij,sj->s", bits, matrix, bits) + offset


def simulate(theta: np.ndarray, energies: np.ndarray, n: int, reps: int) -> np.ndarray:
    """Exact QAOA probabilities: |+>^n, then per layer exp(-i g E) and RX(2b) on every qubit.

    ``theta`` is ordered like ``QAOAAnsatz.parameters`` (all betas, then all gammas). The cost
    phase uses the QUBO energy directly, which equals the Ising Hamiltonian up to a global phase.
    """
    betas, gammas = theta[:reps], theta[reps:]
    state = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)
    for beta, gamma in zip(betas, gammas, strict=True):
        state = state * np.exp(-1j * gamma * energies)
        cos, sin = np.cos(beta), -1j * np.sin(beta)
        for qubit in range(n):
            view = state.reshape(1 << (n - qubit - 1), 2, 1 << qubit)
            zero, one = view[:, 0, :].copy(), view[:, 1, :].copy()
            view[:, 0, :] = cos * zero + sin * one
            view[:, 1, :] = sin * zero + cos * one
    return np.abs(state) ** 2


def run_qaoa(entry: dict, reps: int, restarts: int, shots: int, seed: int) -> dict:
    matrix = np.asarray(entry["matrix"], dtype=np.float64)
    offset = float(entry["offset"])
    n = matrix.shape[0]
    program = to_program(matrix, offset, entry["labels"])
    hamiltonian, ising_offset = program.to_ising()
    energies = bit_energies(matrix, offset)
    exact = float(energies.min())
    optimal = np.isclose(energies, exact, atol=1e-9)
    scale = float(np.abs(matrix).max()) or 1.0
    rng = np.random.default_rng(seed)

    # Cross-check the fast simulator against Qiskit's QAOAAnsatz statevector once per QUBO.
    ansatz = QAOAAnsatz(hamiltonian, reps=reps).decompose(reps=3)
    probe = np.concatenate([rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi / scale, reps)])
    qiskit_probs = Statevector.from_instruction(ansatz.assign_parameters(probe)).probabilities()
    max_difference = float(np.abs(qiskit_probs - simulate(probe, energies, n, reps)).max())
    if max_difference > 1e-8:
        raise RuntimeError(f"fast simulator disagrees with Qiskit by {max_difference:.2e}")

    def expectation(theta: np.ndarray) -> float:
        return float(simulate(theta, energies, n, reps) @ energies)

    best_theta, best_value, evaluations = probe, np.inf, 0
    started = time.perf_counter()
    for _ in range(restarts):
        initial = np.concatenate([rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi / scale, reps)])
        result = minimize(expectation, initial, method="COBYLA", options={"maxiter": 300})
        evaluations += int(result.nfev)
        if result.fun < best_value:
            best_theta, best_value = result.x, float(result.fun)
    runtime = time.perf_counter() - started
    final = simulate(best_theta, energies, n, reps)
    sample_rng = np.random.default_rng(seed + 1)
    drawn = sample_rng.choice(len(final), size=shots, p=final / final.sum())
    unique, counts = np.unique(drawn, return_counts=True)
    sampled_energies = energies[unique]
    samples = [
        {
            "x": [int((int(code) >> i) & 1) for i in range(n)],
            "count": int(count),
            "energy": float(energy),
        }
        for code, count, energy in zip(unique, counts, sampled_energies, strict=True)
    ]
    samples.sort(key=lambda item: item["energy"])
    return {
        "id": entry["id"],
        "kind": entry["kind"],
        "num_qubits": n,
        "reps": reps,
        "exact_energy": exact,
        "expected_energy": best_value,
        "optimal_probability": float(final[optimal].sum()),
        "best_sampled_energy": float(sampled_energies.min()),
        "best_sample_is_optimal": bool(np.isclose(sampled_energies.min(), exact, atol=1e-9)),
        "random_guess_optimal_probability": float(optimal.mean()),
        "energy_evaluations": evaluations,
        "runtime_seconds": runtime,
        "qiskit_crosscheck_max_difference": max_difference,
        "ising_offset": float(ising_offset),
        "samples": samples[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    results = []
    done = set()
    if args.output.exists():
        results = json.loads(args.output.read_text())["results"]
        done = {(row["id"], row["reps"]) for row in results}
    for entry in payload["qubos"]:
        for reps in args.reps:
            if (entry["id"], reps) in done:
                continue
            row = run_qaoa(entry, reps, args.restarts, args.shots, args.seed)
            results.append(row)
            print(
                f"{entry['id']} p={reps} n={row['num_qubits']} "
                f"P(opt)={row['optimal_probability']:.3f} "
                f"best_is_opt={row['best_sample_is_optimal']} t={row['runtime_seconds']:.1f}s",
                flush=True,
            )
            args.output.write_text(json.dumps({"source": str(args.input), "results": results}))
    print("QAOA_DONE")


if __name__ == "__main__":
    main()
