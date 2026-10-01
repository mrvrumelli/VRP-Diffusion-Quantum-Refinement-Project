"""Tests for diffusion-biased QUBO terms (task Q1.5)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from test_quantum_neighborhoods import make_instance
from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    ReorderSubproblem,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
)
from vrp_diffusion_quantum.quantum.qubo import solve_exact
from vrp_diffusion_quantum.quantum.qubo_bias import (
    DiffusionBiasConfig,
    build_biased_exchange_qubo,
    build_biased_reorder_qubo,
)
from vrp_diffusion_quantum.quantum.qubo_exchange import build_exchange_qubo, decode_exchange
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    build_reorder_qubo,
    decode_reorder,
    encode_reorder,
)


def tied_reorder() -> tuple[CVRPInstance, list[list[int]], np.ndarray]:
    # Fixed customers 0 at (0, 0) and 3 at (2, 0); free customers 1 at (1, 1) and 2 at (1, -1).
    # Both orders of the free pair cost the same, so only the prior can break the tie.
    instance = make_instance([(0.0, 0.0), (1.0, 1.0), (1.0, -1.0), (2.0, 0.0)], depot=(1.0, -3.0))
    routes = [[0, 2, 1, 3]]
    m_prob = np.full((4, 4), 0.5)
    m_prob[0, 1] = m_prob[1, 0] = 0.9  # 0 should precede 1
    m_prob[0, 2] = m_prob[2, 0] = 0.1
    m_prob[2, 3] = m_prob[3, 2] = 0.9  # 2 should precede 3
    m_prob[1, 3] = m_prob[3, 1] = 0.1
    np.fill_diagonal(m_prob, 0.0)
    return instance, routes, m_prob


def tied_reorder_subproblem() -> tuple[ReorderSubproblem, np.ndarray]:
    instance, routes, m_prob = tied_reorder()
    neighborhood = Neighborhood("low_confidence_edges", "reorder", (0,), (2, 1), 1.0, (1, 3))
    return extract_reorder_subproblem(instance, routes, neighborhood), m_prob


@pytest.mark.parametrize(
    "config",
    [
        DiffusionBiasConfig(),
        DiffusionBiasConfig(enabled=True, alpha=0.0),
        DiffusionBiasConfig(alpha=3.0),
    ],
)
def test_inactive_bias_leaves_the_qubos_unchanged(config: DiffusionBiasConfig) -> None:
    subproblem, m_prob = tied_reorder_subproblem()
    plain = build_reorder_qubo(subproblem)
    biased = build_biased_reorder_qubo(subproblem, m_prob, config)
    assert np.array_equal(plain.qubo.matrix, biased.qubo.matrix)
    assert plain.qubo.offset == biased.qubo.offset


def test_reorder_bias_breaks_a_distance_tie_toward_the_prior() -> None:
    subproblem, m_prob = tied_reorder_subproblem()
    assert subproblem.path_cost((1, 2)) == pytest.approx(subproblem.path_cost((2, 1)))
    config = DiffusionBiasConfig(enabled=True, alpha=0.2)
    biased = build_biased_reorder_qubo(subproblem, m_prob, config)
    assert decode_reorder(biased, solve_exact(biased.qubo).x) == (1, 2)
    assert biased.qubo.penalty_weights["diffusion_bias_alpha"] == 0.2


def test_biased_permutation_energy_is_path_cost_plus_scaled_pair_costs() -> None:
    subproblem, m_prob = tied_reorder_subproblem()
    alpha = 0.7
    biased = build_biased_reorder_qubo(subproblem, m_prob, DiffusionBiasConfig(True, alpha))
    distances = biased.subproblem.distances
    scale = alpha * distances[distances > 0].mean()
    for order in itertools.permutations(biased.subproblem.customers):
        path = [0, *order, 3]
        bias = sum(1.0 - m_prob[a, b] for a, b in itertools.pairwise(path))
        expected = biased.subproblem.path_cost(order) + scale * bias
        assert biased.qubo.energy(encode_reorder(biased, order)) == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["probability", "confidence"])
def test_large_alpha_keeps_reorder_ground_state_a_permutation(mode: str) -> None:
    rng = np.random.default_rng(5)
    instance = make_instance([tuple(p) for p in rng.random((5, 2))])
    route = [0, 1, 2, 3, 4]
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), (1, 2, 3), 1.0, (1, 4))
    subproblem = extract_reorder_subproblem(instance, [route], neighborhood)
    m_prob = rng.random((5, 5))
    m_prob = (m_prob + m_prob.T) / 2
    config = DiffusionBiasConfig(True, alpha=25.0, mode=mode)  # type: ignore[arg-type]
    biased = build_biased_reorder_qubo(subproblem, m_prob, config)
    assert decode_reorder(biased, solve_exact(biased.qubo).x) is not None


def tied_exchange() -> tuple[CVRPInstance, list[list[int]], np.ndarray]:
    # Route 0 fixed customer 0 at (1, 0); route 1 fixed customer 1 at (-1, 0); movable customer 2
    # at (0, 1) is equally cheap to insert in either route.
    instance = make_instance([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)], demands=[1, 1, 1], capacity=5)
    routes = [[0], [1, 2]]
    m_prob = np.full((3, 3), 0.5)
    m_prob[2, 0] = m_prob[0, 2] = 0.9
    m_prob[2, 1] = m_prob[1, 2] = 0.1
    np.fill_diagonal(m_prob, 0.0)
    return instance, routes, m_prob


def test_exchange_bias_moves_a_tied_customer_toward_its_predicted_route() -> None:
    instance, routes, m_prob = tied_exchange()
    neighborhood = Neighborhood("uncertain_m", "exchange", (0, 1), (2,), 1.0)
    subproblem = extract_exchange_subproblem(instance, routes, neighborhood)
    plain = build_exchange_qubo(instance, subproblem)
    assert plain.insertion_costs[0, 0] == pytest.approx(plain.insertion_costs[0, 1])
    biased = build_biased_exchange_qubo(
        instance,
        subproblem,
        m_prob,
        DiffusionBiasConfig(enabled=True, alpha=0.5),
    )
    assert decode_exchange(biased, solve_exact(biased.qubo).x) == (1,)
    off = build_biased_exchange_qubo(instance, subproblem, m_prob, DiffusionBiasConfig())
    assert np.array_equal(off.qubo.matrix, plain.qubo.matrix)


def test_large_alpha_keeps_exchange_capacity_penalty_dominant() -> None:
    xy = [(2.0, 0.0), (2.0, 1.0), (-2.0, 0.0), (-2.0, 1.0), (2.5, 0.5)]
    instance = make_instance(xy, demands=[1, 2, 1, 1, 2], capacity=4)
    routes = [[0, 1], [2, 3, 4]]
    neighborhood = Neighborhood("two_route_exchange", "exchange", (0, 1), (4, 1, 3), 1.0)
    subproblem = extract_exchange_subproblem(instance, routes, neighborhood)
    # A prior that wants every movable customer in the eastern route, which is infeasible.
    m_prob = np.full((5, 5), 0.95)
    m_prob[[4, 1, 3], 2] = m_prob[2, [4, 1, 3]] = 0.0
    np.fill_diagonal(m_prob, 0.0)
    biased = build_biased_exchange_qubo(
        instance, subproblem, m_prob, DiffusionBiasConfig(enabled=True, alpha=20.0)
    )
    qubo = biased.qubo
    states = np.asarray(list(itertools.product((0, 1), repeat=qubo.num_variables)), dtype=float)
    energies = qubo.energies(states)
    feasible_min: dict[tuple[int, ...], float] = {}
    infeasible = np.inf
    for state, energy in zip(states, energies, strict=True):
        assignment = decode_exchange(biased, state.astype(int).tolist())
        if subproblem.is_feasible(assignment):
            feasible_min[assignment] = min(feasible_min.get(assignment, np.inf), float(energy))
        else:
            infeasible = min(infeasible, float(energy))
    assert infeasible > max(feasible_min.values())
    assert subproblem.is_feasible(decode_exchange(biased, solve_exact(qubo).x))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        DiffusionBiasConfig(enabled=True, alpha=-1.0)
    with pytest.raises(ValueError, match="mode"):
        DiffusionBiasConfig(enabled=True, alpha=1.0, mode="entropy")  # type: ignore[arg-type]
    assert not DiffusionBiasConfig(enabled=False, alpha=2.0).active
    assert DiffusionBiasConfig(enabled=True, alpha=2.0).active
