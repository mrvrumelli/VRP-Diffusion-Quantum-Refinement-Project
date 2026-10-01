"""Tests for refinement neighborhood selection and subproblem extraction (task Q1.2)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    NeighborhoodConfig,
    apply_exchange,
    apply_reorder,
    evaluate_exchange,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
    route_affinities,
    select_high_cost_routes,
    select_low_confidence_edges,
    select_neighborhoods,
    select_two_route_exchanges,
    select_uncertain_customers,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost


def make_instance(
    customer_xy: list[tuple[float, float]],
    demands: list[float] | None = None,
    *,
    capacity: float = 100.0,
    depot: tuple[float, float] = (0.0, 0.0),
) -> CVRPInstance:
    n = len(customer_xy)
    customer_demands = demands if demands is not None else [1.0] * n
    return CVRPInstance(
        coords=np.asarray([depot, *customer_xy], dtype=np.float64),
        demands=np.asarray([0.0, *customer_demands], dtype=np.float64),
        capacity=capacity,
        depot_index=0,
        instance_id="unit",
        n_customers=n,
        seed=0,
        generator_settings={},
    )


def block_matrix(routes: list[list[int]], n: int, *, inside: float, outside: float) -> np.ndarray:
    m_prob = np.full((n, n), outside)
    for route in routes:
        for a in route:
            for b in route:
                m_prob[a, b] = inside
    np.fill_diagonal(m_prob, 0.0)
    return m_prob


def test_high_cost_route_ranks_by_cost_per_customer_and_windows_long_routes() -> None:
    # Route 0: two customers near the depot. Route 1: a long line with one far detour (index 5).
    xy = [(0.1, 0.0), (0.2, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 5.0), (4.0, 0.0)]
    instance = make_instance(xy)
    routes = [[0, 1], [2, 3, 4, 5, 6]]
    selected = select_high_cost_routes(instance, routes, NeighborhoodConfig(max_reorder_size=3))

    assert [n.route_indices for n in selected] == [(1,), (0,)]
    top = selected[0]
    assert top.kind == "reorder" and top.segment is not None
    start, end = top.segment
    assert end - start == 3
    assert top.customers == tuple(routes[1][start:end])
    assert 5 in top.customers  # the detour customer sits on the most expensive edges
    assert selected[1].customers == (0, 1)


def test_uncertain_customers_pair_routes_by_affinity_margin() -> None:
    xy = [(float(i), 0.0) for i in range(6)]
    instance = make_instance(xy)
    routes = [[0, 1, 2], [3, 4, 5]]
    m_prob = block_matrix(routes, 6, inside=0.9, outside=0.1)
    m_prob[2, [3, 4, 5]] = m_prob[[3, 4, 5], 2] = 0.8  # customer 2 is torn between both routes
    m_prob[2, [0, 1]] = m_prob[[0, 1], 2] = 0.7

    assert route_affinities(m_prob, routes, 2) == pytest.approx([0.7, 0.8])
    selected = select_uncertain_customers(instance, routes, m_prob)
    assert len(selected) == 1
    assert selected[0].route_indices == (0, 1)
    assert selected[0].kind == "exchange"
    assert selected[0].customers[0] == 2
    assert 4 not in selected[0].customers  # confident customers are not movable


def test_low_confidence_edges_center_on_weak_links() -> None:
    xy = [(float(i), 0.0) for i in range(6)]
    instance = make_instance(xy)
    routes = [[0, 1, 2, 3, 4, 5]]
    m_prob = block_matrix(routes, 6, inside=0.9, outside=0.1)
    m_prob[3, 4] = m_prob[4, 3] = 0.2
    selected = select_low_confidence_edges(
        instance, routes, m_prob, NeighborhoodConfig(max_reorder_size=4)
    )
    assert len(selected) == 1
    assert selected[0].segment == (2, 6)
    assert selected[0].customers == (2, 3, 4, 5)
    assert selected[0].score == pytest.approx(0.3)


def test_two_route_exchange_prefers_adjacent_routes() -> None:
    # Routes 0 and 1 face each other across x = 0.5; route 2 is far away.
    xy = [(0.4, 1.0), (0.3, 2.0), (0.6, 1.0), (0.7, 2.0), (9.0, 9.0), (9.5, 9.0)]
    instance = make_instance(xy)
    routes = [[0, 1], [2, 3], [4, 5]]
    selected = select_two_route_exchanges(
        instance, routes, NeighborhoodConfig(max_exchange_size=3, exchange_route_pairs_per_route=1)
    )
    assert selected[0].route_indices == (0, 1)
    assert selected[0].customers == (0, 2, 1)  # alternates, nearest to the other route first
    for neighborhood in selected:
        allowed = set(routes[neighborhood.route_indices[0]]) | set(
            routes[neighborhood.route_indices[1]]
        )
        assert set(neighborhood.customers) <= allowed


def test_select_neighborhoods_requires_matrix_for_matrix_types() -> None:
    instance = make_instance([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
    routes = [[0, 1], [2]]
    with pytest.raises(ValueError, match="require m_prob"):
        select_neighborhoods(instance, routes, types=("uncertain_m",))
    selected = select_neighborhoods(
        instance, routes, types=("high_cost_route", "two_route_exchange")
    )
    assert {n.neighborhood_type for n in selected} == {"high_cost_route", "two_route_exchange"}


def test_reorder_path_cost_change_equals_route_cost_change_for_every_order() -> None:
    rng = np.random.default_rng(3)
    instance = make_instance([tuple(p) for p in rng.random((7, 2))])
    routes = [[0, 1, 2, 3, 4, 5], [6]]
    neighborhood = Neighborhood(
        "high_cost_route", "reorder", (0,), tuple(routes[0][1:5]), 1.0, (1, 5)
    )
    subproblem = extract_reorder_subproblem(instance, routes, neighborhood)
    assert subproblem.start_node == 0 and subproblem.end_node == 5
    base = route_cost(instance, routes)
    for order in itertools.permutations(subproblem.customers):
        updated = apply_reorder(routes, subproblem, order)
        delta = route_cost(instance, updated) - base
        assert delta == pytest.approx(subproblem.path_cost(order) - subproblem.current_cost)


def test_full_route_reorder_uses_depot_at_both_ends() -> None:
    instance = make_instance([(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    routes = [[0, 2, 1]]
    subproblem = extract_reorder_subproblem(
        instance, routes, Neighborhood("high_cost_route", "reorder", (0,), (0, 2, 1), 1.0, (0, 3))
    )
    assert subproblem.start_node is None and subproblem.end_node is None
    assert subproblem.current_cost == pytest.approx(route_cost(instance, routes))
    assert subproblem.path_cost((0, 1, 2)) == pytest.approx(4.0)


def test_reorder_rejects_a_stale_neighborhood() -> None:
    instance = make_instance([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
    stale = Neighborhood("high_cost_route", "reorder", (0,), (1, 0), 1.0, (0, 2))
    with pytest.raises(ValueError, match="no longer matches"):
        extract_reorder_subproblem(instance, [[0, 1, 2]], stale)


def test_exchange_extract_loads_and_apply_round_trip() -> None:
    xy = [(1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 1.0)]
    instance = make_instance(xy, demands=[3.0, 4.0, 2.0, 5.0, 6.0], capacity=12.0)
    routes = [[0, 1, 4], [2, 3]]
    neighborhood = Neighborhood("two_route_exchange", "exchange", (0, 1), (4, 2), 1.0)
    subproblem = extract_exchange_subproblem(instance, routes, neighborhood)

    assert subproblem.fixed == ((0, 1), (3,))
    assert subproblem.initial_assignment == (1, 0)
    assert subproblem.fixed_loads == (7.0, 5.0)
    assert subproblem.loads((1, 0)) == (13.0, 7.0)
    assert not subproblem.is_feasible((1, 0))  # the current split is over capacity
    assert subproblem.is_feasible((0, 1))
    assert not subproblem.is_feasible((1, 1))

    cost, first, second = evaluate_exchange(instance, subproblem, (0, 1))
    updated = apply_exchange(routes, subproblem, first, second)
    assert sorted(updated[0]) == [0, 1, 2] and sorted(updated[1]) == [3, 4]
    assert cost == pytest.approx(route_cost(instance, updated))
    with pytest.raises(ValueError, match="exactly the subproblem"):
        apply_exchange(routes, subproblem, [0, 1], [3])
