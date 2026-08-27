"""Tests for the autoregressive dual-pointer CVRP decoder (P4.3/P4.4/P4.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import CVRPBatch, collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.decoder import (
    POMO_START_NODE_CAP,
    CVRPPolicy,
    DecoderRollout,
    DualPointerDecoder,
    PolicyEncoding,
    _pairwise_savings_bias,
    actions_to_routes,
    build_decoder_local_adjacency,
    paper_num_starts,
    select_start_nodes,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

EMBEDDING_DIM = 32


def _example(n_customers: int, *, seed: int = 0, capacity: float = 10.0) -> CVRPExample:
    """Build a small CVRP example whose capacity forces several routes."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
    demands = np.concatenate([[0.0], rng.integers(1, 5, size=n_customers).astype(np.float64)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
        depot_index=0,
        instance_id=f"cvrp{n_customers}_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    routes: list[list[int]] = []
    current: list[int] = []
    load = 0.0
    for customer in range(n_customers):
        demand = float(demands[customer + 1])
        if load + demand > capacity and current:
            routes.append(current)
            current = []
            load = 0.0
        current.append(customer)
        load += demand
    if current:
        routes.append(current)
    solution = LabeledSolution(
        routes=routes,
        cost=1.0,
        num_vehicles=len(routes),
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def _batch(sizes: list[int], *, seed: int = 0, capacity: float = 10.0) -> CVRPBatch:
    return collate_batch(
        [_example(size, seed=seed + i, capacity=capacity) for i, size in enumerate(sizes)]
    )


def _policy(**overrides: object) -> CVRPPolicy:
    torch.manual_seed(0)
    kwargs: dict[str, object] = {
        "embedding_dim": EMBEDDING_DIM,
        "global_num_layers": 1,
        "global_num_heads": 4,
        "local_num_layers": 1,
        "local_num_heads": 4,
        "feed_forward_dim": 64,
        "decoder_num_heads": 4,
    }
    kwargs.update(overrides)
    return CVRPPolicy(**kwargs)  # type: ignore[arg-type]


def _encode(
    policy: CVRPPolicy, batch: CVRPBatch, *, m_hat: torch.Tensor | None = None
) -> PolicyEncoding:
    prior = batch.constraint_matrix if m_hat is None else m_hat
    return policy.encode(
        batch.coords,
        batch.demands,
        batch.capacity,
        batch.depot_index,
        batch.node_mask,
        m_hat=prior,
        customer_node_indices=batch.customer_node_indices,
        customer_mask=batch.customer_mask,
    )


def _assert_feasible(
    batch: CVRPBatch,
    rollout: DecoderRollout,
    examples: list[CVRPExample],
    repeats: int = 1,
) -> list[list[list[int]]]:
    routes = actions_to_routes(
        rollout.actions,
        batch.depot_index.repeat_interleave(repeats),
        batch.customer_node_indices.repeat_interleave(repeats, dim=0),
        batch.customer_mask.repeat_interleave(repeats, dim=0),
    )
    for index, route_set in enumerate(routes):
        instance = examples[index // repeats].instance
        report = validate_routes(instance, route_set)
        assert report.feasible, report.violations
        assert math.isclose(
            route_cost(instance, route_set),
            float(rollout.cost[index]),
            rel_tol=1e-4,
            abs_tol=1e-4,
        )
    return routes


def test_greedy_rollout_produces_feasible_routes() -> None:
    examples = [_example(size, seed=size) for size in (6, 9, 12)]
    batch = collate_batch(examples)
    policy = _policy()
    with torch.no_grad():
        rollout = policy.rollout(_encode(policy, batch), decode_mode="greedy", num_starts=1)

    assert rollout.actions.shape[0] == len(examples)
    assert torch.isfinite(rollout.cost).all()
    assert (rollout.cost > 0).all()
    _assert_feasible(batch, rollout, examples)


def test_mixed_size_padded_batch_stays_feasible() -> None:
    examples = [_example(4, seed=1), _example(11, seed=2), _example(7, seed=3)]
    batch = collate_batch(examples)
    policy = _policy()
    with torch.no_grad():
        rollout = policy.rollout(_encode(policy, batch), decode_mode="greedy", num_starts=1)
    routes = _assert_feasible(batch, rollout, examples)
    for index, example in enumerate(examples):
        served = sorted(customer for route in routes[index] for customer in route)
        assert served == list(range(example.instance.n_customers))


def test_capacity_mask_forces_multiple_routes() -> None:
    example = _example(10, seed=7, capacity=6.0)
    batch = collate_batch([example])
    policy = _policy()
    with torch.no_grad():
        rollout = policy.rollout(_encode(policy, batch), decode_mode="greedy", num_starts=1)
    routes = _assert_feasible(batch, rollout, [example])[0]

    total_demand = float(example.instance.customer_demands().sum())
    assert len(routes) >= math.ceil(total_demand / example.instance.capacity)
    for route in routes:
        load = sum(float(example.instance.customer_demands()[c]) for c in route)
        assert load <= example.instance.capacity + 1e-9


def test_oversized_demand_is_rejected() -> None:
    example = _example(5, seed=4, capacity=2.0)
    batch = collate_batch([example])
    policy = _policy()
    with pytest.raises(ValueError, match="fit within the vehicle capacity"):
        policy.rollout(_encode(policy, batch), num_starts=1)


def test_multi_start_rollout_uses_distinct_starts_and_stays_feasible() -> None:
    examples = [_example(8, seed=11), _example(8, seed=12)]
    batch = collate_batch(examples)
    policy = _policy()
    num_starts = 4
    with torch.no_grad():
        rollout = policy.rollout(_encode(policy, batch), num_starts=num_starts)

    assert rollout.cost.shape == (len(examples) * num_starts,)
    first_actions = rollout.actions[:, 0].view(len(examples), num_starts)
    for row in first_actions:
        assert len(set(row.tolist())) == num_starts
    _assert_feasible(batch, rollout, examples, repeats=num_starts)


def test_sampling_is_reproducible_and_differs_from_greedy() -> None:
    batch = _batch([9], seed=21)
    policy = _policy()
    encoding = _encode(policy, batch)
    with torch.no_grad():
        first = policy.rollout(
            encoding,
            decode_mode="sampling",
            num_starts=1,
            generator=torch.Generator().manual_seed(5),
        )
        second = policy.rollout(
            encoding,
            decode_mode="sampling",
            num_starts=1,
            generator=torch.Generator().manual_seed(5),
        )
        third = policy.rollout(
            encoding,
            decode_mode="sampling",
            num_starts=1,
            generator=torch.Generator().manual_seed(6),
        )
    assert torch.equal(first.actions, second.actions)
    assert torch.allclose(first.cost, second.cost)
    assert not torch.equal(first.actions, third.actions)


def test_log_probability_is_finite_and_differentiable() -> None:
    batch = _batch([7, 7], seed=31)
    policy = _policy()
    rollout = policy.rollout(
        _encode(policy, batch),
        decode_mode="sampling",
        num_starts=1,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.isfinite(rollout.log_probability).all()
    assert (rollout.log_probability <= 0).all()
    assert torch.isfinite(rollout.entropy).all()

    loss = (rollout.cost.detach() * rollout.log_probability).mean()
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_pointer_and_bias_ablations_change_the_module_graph() -> None:
    full = DualPointerDecoder(embedding_dim=EMBEDDING_DIM, num_heads=4)
    assert full.local_pointer is not None
    assert full.global_pointer is not None
    assert full.context_perception is not None

    global_only = DualPointerDecoder(
        embedding_dim=EMBEDDING_DIM,
        num_heads=4,
        use_local_pointer=False,
        use_savings_bias=False,
        use_context_perception=False,
    )
    assert global_only.local_pointer is None
    assert global_only.context_perception is None
    assert global_only.use_savings_bias is False

    with pytest.raises(ValueError, match="at least one of"):
        DualPointerDecoder(use_local_pointer=False, use_global_pointer=False)


def test_every_ablation_variant_still_decodes_feasible_routes() -> None:
    examples = [_example(8, seed=41)]
    batch = collate_batch(examples)
    variants: list[dict[str, object]] = [
        {},
        {"use_local_encoder": False},
        {"use_local_pointer": False},
        {"use_global_pointer": False},
        {"use_savings_bias": False},
        {"use_context_perception": False},
        {"use_local_encoder": False, "use_local_pointer": False},
    ]
    for overrides in variants:
        policy = _policy(**overrides)
        with torch.no_grad():
            rollout = policy.rollout(_encode(policy, batch), decode_mode="greedy", num_starts=1)
        _assert_feasible(batch, rollout, examples)


def test_savings_bias_matches_clarke_wright_savings() -> None:
    batch = _batch([5], seed=51)
    bias = _pairwise_savings_bias(
        batch.coords,
        batch.depot_index,
        epsilon=1e-2,
        depot_bias=torch.tensor(-0.75),
    )
    coords = batch.coords[0].numpy()
    depot = int(batch.depot_index[0])
    for i in range(1, coords.shape[0]):
        for j in range(1, coords.shape[0]):
            savings = (
                float(np.linalg.norm(coords[depot] - coords[i]))
                + float(np.linalg.norm(coords[depot] - coords[j]))
                - float(np.linalg.norm(coords[i] - coords[j]))
            )
            assert bias[0, i, j] == pytest.approx(math.log(max(savings, 1e-2)), abs=1e-5)
        assert bias[0, i, depot] == pytest.approx(-0.75)


def test_savings_bias_changes_the_action_distribution() -> None:
    batch = _batch([12], seed=51)
    with_bias = _policy(use_savings_bias=True)
    without_bias = _policy(use_savings_bias=False)
    with torch.no_grad():
        biased = with_bias.rollout(_encode(with_bias, batch), decode_mode="greedy")
        unbiased = without_bias.rollout(_encode(without_bias, batch), decode_mode="greedy")
    # Identical seeds give both policies the same weights, so any difference is the savings term.
    assert not torch.allclose(biased.log_probability, unbiased.log_probability)


def test_local_prior_influences_decoding() -> None:
    batch = _batch([10], seed=61)
    policy = _policy()
    true_prior = batch.constraint_matrix
    pair_mask = batch.customer_mask[:, :, None] * batch.customer_mask[:, None, :]
    inverted = (1.0 - true_prior) * pair_mask
    with torch.no_grad():
        informed = policy.rollout(
            _encode(policy, batch, m_hat=true_prior), decode_mode="greedy", num_starts=1
        )
        misled = policy.rollout(
            _encode(policy, batch, m_hat=inverted), decode_mode="greedy", num_starts=1
        )
    assert not torch.equal(informed.actions, misled.actions)


def test_local_encoder_requires_a_prior() -> None:
    batch = _batch([5], seed=71)
    policy = _policy()
    with pytest.raises(ValueError, match="requires m_hat"):
        policy.encode(
            batch.coords,
            batch.demands,
            batch.capacity,
            batch.depot_index,
            batch.node_mask,
        )


def test_decoder_local_adjacency_keeps_the_depot_reachable() -> None:
    batch = _batch([6], seed=81)
    adjacency = build_decoder_local_adjacency(
        torch.zeros_like(batch.constraint_matrix),
        batch.customer_node_indices,
        batch.customer_mask,
        batch.depot_index,
        batch.node_mask,
        threshold=0.5,
    )
    depot = int(batch.depot_index[0])
    customers = [int(i) for i in batch.customer_node_indices[0].tolist()]
    assert all(bool(adjacency[0, customer, depot]) for customer in customers)


def test_select_start_nodes_prefers_customers_near_the_depot() -> None:
    batch = _batch([6], seed=91)
    starts = select_start_nodes(batch.coords, batch.depot_index, batch.node_mask, 3)
    assert starts.shape == (1, 3)
    depot = int(batch.depot_index[0])
    assert depot not in starts[0].tolist()

    depot_coords = batch.coords[0, depot]
    distances = [float(torch.linalg.norm(batch.coords[0, i] - depot_coords)) for i in starts[0]]
    assert distances == sorted(distances)


def test_actions_to_routes_splits_on_the_depot() -> None:
    batch = _batch([4], seed=101)
    actions = torch.tensor([[1, 2, 0, 3, 4, 0]])
    routes = actions_to_routes(
        actions,
        batch.depot_index,
        batch.customer_node_indices,
        batch.customer_mask,
    )
    assert routes == [[[0, 1], [2, 3]]]
