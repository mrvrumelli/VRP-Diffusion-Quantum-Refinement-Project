"""Tests for the full depot-and-customer CVRP global encoder."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models import GlobalEncoder as GlobalEncoderFromPackage
from vrp_diffusion_quantum.models.global_encoder import (
    GlobalEncoder,
    build_global_node_features,
)


def _real_example(n_customers: int, *, seed: int = 0) -> CVRPExample:
    """Build a real `CVRPExample` (depot at index 0) via the actual data-layer types."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
    demands = np.concatenate([[0.0], np.ones(n_customers)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=float(max(n_customers, 1)),
        depot_index=0,
        instance_id=f"cvrp{n_customers}_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    mid = n_customers // 2
    routes = [r for r in (list(range(mid)), list(range(mid, n_customers))) if r]
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


def _mixed_size_inputs() -> tuple[torch.Tensor, ...]:
    coords = torch.tensor(
        [
            [[0.5, 0.5], [0.1, 0.2], [0.7, 0.8], [0.3, 0.9]],
            [[0.2, 0.3], [0.9, 0.8], [0.4, 0.1], [99.0, 99.0]],
        ],
        dtype=torch.float32,
    )
    demands = torch.tensor([[0.0, 2.0, 3.0, 1.0], [2.0, 0.0, 4.0, 99.0]])
    capacity = torch.tensor([10.0, 8.0])
    depot_index = torch.tensor([0, 1])
    node_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    return coords, demands, capacity, depot_index, node_mask


def test_global_features_include_depot_and_normalized_demand() -> None:
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()
    features = build_global_node_features(coords, demands, capacity, depot_index, node_mask)

    assert features.shape == (2, 4, 4)
    assert torch.equal(features[:, :, 3], torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]))
    assert torch.allclose(features[0, :, 2], torch.tensor([0.0, 0.2, 0.3, 0.1]))
    assert torch.equal(features[1, 3], torch.zeros(4))


def test_global_encoder_shapes_padding_and_pooling() -> None:
    torch.manual_seed(7)
    encoder = GlobalEncoder(
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
    )
    inputs = _mixed_size_inputs()
    output = encoder(*inputs)

    assert output.node_embeddings.shape == (2, 4, 16)
    assert output.graph_embedding.shape == (2, 16)
    assert torch.isfinite(output.node_embeddings).all()
    assert torch.equal(output.node_embeddings[1, 3], torch.zeros(16))
    assert torch.allclose(output.graph_embedding[0], output.node_embeddings[0].mean(dim=0))
    assert torch.allclose(output.graph_embedding[1], output.node_embeddings[1, :3].mean(dim=0))


def test_padded_values_cannot_change_real_node_embeddings() -> None:
    torch.manual_seed(11)
    encoder = GlobalEncoder(
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
    )
    encoder.eval()
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()
    changed_coords = coords.clone()
    changed_demands = demands.clone()
    changed_coords[1, 3] = torch.tensor([-1_000.0, 2_000.0])
    changed_demands[1, 3] = -5_000.0

    original = encoder(coords, demands, capacity, depot_index, node_mask)
    changed = encoder(changed_coords, changed_demands, capacity, depot_index, node_mask)

    assert torch.allclose(original.node_embeddings[1, :3], changed.node_embeddings[1, :3])
    assert torch.allclose(original.graph_embedding[1], changed.graph_embedding[1])


def test_global_encoder_is_equivariant_to_node_order() -> None:
    torch.manual_seed(12)
    encoder = GlobalEncoder(
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
    )
    encoder.eval()
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()
    coords = coords[:1]
    demands = demands[:1]
    capacity = capacity[:1]
    depot_index = depot_index[:1]
    node_mask = node_mask[:1]
    permutation = torch.tensor([2, 0, 3, 1])
    inverse_permutation = torch.argsort(permutation)

    original = encoder(coords, demands, capacity, depot_index, node_mask)
    permuted = encoder(
        coords[:, permutation],
        demands[:, permutation],
        capacity,
        torch.tensor([int(inverse_permutation[depot_index[0]])]),
        node_mask[:, permutation],
    )

    assert torch.allclose(
        original.node_embeddings,
        permuted.node_embeddings[:, inverse_permutation],
        atol=1e-6,
    )
    assert torch.allclose(original.graph_embedding, permuted.graph_embedding, atol=1e-6)


def test_global_encoder_backpropagates_through_real_nodes() -> None:
    torch.manual_seed(13)
    encoder = GlobalEncoder(
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feed_forward_dim=16,
    )
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()
    coords.requires_grad_(True)
    output = encoder(coords, demands, capacity, depot_index, node_mask)
    output.graph_embedding.square().sum().backward()

    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert torch.equal(coords.grad[1, 3], torch.zeros(2))


def test_global_encoder_rejects_masked_depot() -> None:
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()
    node_mask[0, 0] = False

    encoder = GlobalEncoder(embedding_dim=8, num_layers=1, num_heads=2)
    try:
        encoder(coords, demands, capacity, depot_index, node_mask)
    except ValueError as error:
        assert "depot" in str(error)
    else:
        raise AssertionError("expected a masked depot to be rejected")


# --- Additional coverage -----------------------------------------------------------------


def test_global_encoder_is_reexported_from_models_package() -> None:
    assert GlobalEncoderFromPackage is GlobalEncoder


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"num_layers": 0}, "num_layers"),
        ({"num_heads": 0}, "num_heads"),
        ({"embedding_dim": 10, "num_heads": 3}, "divisible"),
        ({"feed_forward_dim": 0}, "feed_forward_dim"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_global_encoder_rejects_bad_constructor_args(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GlobalEncoder(**kwargs)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c, d, cap, dep, m: (c[:, :, :1], d, cap, dep, m), "coords"),
        (lambda c, d, cap, dep, m: (c, d[:, :1], cap, dep, m), "demands"),
        (lambda c, d, cap, dep, m: (c, d, cap[:1], dep, m), "capacity"),
        (lambda c, d, cap, dep, m: (c, d, cap, dep[:1], m), "depot_index"),
        (lambda c, d, cap, dep, m: (c, d, cap, dep, m[:, :1]), "node_mask"),
        (lambda c, d, cap, dep, m: (c, d, cap, dep.to(torch.float32), m), "integer dtype"),
        (lambda c, d, cap, dep, m: (c, d, cap, dep + 100, m), r"\[0, 4\)"),
        (lambda c, d, cap, dep, m: (c, d, cap * 0, dep, m), "positive"),
        (
            lambda c, d, cap, dep, m: (
                c.index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor([float("nan")])),
                d,
                cap,
                dep,
                m,
            ),
            "finite",
        ),
    ],
)
def test_global_encoder_rejects_malformed_forward_inputs(
    mutate: Callable[..., tuple[torch.Tensor, ...]], match: str
) -> None:
    inputs = _mixed_size_inputs()
    mutated = mutate(*inputs)

    encoder = GlobalEncoder(embedding_dim=8, num_layers=1, num_heads=2)
    with pytest.raises(ValueError, match=match):
        encoder(*mutated)


def test_global_encoder_handles_depot_only_graph() -> None:
    """A CVRP instance with zero customers (depot node only) must not crash."""
    example = _real_example(0, seed=1)
    batch = collate_batch([example])

    torch.manual_seed(3)
    encoder = GlobalEncoder(embedding_dim=8, num_layers=2, num_heads=2, feed_forward_dim=16)
    output = encoder(
        batch.coords, batch.demands, batch.capacity, batch.depot_index, batch.node_mask
    )

    assert output.node_embeddings.shape == (1, 1, 8)
    assert torch.allclose(output.graph_embedding[0], output.node_embeddings[0, 0])


def test_global_encoder_integrates_with_real_mixed_size_batch() -> None:
    """Feed the encoder a real `CVRPBatch` (via collate_batch) mixing two instance sizes."""
    small = _real_example(5, seed=1)
    large = _real_example(9, seed=2)
    batch = collate_batch([small, large])

    torch.manual_seed(5)
    encoder = GlobalEncoder(embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32)
    output = encoder(
        batch.coords, batch.demands, batch.capacity, batch.depot_index, batch.node_mask
    )

    assert output.node_embeddings.shape == (2, 10, 16)
    assert torch.isfinite(output.node_embeddings).all()
    # small's batch row is padded past its 6 real nodes (depot + 5 customers).
    assert torch.equal(output.node_embeddings[0, 6:], torch.zeros(4, 16))
    assert not torch.equal(output.node_embeddings[1, 6:], torch.zeros(4, 16))


def test_global_encoder_output_independent_of_other_batch_rows() -> None:
    """A real instance's embedding must not change depending on what else shares its batch."""
    small = _real_example(4, seed=7)
    large = _real_example(11, seed=8)

    torch.manual_seed(21)
    encoder = GlobalEncoder(embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32)
    encoder.eval()

    alone_batch = collate_batch([small])
    joint_batch = collate_batch([small, large])

    alone = encoder(
        alone_batch.coords,
        alone_batch.demands,
        alone_batch.capacity,
        alone_batch.depot_index,
        alone_batch.node_mask,
    )
    joint = encoder(
        joint_batch.coords,
        joint_batch.demands,
        joint_batch.capacity,
        joint_batch.depot_index,
        joint_batch.node_mask,
    )

    n_real_nodes = small.instance.n_customers + 1
    assert torch.allclose(
        alone.node_embeddings[0, :n_real_nodes],
        joint.node_embeddings[0, :n_real_nodes],
        atol=1e-6,
    )
    assert torch.allclose(alone.graph_embedding[0], joint.graph_embedding[0], atol=1e-6)


def test_global_encoder_dropout_is_stochastic_in_train_and_stable_in_eval() -> None:
    torch.manual_seed(42)
    encoder = GlobalEncoder(
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.5,
    )
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()

    encoder.train()
    torch.manual_seed(100)
    train_out_1 = encoder(coords, demands, capacity, depot_index, node_mask).node_embeddings
    torch.manual_seed(200)
    train_out_2 = encoder(coords, demands, capacity, depot_index, node_mask).node_embeddings
    assert not torch.allclose(train_out_1, train_out_2)

    encoder.eval()
    eval_out_1 = encoder(coords, demands, capacity, depot_index, node_mask).node_embeddings
    eval_out_2 = encoder(coords, demands, capacity, depot_index, node_mask).node_embeddings
    assert torch.allclose(eval_out_1, eval_out_2)


def test_global_encoder_gradients_flow_through_demands() -> None:
    torch.manual_seed(9)
    encoder = GlobalEncoder(embedding_dim=8, num_layers=1, num_heads=2, feed_forward_dim=16)
    coords, demands, capacity, depot_index, node_mask = _mixed_size_inputs()
    demands.requires_grad_(True)

    output = encoder(coords, demands, capacity, depot_index, node_mask)
    output.graph_embedding.square().sum().backward()

    assert demands.grad is not None
    assert torch.isfinite(demands.grad).all()
    assert torch.equal(demands.grad[1, 3], torch.zeros(()))
