"""Tests for learned gated fusion of global and local CVRP embeddings."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models import FusionEncoder as FusionEncoderFromPackage
from vrp_diffusion_quantum.models.fusion_encoder import FusionEncoder
from vrp_diffusion_quantum.models.global_encoder import GlobalEncoder
from vrp_diffusion_quantum.models.local_masked_encoder import LocalMaskedEncoder


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


def _mixed_size_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(30)
    global_embeddings = torch.randn(2, 4, 16)
    local_embeddings = torch.randn(2, 4, 16)
    node_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    global_embeddings[1, 3] = 500.0
    local_embeddings[1, 3] = -500.0
    return global_embeddings, local_embeddings, node_mask


def test_fusion_shapes_padding_pooling_and_initial_gate() -> None:
    encoder = FusionEncoder(
        embedding_dim=16,
        gate_hidden_dim=8,
        feed_forward_dim=32,
        initial_local_weight=0.25,
    )
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()
    output = encoder(global_embeddings, local_embeddings, node_mask)

    assert output.node_embeddings.shape == (2, 4, 16)
    assert output.graph_embedding.shape == (2, 16)
    assert output.fusion_gate.shape == (2, 4, 16)
    assert torch.isfinite(output.node_embeddings).all()
    assert torch.equal(output.node_embeddings[1, 3], torch.zeros(16))
    assert torch.equal(output.fusion_gate[1, 3], torch.zeros(16))
    assert torch.allclose(output.fusion_gate[0], torch.full((4, 16), 0.25))
    assert torch.allclose(output.graph_embedding[0], output.node_embeddings[0].mean(dim=0))
    assert torch.allclose(output.graph_embedding[1], output.node_embeddings[1, :3].mean(dim=0))


def test_padded_values_cannot_change_fused_real_nodes() -> None:
    torch.manual_seed(31)
    encoder = FusionEncoder(embedding_dim=16, gate_hidden_dim=8, feed_forward_dim=32)
    encoder.eval()
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()
    changed_global = global_embeddings.clone()
    changed_local = local_embeddings.clone()
    changed_global[1, 3] = -10_000.0
    changed_local[1, 3] = 20_000.0

    original = encoder(global_embeddings, local_embeddings, node_mask)
    changed = encoder(changed_global, changed_local, node_mask)

    assert torch.allclose(original.node_embeddings[1, :3], changed.node_embeddings[1, :3])
    assert torch.allclose(original.graph_embedding[1], changed.graph_embedding[1])


def test_fusion_backpropagates_to_both_encoder_paths_and_gate() -> None:
    torch.manual_seed(32)
    encoder = FusionEncoder(embedding_dim=8, gate_hidden_dim=6, feed_forward_dim=16)
    global_embeddings = torch.randn(1, 4, 8, requires_grad=True)
    local_embeddings = torch.randn(1, 4, 8, requires_grad=True)
    node_mask = torch.tensor([[True, True, True, False]])

    output = encoder(global_embeddings, local_embeddings, node_mask)
    coefficients = torch.arange(1, 9, dtype=output.node_embeddings.dtype)
    loss = torch.dot(output.node_embeddings[0, 1], coefficients)
    loss.backward()

    assert global_embeddings.grad is not None
    assert local_embeddings.grad is not None
    assert global_embeddings.grad[0, :3].abs().sum() > 0
    assert local_embeddings.grad[0, :3].abs().sum() > 0
    assert torch.equal(global_embeddings.grad[0, 3], torch.zeros(8))
    assert torch.equal(local_embeddings.grad[0, 3], torch.zeros(8))
    gate_output = encoder.gate_network[-1]
    assert isinstance(gate_output, torch.nn.Linear)
    assert gate_output.bias.grad is not None
    assert gate_output.bias.grad.abs().sum() > 0


def test_fusion_is_equivariant_to_node_order() -> None:
    torch.manual_seed(33)
    encoder = FusionEncoder(embedding_dim=16, gate_hidden_dim=8, feed_forward_dim=32)
    encoder.eval()
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse_permutation = torch.argsort(permutation)

    original = encoder(global_embeddings[:1], local_embeddings[:1], node_mask[:1])
    permuted = encoder(
        global_embeddings[:1, permutation],
        local_embeddings[:1, permutation],
        node_mask[:1, permutation],
    )

    assert torch.allclose(
        original.node_embeddings,
        permuted.node_embeddings[:, inverse_permutation],
        atol=1e-6,
    )
    assert torch.allclose(original.graph_embedding, permuted.graph_embedding, atol=1e-6)
    assert torch.allclose(
        original.fusion_gate,
        permuted.fusion_gate[:, inverse_permutation],
        atol=1e-6,
    )


def test_fusion_rejects_misaligned_local_embeddings() -> None:
    encoder = FusionEncoder(embedding_dim=16)
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()

    try:
        encoder(global_embeddings, local_embeddings[:, :3], node_mask)
    except ValueError as error:
        assert "match" in str(error)
    else:
        raise AssertionError("expected misaligned local embeddings to be rejected")


# --- Additional coverage -----------------------------------------------------------------


def test_fusion_encoder_is_reexported_from_models_package() -> None:
    assert FusionEncoderFromPackage is FusionEncoder


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"gate_hidden_dim": 0}, "gate_hidden_dim"),
        ({"feed_forward_dim": 0}, "feed_forward_dim"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"initial_local_weight": 0.0}, "initial_local_weight"),
        ({"initial_local_weight": 1.0}, "initial_local_weight"),
    ],
)
def test_fusion_encoder_rejects_bad_constructor_args(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FusionEncoder(**kwargs)


def test_fusion_encoder_default_gate_hidden_dim_matches_embedding_dim() -> None:
    """`gate_hidden_dim=None` (the default) must not crash and should shape-check correctly."""
    encoder = FusionEncoder(embedding_dim=12, feed_forward_dim=24)
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()
    global_embeddings = torch.randn(2, 4, 12)
    local_embeddings = torch.randn(2, 4, 12)

    output = encoder(global_embeddings, local_embeddings, node_mask)

    assert output.node_embeddings.shape == (2, 4, 12)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda g, loc, m: (g[..., 0], loc, m), "global_node_embeddings"),
        (lambda g, loc, m: (g, loc[:, :3], m), "match"),
        (lambda g, loc, m: (g, loc, m[:, :1]), "node_mask"),
        (lambda g, loc, m: (g, loc.to(torch.float64), m), "same dtype"),
        (
            lambda g, loc, m: (
                g.index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor([float("nan")]),
                ),
                loc,
                m,
            ),
            "global_node_embeddings must contain only finite",
        ),
        (
            lambda g, loc, m: (
                g,
                loc.index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor([float("nan")]),
                ),
                m,
            ),
            "local_node_embeddings must contain only finite",
        ),
        (
            lambda g, loc, m: (
                g,
                loc,
                m.clone().index_put_((torch.tensor([1]),), torch.zeros_like(m[1])),
            ),
            "at least one real node",
        ),
    ],
)
def test_fusion_encoder_rejects_malformed_inputs(
    mutate: Callable[..., tuple[torch.Tensor, ...]], match: str
) -> None:
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()
    mutated_g, mutated_l, mutated_m = mutate(global_embeddings, local_embeddings, node_mask)

    encoder = FusionEncoder(embedding_dim=16, gate_hidden_dim=8, feed_forward_dim=32)
    with pytest.raises(ValueError, match=match):
        encoder(mutated_g, mutated_l, mutated_m)


def test_fusion_encoder_gate_stays_in_unit_interval_for_real_nodes() -> None:
    torch.manual_seed(40)
    encoder = FusionEncoder(embedding_dim=16, gate_hidden_dim=8, feed_forward_dim=32)
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()

    output = encoder(global_embeddings, local_embeddings, node_mask)
    real = output.fusion_gate[node_mask]

    assert torch.all(real > 0.0)
    assert torch.all(real < 1.0)


def test_fusion_encoder_matches_manual_recomputation_of_documented_formula() -> None:
    """Reproduce `mixed = h_global + gate * (h_local - h_global)` externally using the module's
    own submodules, pinning the untrained (zero-init gate weight) constant-gate behavior."""
    torch.manual_seed(41)
    initial_local_weight = 0.3
    encoder = FusionEncoder(
        embedding_dim=8,
        gate_hidden_dim=6,
        feed_forward_dim=16,
        initial_local_weight=initial_local_weight,
    )
    encoder.eval()
    global_embeddings = torch.randn(1, 3, 8)
    local_embeddings = torch.randn(1, 3, 8)
    node_mask = torch.ones((1, 3), dtype=torch.bool)

    output = encoder(global_embeddings, local_embeddings, node_mask)

    mixed = global_embeddings + initial_local_weight * (local_embeddings - global_embeddings)
    feed_forward_update = encoder.feed_forward(encoder.feed_forward_norm(mixed))
    expected = encoder.output_norm(mixed + feed_forward_update)

    assert torch.allclose(
        output.fusion_gate, torch.full_like(output.fusion_gate, initial_local_weight), atol=1e-6
    )
    assert torch.allclose(output.node_embeddings, expected, atol=1e-5)


def test_fusion_encoder_dropout_is_stochastic_in_train_and_stable_in_eval() -> None:
    torch.manual_seed(42)
    encoder = FusionEncoder(
        embedding_dim=16,
        gate_hidden_dim=8,
        feed_forward_dim=32,
        dropout=0.5,
    )
    global_embeddings, local_embeddings, node_mask = _mixed_size_inputs()

    encoder.train()
    torch.manual_seed(100)
    train_out_1 = encoder(global_embeddings, local_embeddings, node_mask).node_embeddings
    torch.manual_seed(200)
    train_out_2 = encoder(global_embeddings, local_embeddings, node_mask).node_embeddings
    assert not torch.allclose(train_out_1, train_out_2)

    encoder.eval()
    eval_out_1 = encoder(global_embeddings, local_embeddings, node_mask).node_embeddings
    eval_out_2 = encoder(global_embeddings, local_embeddings, node_mask).node_embeddings
    assert torch.allclose(eval_out_1, eval_out_2)


def test_fusion_encoder_handles_depot_only_graph() -> None:
    """A single-real-node graph (e.g. depot only) must not crash."""
    torch.manual_seed(43)
    encoder = FusionEncoder(embedding_dim=8, gate_hidden_dim=6, feed_forward_dim=16)
    global_embeddings = torch.randn(1, 1, 8)
    local_embeddings = torch.randn(1, 1, 8)
    node_mask = torch.tensor([[True]])

    output = encoder(global_embeddings, local_embeddings, node_mask)

    assert output.node_embeddings.shape == (1, 1, 8)
    assert torch.allclose(output.graph_embedding[0], output.node_embeddings[0, 0])


def test_fusion_encoder_integrates_with_global_and_local_encoders_on_real_batch() -> None:
    """Chain GlobalEncoder -> LocalMaskedEncoder -> FusionEncoder over a real, mixed-size batch."""
    small = _real_example(5, seed=1)
    large = _real_example(9, seed=2)
    batch = collate_batch([small, large])

    torch.manual_seed(6)
    global_encoder = GlobalEncoder(embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32)
    local_encoder = LocalMaskedEncoder(
        embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32
    )
    fusion_encoder = FusionEncoder(embedding_dim=16, gate_hidden_dim=8, feed_forward_dim=32)

    global_output = global_encoder(
        batch.coords, batch.demands, batch.capacity, batch.depot_index, batch.node_mask
    )
    local_output = local_encoder(
        global_output.node_embeddings,
        batch.constraint_matrix,
        batch.customer_node_indices,
        batch.customer_mask,
        batch.depot_index,
        batch.node_mask,
    )
    output = fusion_encoder(
        global_output.node_embeddings, local_output.node_embeddings, batch.node_mask
    )

    assert output.node_embeddings.shape == (2, 10, 16)
    assert torch.isfinite(output.node_embeddings).all()
    assert torch.equal(output.node_embeddings[0, 6:], torch.zeros(4, 16))
    assert not torch.equal(output.node_embeddings[1, 6:], torch.zeros(4, 16))
    assert torch.equal(output.fusion_gate[0, 6:], torch.zeros(4, 16))


def test_fusion_encoder_output_independent_of_other_batch_rows() -> None:
    """A real instance's fused embedding must not change with what else shares its batch."""
    small = _real_example(4, seed=7)
    large = _real_example(11, seed=8)

    torch.manual_seed(23)
    encoder = FusionEncoder(embedding_dim=16, gate_hidden_dim=8, feed_forward_dim=32)
    encoder.eval()

    alone_batch = collate_batch([small])
    joint_batch = collate_batch([small, large])
    n_real_nodes = small.instance.n_customers + 1
    max_nodes_joint = joint_batch.coords.shape[1]

    torch.manual_seed(24)
    alone_global = torch.randn(1, n_real_nodes, 16)
    alone_local = torch.randn(1, n_real_nodes, 16)
    joint_global = torch.zeros(2, max_nodes_joint, 16)
    joint_local = torch.zeros(2, max_nodes_joint, 16)
    joint_global[0, :n_real_nodes] = alone_global[0]
    joint_local[0, :n_real_nodes] = alone_local[0]
    joint_global[1, : large.instance.n_customers + 1] = torch.randn(
        large.instance.n_customers + 1, 16
    )
    joint_local[1, : large.instance.n_customers + 1] = torch.randn(
        large.instance.n_customers + 1, 16
    )

    alone = encoder(alone_global, alone_local, alone_batch.node_mask)
    joint = encoder(joint_global, joint_local, joint_batch.node_mask)

    assert torch.allclose(
        alone.node_embeddings[0, :n_real_nodes],
        joint.node_embeddings[0, :n_real_nodes],
        atol=1e-6,
    )
    assert torch.allclose(alone.graph_embedding[0], joint.graph_embedding[0], atol=1e-6)
