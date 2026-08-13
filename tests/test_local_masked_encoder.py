"""Tests for M_hat-guided local attention over full CVRP node batches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models import LocalMaskedEncoder as LocalMaskedEncoderFromPackage
from vrp_diffusion_quantum.models.global_encoder import GlobalEncoder
from vrp_diffusion_quantum.models.local_masked_encoder import (
    LocalMaskedEncoder,
    build_local_attention_prior,
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
    torch.manual_seed(20)
    global_embeddings = torch.randn(2, 4, 16)
    global_embeddings[1, 3] = 0.0
    m_hat = torch.tensor(
        [
            [[0.0, 0.9, 0.1], [0.7, 0.0, 0.6], [0.3, 0.8, 0.0]],
            [[0.0, 0.75, 0.0], [0.65, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    customer_node_indices = torch.tensor([[1, 2, 3], [0, 2, -1]])
    customer_mask = torch.tensor([[True, True, True], [True, True, False]])
    depot_index = torch.tensor([0, 1])
    node_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    return (
        global_embeddings,
        m_hat,
        customer_node_indices,
        customer_mask,
        depot_index,
        node_mask,
    )


def test_build_local_attention_prior_maps_customer_indices() -> None:
    _, m_hat, customer_indices, customer_mask, depot_index, node_mask = _mixed_size_inputs()
    prior = build_local_attention_prior(
        m_hat,
        customer_indices,
        customer_mask,
        depot_index,
        node_mask,
    )

    assert prior.weights.shape == (2, 4, 4)
    assert torch.allclose(prior.weights[0, 1, 2], torch.tensor(0.8))
    assert torch.allclose(prior.weights[0, 2, 1], torch.tensor(0.8))
    assert torch.allclose(prior.weights[1, 0, 2], torch.tensor(0.7))
    assert torch.equal(torch.diagonal(prior.weights[0]), torch.ones(4))
    assert torch.equal(torch.diagonal(prior.weights[1]), torch.tensor([1.0, 1.0, 1.0, 0.0]))

    # Customer queries can read the depot; the depot cannot aggregate customer states.
    assert prior.weights[0, 1, 0] == 1
    assert prior.weights[0, 0, 1] == 0
    assert not prior.allowed_pairs[1, 3].any()
    assert not prior.allowed_pairs[1, :, 3].any()


def test_local_masked_encoder_shapes_padding_and_pooling() -> None:
    encoder = LocalMaskedEncoder(
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
    )
    output = encoder(*_mixed_size_inputs())

    assert output.node_embeddings.shape == (2, 4, 16)
    assert output.graph_embedding.shape == (2, 16)
    assert torch.isfinite(output.node_embeddings).all()
    assert torch.equal(output.node_embeddings[1, 3], torch.zeros(16))
    assert torch.allclose(output.graph_embedding[0], output.node_embeddings[0].mean(dim=0))
    assert torch.allclose(output.graph_embedding[1], output.node_embeddings[1, :3].mean(dim=0))


def test_hard_mask_prevents_cross_group_information_leakage() -> None:
    torch.manual_seed(21)
    encoder = LocalMaskedEncoder(
        embedding_dim=8,
        num_layers=3,
        num_heads=2,
        feed_forward_dim=16,
        adjacency_mode="hard",
        hard_threshold=0.5,
    )
    encoder.eval()
    embeddings = torch.randn(1, 5, 8)
    m_hat = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    customer_indices = torch.tensor([[1, 2, 3, 4]])
    customer_mask = torch.ones((1, 4), dtype=torch.bool)
    depot_index = torch.tensor([0])
    node_mask = torch.ones((1, 5), dtype=torch.bool)

    original = encoder(embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask)
    changed_embeddings = embeddings.clone()
    changed_embeddings[0, 4] += 1_000.0
    changed = encoder(
        changed_embeddings,
        m_hat,
        customer_indices,
        customer_mask,
        depot_index,
        node_mask,
    )

    assert torch.allclose(original.node_embeddings[:, :3], changed.node_embeddings[:, :3])


def test_soft_prior_backpropagates_to_m_hat() -> None:
    torch.manual_seed(22)
    encoder = LocalMaskedEncoder(
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feed_forward_dim=16,
        adjacency_mode="soft",
    )
    embeddings = torch.randn(1, 4, 8)
    m_hat = torch.tensor(
        [[[0.0, 0.8, 0.2], [0.8, 0.0, 0.6], [0.2, 0.6, 0.0]]],
        requires_grad=True,
    )
    customer_indices = torch.tensor([[1, 2, 3]])
    customer_mask = torch.ones((1, 3), dtype=torch.bool)
    depot_index = torch.tensor([0])
    node_mask = torch.ones((1, 4), dtype=torch.bool)

    output = encoder(embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask)
    coefficients = torch.arange(1, 9, dtype=output.node_embeddings.dtype)
    loss = torch.dot(output.node_embeddings[0, 1], coefficients)
    loss.backward()

    assert m_hat.grad is not None
    assert torch.isfinite(m_hat.grad).all()
    assert m_hat.grad.abs().sum() > 0


def test_local_encoder_rejects_customer_mapping_that_contains_depot() -> None:
    inputs = list(_mixed_size_inputs())
    inputs[2] = inputs[2].clone()
    inputs[2][0, 0] = 0
    encoder = LocalMaskedEncoder(embedding_dim=16, num_layers=1, num_heads=4)

    try:
        encoder(*inputs)
    except ValueError as error:
        assert "depot" in str(error)
    else:
        raise AssertionError("expected a customer mapping containing the depot to be rejected")


# --- Additional coverage -----------------------------------------------------------------


def test_local_masked_encoder_is_reexported_from_models_package() -> None:
    assert LocalMaskedEncoderFromPackage is LocalMaskedEncoder


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
        ({"adjacency_mode": "loose"}, "adjacency_mode"),
        ({"hard_threshold": 0.0}, "hard_threshold"),
        ({"hard_threshold": 1.1}, "hard_threshold"),
        ({"soft_bias_strength": -1.0}, "soft_bias_strength"),
        ({"minimum_soft_weight": 0.0}, "minimum_soft_weight"),
        ({"minimum_soft_weight": 1.1}, "minimum_soft_weight"),
    ],
)
def test_local_masked_encoder_rejects_bad_constructor_args(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        LocalMaskedEncoder(**kwargs)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m, ci, cm, dep, nm: (m[:, :2], ci, cm, dep, nm), "m_hat"),
        (lambda m, ci, cm, dep, nm: (m, ci[:, :2], cm, dep, nm), "customer_node_indices"),
        (lambda m, ci, cm, dep, nm: (m, ci, cm[:1], dep, nm), "customer_mask"),
        (lambda m, ci, cm, dep, nm: (m, ci, cm, dep[:1], nm), "depot_index"),
        (lambda m, ci, cm, dep, nm: (m, ci, cm, dep, nm[None]), "node_mask"),
        (lambda m, ci, cm, dep, nm: (m.to(torch.int64), ci, cm, dep, nm), "floating-point"),
        (
            lambda m, ci, cm, dep, nm: (m.index_fill(1, torch.tensor([0]), 1.5), ci, cm, dep, nm),
            "probabilities",
        ),
        (
            lambda m, ci, cm, dep, nm: (
                m.index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor([float("nan")]),
                ),
                ci,
                cm,
                dep,
                nm,
            ),
            "finite",
        ),
        (lambda m, ci, cm, dep, nm: (m, ci.to(torch.float32), cm, dep, nm), "integer dtype"),
        (lambda m, ci, cm, dep, nm: (m, ci, cm, dep.to(torch.float32), nm), "integer dtype"),
        (lambda m, ci, cm, dep, nm: (m, ci, cm, dep + 100, nm), r"\[0, 4\)"),
        (
            lambda m, ci, cm, dep, nm: (
                m,
                ci,
                cm,
                dep,
                nm.clone().index_put_((torch.tensor([0]), dep[:1]), torch.tensor([False])),
            ),
            "depot must be marked",
        ),
        (
            lambda m, ci, cm, dep, nm: (
                m,
                ci.clone().index_fill_(1, torch.tensor([0]), 99),
                cm,
                dep,
                nm,
            ),
            r"customer_node_indices values must be in \[0, 4\)",
        ),
        (
            lambda m, ci, cm, dep, nm: (
                m,
                ci.clone().index_put_((torch.tensor([1]), torch.tensor([2])), torch.tensor([5])),
                cm,
                dep,
                nm,
            ),
            "must be -1",
        ),
        (
            lambda m, ci, cm, dep, nm: (
                m,
                ci.clone().index_put_((torch.tensor([0]), torch.tensor([1])), ci[0, 0:1]),
                cm,
                dep,
                nm,
            ),
            "unique",
        ),
    ],
)
def test_build_local_attention_prior_rejects_malformed_inputs(
    mutate: Callable[..., tuple[torch.Tensor, ...]], match: str
) -> None:
    _, m_hat, customer_indices, customer_mask, depot_index, node_mask = _mixed_size_inputs()
    mutated = mutate(m_hat, customer_indices, customer_mask, depot_index, node_mask)

    with pytest.raises(ValueError, match=match):
        build_local_attention_prior(*mutated)


def test_local_masked_encoder_rejects_customer_index_that_maps_onto_depot() -> None:
    inputs = list(_mixed_size_inputs())
    inputs[2] = inputs[2].clone()
    inputs[2][0, 0] = 0  # customer slot 0 now points at node 0, which is batch 0's depot.
    encoder = LocalMaskedEncoder(embedding_dim=16, num_layers=1, num_heads=4)

    with pytest.raises(ValueError, match="customer_node_indices must not contain the depot"):
        encoder(*inputs)


def test_local_masked_encoder_rejects_node_mask_inconsistent_with_customer_mapping() -> None:
    inputs = list(_mixed_size_inputs())
    inputs[5] = inputs[5].clone()
    inputs[5][0, 2] = False  # node 2 is a real mapped customer but marked padding here.
    encoder = LocalMaskedEncoder(embedding_dim=16, num_layers=1, num_heads=4)

    with pytest.raises(ValueError, match="exactly one depot plus all mapped customers"):
        encoder(*inputs)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda g, m, ci, cm, dep, nm: (g[..., 0], m, ci, cm, dep, nm), "global_node_embeddings"),
        (lambda g, m, ci, cm, dep, nm: (g[:, :, :4], m, ci, cm, dep, nm), "embedding_dim"),
        (lambda g, m, ci, cm, dep, nm: (g, m, ci, cm, dep, nm[:, :1]), "node_mask"),
        (
            lambda g, m, ci, cm, dep, nm: (
                g.index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor([float("nan")]),
                ),
                m,
                ci,
                cm,
                dep,
                nm,
            ),
            "finite",
        ),
    ],
)
def test_local_masked_encoder_forward_rejects_malformed_inputs(
    mutate: Callable[..., tuple[torch.Tensor, ...]], match: str
) -> None:
    global_embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask = (
        _mixed_size_inputs()
    )
    mutated = mutate(
        global_embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask
    )

    encoder = LocalMaskedEncoder(embedding_dim=16, num_layers=1, num_heads=4)
    with pytest.raises(ValueError, match=match):
        encoder(*mutated)


def test_local_masked_encoder_handles_depot_only_graph() -> None:
    """A CVRP instance with zero customers (depot node only) must not crash."""
    example = _real_example(0, seed=1)
    batch = collate_batch([example])
    global_embeddings = torch.randn(1, 1, 8)

    torch.manual_seed(3)
    encoder = LocalMaskedEncoder(embedding_dim=8, num_layers=2, num_heads=2, feed_forward_dim=16)
    m_hat = batch.constraint_matrix
    output = encoder(
        global_embeddings,
        m_hat,
        batch.customer_node_indices,
        batch.customer_mask,
        batch.depot_index,
        batch.node_mask,
    )

    assert output.node_embeddings.shape == (1, 1, 8)
    assert torch.allclose(output.graph_embedding[0], output.node_embeddings[0, 0])


def test_local_masked_encoder_integrates_with_global_encoder_and_real_batch() -> None:
    """Chain GlobalEncoder -> LocalMaskedEncoder over a real, mixed-size `CVRPBatch`."""
    small = _real_example(5, seed=1)
    large = _real_example(9, seed=2)
    batch = collate_batch([small, large])

    torch.manual_seed(5)
    global_encoder = GlobalEncoder(embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32)
    local_encoder = LocalMaskedEncoder(
        embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32
    )

    global_output = global_encoder(
        batch.coords, batch.demands, batch.capacity, batch.depot_index, batch.node_mask
    )
    # batch.constraint_matrix is the ground-truth route-membership matrix; it is a valid
    # (binary) stand-in for a predicted m_hat since both live in [0, 1].
    output = local_encoder(
        global_output.node_embeddings,
        batch.constraint_matrix,
        batch.customer_node_indices,
        batch.customer_mask,
        batch.depot_index,
        batch.node_mask,
    )

    assert output.node_embeddings.shape == (2, 10, 16)
    assert torch.isfinite(output.node_embeddings).all()
    assert torch.equal(output.node_embeddings[0, 6:], torch.zeros(4, 16))
    assert not torch.equal(output.node_embeddings[1, 6:], torch.zeros(4, 16))


def test_local_masked_encoder_output_independent_of_other_batch_rows() -> None:
    """A real instance's local embedding must not change with what else shares its batch."""
    small = _real_example(4, seed=7)
    large = _real_example(11, seed=8)

    torch.manual_seed(21)
    encoder = LocalMaskedEncoder(embedding_dim=16, num_layers=2, num_heads=4, feed_forward_dim=32)
    encoder.eval()

    alone_batch = collate_batch([small])
    joint_batch = collate_batch([small, large])
    n_real_nodes = small.instance.n_customers + 1

    alone_embeddings = torch.randn(1, n_real_nodes, 16)
    joint_embeddings = torch.zeros(2, joint_batch.coords.shape[1], 16)
    joint_embeddings[0, :n_real_nodes] = alone_embeddings[0]
    joint_embeddings[1, : large.instance.n_customers + 1] = torch.randn(
        large.instance.n_customers + 1, 16
    )

    alone = encoder(
        alone_embeddings,
        alone_batch.constraint_matrix,
        alone_batch.customer_node_indices,
        alone_batch.customer_mask,
        alone_batch.depot_index,
        alone_batch.node_mask,
    )
    joint = encoder(
        joint_embeddings,
        joint_batch.constraint_matrix,
        joint_batch.customer_node_indices,
        joint_batch.customer_mask,
        joint_batch.depot_index,
        joint_batch.node_mask,
    )

    assert torch.allclose(
        alone.node_embeddings[0, :n_real_nodes],
        joint.node_embeddings[0, :n_real_nodes],
        atol=1e-6,
    )
    assert torch.allclose(alone.graph_embedding[0], joint.graph_embedding[0], atol=1e-6)


def test_local_masked_encoder_soft_mode_depot_ignores_customer_perturbation() -> None:
    """Even in the default soft mode, the depot query only ever attends to itself."""
    torch.manual_seed(30)
    encoder = LocalMaskedEncoder(
        embedding_dim=8, num_layers=2, num_heads=2, feed_forward_dim=16, adjacency_mode="soft"
    )
    encoder.eval()
    embeddings = torch.randn(1, 4, 8)
    m_hat = torch.tensor(
        [[[0.0, 0.9, 0.2], [0.9, 0.0, 0.1], [0.2, 0.1, 0.0]]],
    )
    customer_indices = torch.tensor([[1, 2, 3]])
    customer_mask = torch.ones((1, 3), dtype=torch.bool)
    depot_index = torch.tensor([0])
    node_mask = torch.ones((1, 4), dtype=torch.bool)

    original = encoder(embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask)
    perturbed_embeddings = embeddings.clone()
    perturbed_embeddings[0, 2] += 1_000.0
    perturbed = encoder(
        perturbed_embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask
    )

    assert torch.allclose(original.node_embeddings[0, 0], perturbed.node_embeddings[0, 0])


def test_local_masked_encoder_hard_mode_cuts_gradient_to_m_hat() -> None:
    """Hard thresholding is a non-differentiable step, so m_hat must receive no gradient."""
    torch.manual_seed(31)
    encoder = LocalMaskedEncoder(
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feed_forward_dim=16,
        adjacency_mode="hard",
        hard_threshold=0.5,
    )
    embeddings = torch.randn(1, 4, 8)
    m_hat = torch.tensor(
        [[[0.0, 0.8, 0.2], [0.8, 0.0, 0.6], [0.2, 0.6, 0.0]]],
        requires_grad=True,
    )
    customer_indices = torch.tensor([[1, 2, 3]])
    customer_mask = torch.ones((1, 3), dtype=torch.bool)
    depot_index = torch.tensor([0])
    node_mask = torch.ones((1, 4), dtype=torch.bool)

    output = encoder(embeddings, m_hat, customer_indices, customer_mask, depot_index, node_mask)
    output.node_embeddings.square().sum().backward()

    assert m_hat.grad is None


def test_local_masked_encoder_dropout_is_stochastic_in_train_and_stable_in_eval() -> None:
    torch.manual_seed(42)
    encoder = LocalMaskedEncoder(
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.5,
    )
    inputs = _mixed_size_inputs()

    encoder.train()
    torch.manual_seed(100)
    train_out_1 = encoder(*inputs).node_embeddings
    torch.manual_seed(200)
    train_out_2 = encoder(*inputs).node_embeddings
    assert not torch.allclose(train_out_1, train_out_2)

    encoder.eval()
    eval_out_1 = encoder(*inputs).node_embeddings
    eval_out_2 = encoder(*inputs).node_embeddings
    assert torch.allclose(eval_out_1, eval_out_2)


def test_build_local_attention_prior_dtype_override() -> None:
    _, m_hat, customer_indices, customer_mask, depot_index, node_mask = _mixed_size_inputs()
    prior = build_local_attention_prior(
        m_hat, customer_indices, customer_mask, depot_index, node_mask, dtype=torch.float64
    )
    assert prior.weights.dtype == torch.float64
