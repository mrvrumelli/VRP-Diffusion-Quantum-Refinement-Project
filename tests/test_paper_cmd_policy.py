"""Phase-4 gates for the faithful CMD encoder/decoder policy path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import CVRPBatch, collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.inference.solve_instance import load_policy_checkpoint
from vrp_diffusion_quantum.models.decoder import CVRPPolicy, PolicyEncoding
from vrp_diffusion_quantum.models.gat_encoder import NodeGATEncoder, save_gat_encoder_checkpoint
from vrp_diffusion_quantum.models.paper_cmd_encoder import load_diffusion_gat_checkpoint
from vrp_diffusion_quantum.train.train_policy import (
    build_policy_from_config,
    constraint_matrix_prior,
    evaluate_policy,
    save_policy_checkpoint,
    train_policy,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _example(seed: int, n_customers: int = 5) -> CVRPExample:
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
    demands = np.concatenate([[0.0], rng.integers(1, 4, size=n_customers).astype(float)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=7.0,
        depot_index=0,
        instance_id=f"paper_cmd_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    routes = [[index] for index in range(n_customers)]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=len(routes),
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def _batch() -> CVRPBatch:
    return collate_batch([_example(11), _example(12)])


def _gat_checkpoint(path: Path) -> tuple[Path, NodeGATEncoder]:
    torch.manual_seed(71)
    gat = NodeGATEncoder(hidden_dim=16, num_layers=1, num_heads=4)
    return (
        save_gat_encoder_checkpoint(
            path,
            gat,
            extra={
                "alignment": {
                    "track": "paper_cmd",
                    "contract_version": 1,
                    "paper_id": "arxiv:2603.07568v1",
                    "claim": "unit-test paper reconstruction",
                }
            },
        ),
        gat,
    )


def _model_config(checkpoint: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "paper_cmd",
        "global_gat_checkpoint": str(checkpoint),
        "embedding_dim": 16,
        "global_num_layers": 1,
        "global_num_heads": 4,
        "local_num_layers": 1,
        "local_num_heads": 4,
        "feed_forward_dim": 32,
        "decoder_num_heads": 4,
        "adjacency_mode": "hard",
        "local_threshold": 0.5,
    }
    config.update(overrides)
    return config


def _encode(policy: CVRPPolicy, batch: CVRPBatch, *, with_prior: bool) -> PolicyEncoding:
    return policy.encode(
        batch.coords,
        batch.demands,
        batch.capacity,
        batch.depot_index,
        batch.node_mask,
        m_hat=batch.constraint_matrix if with_prior else None,
        customer_node_indices=batch.customer_node_indices if with_prior else None,
        customer_mask=batch.customer_mask if with_prior else None,
    )


def test_pretrained_global_gat_is_exact_frozen_and_checkpoint_self_contained(
    tmp_path: Path,
) -> None:
    gat_path, pretrained = _gat_checkpoint(tmp_path / "pretrained_gat.pt")
    config = _model_config(gat_path)
    policy = build_policy_from_config(config)

    expected = pretrained.state_dict()
    actual = policy.paper_global_gat.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert all(not parameter.requires_grad for parameter in policy.paper_global_gat.parameters())

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint = save_policy_checkpoint(
        tmp_path / "policy.pt",
        policy=policy,
        optimizer=optimizer,
        epoch=0,
        row={"val_best_cost": 1.0},
        best_metric_name="val_best_cost",
        best_metric_value=1.0,
        extra={
            "model": config,
            "alignment": {
                "track": "paper_cmd",
                "contract_version": 1,
                "paper_id": "arxiv:2603.07568v1",
                "claim": "unit-test paper reconstruction",
            },
        },
    )
    gat_path.unlink()
    loaded, _ = load_policy_checkpoint(checkpoint)

    reloaded = loaded.paper_global_gat.state_dict()
    assert all(torch.equal(expected[key], reloaded[key]) for key in expected)
    assert all(not parameter.requires_grad for parameter in loaded.paper_global_gat.parameters())
    loaded.verify_paper_global_gat_frozen()


def test_frozen_global_gat_receives_no_gradients(tmp_path: Path) -> None:
    gat_path, _ = _gat_checkpoint(tmp_path / "gat.pt")
    policy = build_policy_from_config(_model_config(gat_path))
    batch = _batch()

    policy.train()
    encoding = _encode(policy, batch, with_prior=True)
    rollout = policy.rollout(encoding, decode_mode="sampling", num_starts=3)
    (-rollout.log_probability.mean()).backward()  # type: ignore[no-untyped-call]

    policy.verify_paper_global_gat_frozen()
    assert all(parameter.grad is None for parameter in policy.paper_global_gat.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if not name.startswith("global_encoder.gat.")
    )


def test_paper_policy_rejects_an_ours_robust_global_gat(tmp_path: Path) -> None:
    gat = NodeGATEncoder(hidden_dim=16, num_layers=1, num_heads=4)
    gat_path = save_gat_encoder_checkpoint(
        tmp_path / "robust_gat.pt",
        gat,
        extra={
            "alignment": {
                "track": "ours_robust",
                "contract_version": 1,
                "paper_id": "arxiv:2603.07568v1",
                "claim": "project extension",
            }
        },
    )

    with pytest.raises(ValueError, match="belongs to alignment track 'ours_robust'"):
        build_policy_from_config(_model_config(gat_path))


def test_global_gat_can_be_extracted_from_a_full_diffusion_checkpoint(tmp_path: Path) -> None:
    _, pretrained = _gat_checkpoint(tmp_path / "gat_only.pt")
    denoiser_path = tmp_path / "diffusion.pt"
    torch.save(
        {"model": {f"node_encoder.{key}": value for key, value in pretrained.state_dict().items()}},
        denoiser_path,
    )
    restored = NodeGATEncoder(hidden_dim=16, num_layers=1, num_heads=4)

    load_diffusion_gat_checkpoint(denoiser_path, restored)

    assert all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in pretrained.state_dict().items()
    )


@pytest.mark.paper_smoke
def test_tiny_rl_training_lowers_cost_with_full_feasibility(tmp_path: Path) -> None:
    gat_path, _ = _gat_checkpoint(tmp_path / "gat.pt")
    torch.manual_seed(4)
    policy = build_policy_from_config(_model_config(gat_path))
    assert next(policy.parameters()).device.type == "cpu"
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in policy.paper_global_gat.named_parameters()
    }
    examples = [_example(seed) for seed in range(20, 28)]
    before = evaluate_policy(
        policy,
        examples,
        batch_size=4,
        num_starts=5,
        prior=constraint_matrix_prior,
    )

    history = train_policy(
        policy,
        examples,
        num_epochs=10,
        learning_rate=3e-3,
        batch_size=4,
        num_starts=5,
        prior=constraint_matrix_prior,
        seed=19,
    )

    assert min(float(row["val_best_cost"]) for row in history) < before["best_cost"] - 1e-6
    assert all(float(row["train_feasibility_rate"]) == 1.0 for row in history)
    assert all(float(row["val_feasibility_rate"]) == 1.0 for row in history)
    assert all(
        torch.equal(frozen_before[name], parameter)
        for name, parameter in policy.paper_global_gat.named_parameters()
    )
    policy.verify_paper_global_gat_frozen()


@pytest.mark.parametrize(
    ("overrides", "with_prior"),
    [
        ({"use_local_encoder": False, "use_local_pointer": False}, False),
        ({"use_local_encoder": False, "use_local_pointer": True}, True),
        ({"use_local_encoder": True, "use_local_pointer": False}, True),
    ],
    ids=("no-M", "no-local-encoder", "no-local-pointer"),
)
def test_paper_ablation_modes_decode_feasibly(
    tmp_path: Path,
    overrides: dict[str, object],
    with_prior: bool,
) -> None:
    gat_path, _ = _gat_checkpoint(tmp_path / "gat.pt")
    policy = build_policy_from_config(_model_config(gat_path, **overrides))
    batch = _batch()

    encoding = _encode(policy, batch, with_prior=with_prior)
    rollout = policy.rollout(encoding, decode_mode="greedy", num_starts=2)

    assert torch.isfinite(rollout.cost).all()
    assert rollout.actions.shape[0] == len(batch.metadata) * 2
    if not bool(overrides.get("use_local_pointer", True)):
        assert encoding.local_adjacency is None
