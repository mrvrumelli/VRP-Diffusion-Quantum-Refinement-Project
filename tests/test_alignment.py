"""Tests for explicit paper-baseline versus project-extension contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from vrp_diffusion_quantum.train.train_policy import build_policy_from_config
from vrp_diffusion_quantum.utils.alignment import (
    PAPER_ARXIV_ID,
    resolve_alignment_contract,
    validate_alignment_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "component"),
    [
        ("configs/train/gat_pretrain_paper_cmd.yaml", "gat_pretrain"),
        ("configs/train/diffusion_denoiser_paper_cmd.yaml", "diffusion"),
        ("configs/policy/policy_reinforce_paper_cmd.yaml", "policy"),
    ],
)
def test_paper_cmd_templates_satisfy_declared_contract(relative_path: str, component: str) -> None:
    config = yaml.safe_load((ROOT / relative_path).read_text())

    contract = validate_alignment_config(config, component=component)  # type: ignore[arg-type]

    assert contract.track == "paper_cmd"
    assert contract.paper_id == PAPER_ARXIV_ID
    assert "reconstruction" in contract.claim


def test_paper_cmd_rejects_silent_hyperparameter_drift() -> None:
    path = ROOT / "configs/train/diffusion_denoiser_paper_cmd.yaml"
    config = yaml.safe_load(path.read_text())
    drifted = deepcopy(config)
    drifted["schedule"]["num_timesteps"] = 700
    drifted["model"]["hidden_dim"] = 192

    with pytest.raises(ValueError, match=r"num_timesteps=700.*hidden_dim=192|hidden_dim=192"):
        validate_alignment_config(drifted, component="diffusion")


def test_paper_architecture_requires_explicit_alignment_metadata() -> None:
    config = {"model": {"architecture": "paper_cmd"}}

    with pytest.raises(ValueError, match="requires an explicit alignment block"):
        validate_alignment_config(config, component="policy")


def test_legacy_config_is_loadable_but_cannot_claim_fidelity() -> None:
    contract = resolve_alignment_contract({"model": {}})

    assert contract.track == "legacy_unspecified"
    assert contract.paper_id is None
    assert "no paper-fidelity claim" in contract.claim


def test_ours_robust_policy_template_is_explicit_and_buildable() -> None:
    config = yaml.safe_load((ROOT / "configs/policy/policy_reinforce.yaml").read_text())

    contract = validate_alignment_config(config, component="policy")
    policy = build_policy_from_config(config["model"])

    assert contract.track == "ours_robust"
    assert policy.embedding_dim == 128


def test_paper_policy_cannot_fall_back_to_ours_robust_encoder() -> None:
    config = yaml.safe_load((ROOT / "configs/policy/policy_reinforce_paper_cmd.yaml").read_text())

    with pytest.raises(NotImplementedError, match="paper_cmd policy construction"):
        build_policy_from_config(config["model"])


def test_unknown_alignment_track_and_architecture_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"alignment\.track"):
        resolve_alignment_contract({"alignment": {"track": "paperish", "claim": "invalid"}})
    with pytest.raises(ValueError, match=r"model\.architecture"):
        build_policy_from_config({"architecture": "mystery"})
