"""Tests for explicit paper-baseline versus project-extension contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from vrp_diffusion_quantum.train.train_policy import build_policy_from_config
from vrp_diffusion_quantum.utils.alignment import (
    PAPER_ARXIV_ID,
    PAPER_CONTRACT_VERSION,
    require_artifact_alignment,
    resolve_alignment_contract,
    validate_alignment_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile_name", ["paper_cmd", "ours_robust"])
def test_named_alignment_profiles_reference_matching_component_configs(
    profile_name: str,
) -> None:
    profile = yaml.safe_load(
        (ROOT / "configs" / "alignment" / f"{profile_name}.yaml").read_text()
    )

    assert profile["profile_name"] == profile_name
    assert profile["contract_version"] == PAPER_CONTRACT_VERSION
    for component, relative_path in profile["component_configs"].items():
        config = yaml.safe_load((ROOT / relative_path).read_text())
        contract = validate_alignment_config(config, component=component)
        assert contract.track == profile_name


def test_evidence_ledger_classifies_every_detail_and_tracks_required_questions() -> None:
    ledger = yaml.safe_load(
        (ROOT / "configs" / "alignment" / "cmd_paper_evidence.yaml").read_text()
    )
    allowed = set(ledger["classification_values"])
    details = ledger["details"]

    assert details
    assert all(detail["classification"] in allowed for detail in details)
    assert all(detail.get("source") and detail.get("value") for detail in details)
    assert {detail["classification"] for detail in details} == allowed
    ambiguous = [
        detail
        for detail in details
        if detail["classification"] == "ambiguous_awaiting_author_clarification"
    ]
    assert all(detail.get("reconstruction_choice") for detail in ambiguous)

    required_questions = {
        "exact_seeds",
        "hgs_budget",
        "gat_pretraining_objective",
        "augmentation_enumeration",
        "skipped_transition_formula",
        "test_set_generation_seeds",
        "per_size_training",
    }
    questions = {question["id"]: question for question in ledger["author_questions"]}
    assert required_questions <= questions.keys()
    assert all(questions[question_id]["status"] == "open" for question_id in required_questions)


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
    assert contract.contract_version == PAPER_CONTRACT_VERSION
    assert "reconstruction" in contract.claim


def test_paper_cmd_rejects_silent_hyperparameter_drift() -> None:
    path = ROOT / "configs/train/diffusion_denoiser_paper_cmd.yaml"
    config = yaml.safe_load(path.read_text())
    drifted = deepcopy(config)
    drifted["schedule"]["num_timesteps"] = 700
    drifted["model"]["hidden_dim"] = 192

    with pytest.raises(ValueError, match=r"num_timesteps=700.*hidden_dim=192|hidden_dim=192"):
        validate_alignment_config(drifted, component="diffusion")


def test_paper_cmd_rejects_robust_dataset_policy() -> None:
    path = ROOT / "configs/train/diffusion_denoiser_paper_cmd.yaml"
    config = yaml.safe_load(path.read_text())
    config["dataset"]["label_policy"] = "audited_consensus"

    with pytest.raises(ValueError, match=r"dataset\.label_policy='audited_consensus'"):
        validate_alignment_config(config, component="diffusion")


def test_paper_artifact_gate_accepts_only_matching_versioned_provenance() -> None:
    paper_alignment = {
        "track": "paper_cmd",
        "contract_version": PAPER_CONTRACT_VERSION,
        "paper_id": PAPER_ARXIV_ID,
        "claim": "unit-test paper artifact",
    }
    contract = require_artifact_alignment(
        {"extra": {"alignment": paper_alignment}},
        expected_track="paper_cmd",
        artifact_name="checkpoint",
    )
    assert contract.track == "paper_cmd"

    with pytest.raises(ValueError, match="no alignment provenance"):
        require_artifact_alignment(
            {"extra": {}}, expected_track="paper_cmd", artifact_name="checkpoint"
        )

    robust_alignment = {**paper_alignment, "track": "ours_robust"}
    with pytest.raises(ValueError, match="belongs to alignment track 'ours_robust'"):
        require_artifact_alignment(
            {"extra": {"alignment": robust_alignment}},
            expected_track="paper_cmd",
            artifact_name="checkpoint",
        )


def test_paper_architecture_requires_explicit_alignment_metadata() -> None:
    config = {"model": {"architecture": "paper_cmd"}}

    with pytest.raises(ValueError, match="requires an explicit alignment block"):
        validate_alignment_config(config, component="policy")


def test_explicit_track_requires_the_frozen_contract_version() -> None:
    config = yaml.safe_load((ROOT / "configs/policy/policy_reinforce.yaml").read_text())
    del config["alignment"]["contract_version"]

    with pytest.raises(ValueError, match=r"alignment\.contract_version"):
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


def test_paper_policy_requires_the_pretrained_diffusion_gat() -> None:
    config = yaml.safe_load((ROOT / "configs/policy/policy_reinforce_paper_cmd.yaml").read_text())

    with pytest.raises(ValueError, match="global_gat_checkpoint"):
        build_policy_from_config(config["model"])


def test_unknown_alignment_track_and_architecture_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"alignment\.track"):
        resolve_alignment_contract({"alignment": {"track": "paperish", "claim": "invalid"}})
    with pytest.raises(ValueError, match=r"model\.architecture"):
        build_policy_from_config({"architecture": "mystery"})
