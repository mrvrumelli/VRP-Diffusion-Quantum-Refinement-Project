"""Fail-fast contracts for paper reproduction and project extensions.

The alignment metadata prevents a checkpoint produced by the project's extended architecture from
being reported as a faithful CMD-paper baseline. Legacy configs remain loadable, but any config
that explicitly claims ``paper_cmd`` is checked against the paper's declared hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "PAPER_ARXIV_ID",
    "AlignmentComponent",
    "AlignmentContract",
    "AlignmentTrack",
    "resolve_alignment_contract",
    "validate_alignment_config",
]

PAPER_ARXIV_ID = "arxiv:2603.07568v1"

AlignmentTrack = Literal["paper_cmd", "ours_robust", "legacy_unspecified"]
AlignmentComponent = Literal["gat_pretrain", "diffusion", "policy"]


@dataclass(frozen=True)
class AlignmentContract:
    """Resolved experiment identity stored with configs and checkpoints."""

    track: AlignmentTrack
    paper_id: str | None
    claim: str

    def as_dict(self) -> dict[str, str | None]:
        return {"track": self.track, "paper_id": self.paper_id, "claim": self.claim}


_PAPER_REQUIREMENTS: dict[AlignmentComponent, dict[str, object]] = {
    "gat_pretrain": {
        "model.hidden_dim": 128,
        "model.gat_num_layers": 5,
        "model.gat_num_heads": 8,
        "training.batch_size": 64,
        "training.learning_rate": 1e-4,
        "training.augmentation": False,
        "training.online_augmentation": True,
        "training.augmentation_recipe": "paper_cmd_labeled",
    },
    "diffusion": {
        "model.hidden_dim": 128,
        "model.num_layers": 5,
        "model.time_embed_dim": 128,
        "model.node_encoder_type": "gat",
        "model.gat_num_layers": 5,
        "model.gat_num_heads": 8,
        "model.freeze_node_encoder": True,
        "schedule.num_timesteps": 1000,
        "schedule.beta_start": 1e-4,
        "schedule.beta_end": 2e-2,
        "training.epochs": 50,
        "training.learning_rate": 1e-4,
        "training.batch_size": 32,
        "training.augmentation": False,
        "training.online_augmentation": True,
        "training.augmentation_recipe": "paper_cmd_labeled",
    },
    "policy": {
        "model.architecture": "paper_cmd",
        "model.embedding_dim": 128,
        "model.global_num_layers": 5,
        "model.global_num_heads": 8,
        "model.local_num_layers": 5,
        "model.local_num_heads": 8,
        "model.decoder_num_heads": 8,
        "model.clip_constant": 10.0,
        "model.adjacency_mode": "hard",
        "model.use_local_encoder": True,
        "model.use_local_pointer": True,
        "model.use_global_pointer": True,
        "prior.source": "denoiser",
        "prior.num_inference_steps": 50,
        "prior.sampler": "skipped_posterior",
        "training.epochs": 100,
        "training.learning_rate": 1e-4,
        "training.weight_decay": 1e-6,
        "training.batch_size": 32,
        "training.start_node_cap": 100,
        "training.baseline": "multi_start",
    },
}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nested_value(config: Mapping[str, Any], path: str) -> object:
    value: object = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _same_value(actual: object, expected: object) -> bool:
    if (
        isinstance(expected, float)
        and isinstance(actual, int | float)
        and not isinstance(actual, bool)
    ):
        return abs(float(actual) - expected) <= max(1e-12, abs(expected) * 1e-9)
    return actual == expected


_MISSING = object()


def resolve_alignment_contract(config: Mapping[str, Any]) -> AlignmentContract:
    """Parse top-level alignment metadata, preserving old configs as explicitly legacy."""
    raw = config.get("alignment")
    if raw is None:
        return AlignmentContract(
            track="legacy_unspecified",
            paper_id=None,
            claim="legacy configuration; no paper-fidelity claim",
        )

    alignment = _mapping(raw, name="alignment")
    track = str(alignment.get("track", ""))
    if track not in {"paper_cmd", "ours_robust"}:
        raise ValueError("alignment.track must be 'paper_cmd' or 'ours_robust'")

    paper_id_value = alignment.get("paper_id")
    paper_id = None if paper_id_value is None else str(paper_id_value)
    claim = str(alignment.get("claim", "")).strip()
    if not claim:
        raise ValueError("alignment.claim must be a non-empty description")
    if track == "paper_cmd" and paper_id != PAPER_ARXIV_ID:
        raise ValueError(
            f"paper_cmd requires alignment.paper_id={PAPER_ARXIV_ID!r}, got {paper_id!r}"
        )
    return AlignmentContract(track=track, paper_id=paper_id, claim=claim)  # type: ignore[arg-type]


def validate_alignment_config(
    config: Mapping[str, Any], *, component: AlignmentComponent
) -> AlignmentContract:
    """Validate track identity and paper-declared values for one training component."""
    contract = resolve_alignment_contract(config)
    model = _mapping(config.get("model", {}), name="model")
    architecture = str(model.get("architecture", "ours_robust"))

    if contract.track == "legacy_unspecified":
        if architecture == "paper_cmd":
            raise ValueError("model.architecture='paper_cmd' requires an explicit alignment block")
        return contract

    if contract.track == "ours_robust":
        if component == "policy" and architecture != "ours_robust":
            raise ValueError(
                "alignment.track='ours_robust' requires model.architecture='ours_robust'"
            )
        return contract

    mismatches: list[str] = []
    for path, expected in _PAPER_REQUIREMENTS[component].items():
        actual = _nested_value(config, path)
        if actual is _MISSING:
            mismatches.append(f"{path} is missing (expected {expected!r})")
        elif not _same_value(actual, expected):
            mismatches.append(f"{path}={actual!r} (expected {expected!r})")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"paper_cmd {component} contract mismatch: {details}")
    return contract
