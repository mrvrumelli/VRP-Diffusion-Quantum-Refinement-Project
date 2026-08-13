"""Shared CLI runtime helpers (device, MLflow URI, plot cache)."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "DEFAULT_MLFLOW_DB",
    "capture_rng_state",
    "configure_plot_cache",
    "default_mlflow_tracking_uri",
    "resolve_device",
    "restore_rng_state",
    "seed_everything",
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MLFLOW_DB = _PACKAGE_ROOT / "outputs" / "mlflow.db"


def resolve_device(requested: str | None = "auto") -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; run `python scripts/check_cuda.py "
            "--require-cuda` before training"
        )
    return device


def seed_everything(seed: int, *, deterministic: bool = False) -> dict[str, bool | int]:
    """Seed Python, NumPy, Torch, and CUDA and configure deterministic algorithm handling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return {
        "seed": int(seed),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch, and available CUDA RNG state for checkpoints."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state previously returned by :func:`capture_rng_state`."""
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def default_mlflow_tracking_uri(db_path: Path | None = None) -> str:
    path = db_path or DEFAULT_MLFLOW_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve().as_posix()}"


def configure_plot_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vrp_diffusion_quantum_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
