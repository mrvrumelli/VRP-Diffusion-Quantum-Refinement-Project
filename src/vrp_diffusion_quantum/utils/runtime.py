"""Shared CLI runtime helpers (device, MLflow URI, plot cache)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

__all__ = [
    "DEFAULT_MLFLOW_DB",
    "configure_plot_cache",
    "default_mlflow_tracking_uri",
    "resolve_device",
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MLFLOW_DB = _PACKAGE_ROOT / "outputs" / "mlflow.db"


def resolve_device(requested: str | None = "auto") -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def default_mlflow_tracking_uri(db_path: Path | None = None) -> str:
    path = db_path or DEFAULT_MLFLOW_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve().as_posix()}"


def configure_plot_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vrp_diffusion_quantum_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
