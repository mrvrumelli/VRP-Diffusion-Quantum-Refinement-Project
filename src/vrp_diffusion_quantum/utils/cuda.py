"""CUDA environment diagnostics and a minimal autograd smoke check."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

__all__ = ["cuda_diagnostics"]


def _system_memory_bytes() -> int | None:
    """Best-effort physical RAM detection without adding a runtime dependency."""
    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        return None
    try:
        return int(sysconf("SC_PAGE_SIZE")) * int(sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        return None


def _nvidia_smi_query() -> list[str] | None:
    """Return one compact line per NVIDIA GPU when ``nvidia-smi`` is available."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def cuda_diagnostics(*, run_backward: bool = True) -> dict[str, Any]:
    """Return serializable CUDA/PyTorch environment details and smoke-check status."""
    available = bool(torch.cuda.is_available())
    disk = shutil.disk_usage(Path.cwd())
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "system_memory_bytes": _system_memory_bytes(),
        "working_disk_total_bytes": int(disk.total),
        "working_disk_free_bytes": int(disk.free),
        "nvidia_smi_query": _nvidia_smi_query(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "cuda_available": available,
        "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
        "devices": [],
        "backward_smoke_requested": run_backward,
        "backward_smoke_passed": False,
    }
    if not available:
        return report

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
                "free_memory_bytes": int(free_bytes),
                "runtime_total_memory_bytes": int(total_bytes),
            }
        )
    report["devices"] = devices

    if run_backward:
        value = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
        loss = value.square().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
        report["backward_smoke_passed"] = bool(
            value.grad is not None and torch.isfinite(value.grad).all()
        )
    return report
