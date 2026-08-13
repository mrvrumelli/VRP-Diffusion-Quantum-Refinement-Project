"""Tests for CUDA diagnostics that remain valid on CPU-only development machines."""

from __future__ import annotations

from vrp_diffusion_quantum.utils.cuda import cuda_diagnostics


def test_cuda_diagnostics_reports_environment() -> None:
    report = cuda_diagnostics(run_backward=False)

    assert report["python_version"]
    assert report["torch_version"]
    assert report["platform"]
    assert report["cpu_count"]
    assert report["working_disk_total_bytes"] > 0
    assert "nvidia_smi_query" in report
    assert isinstance(report["cuda_available"], bool)
    assert isinstance(report["cuda_device_count"], int)
    assert isinstance(report["devices"], list)
    assert report["backward_smoke_requested"] is False
    assert report["backward_smoke_passed"] is False
