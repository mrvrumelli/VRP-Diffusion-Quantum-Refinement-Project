"""Tests for shared runtime and reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.utils.runtime import (
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    seed_everything,
)


def test_seed_everything_reproduces_python_numpy_and_torch() -> None:
    first_settings = seed_everything(123, deterministic=True)
    first = (random.random(), float(np.random.random()), torch.rand(3))

    second_settings = seed_everything(123, deterministic=True)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert first_settings == second_settings
    assert first_settings["deterministic_algorithms"] is True

    seed_everything(0, deterministic=False)


def test_resolve_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match=r"check_cuda\.py"):
        resolve_device("cuda")


def test_capture_and_restore_rng_state() -> None:
    seed_everything(77)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(3))

    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
