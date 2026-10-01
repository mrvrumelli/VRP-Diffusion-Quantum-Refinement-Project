"""Shared validation at binary solver and subproblem boundaries."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def binary_vector(
    values: npt.ArrayLike, size: int, *, name: str = "state"
) -> npt.NDArray[np.float64]:
    """Validate shape and binary values before any integer conversion."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all((vector == 0.0) | (vector == 1.0)):
        raise ValueError(f"{name} must be binary")
    return vector
