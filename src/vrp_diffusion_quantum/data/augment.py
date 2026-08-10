"""CVRP x9 augmentation using distance-preserving coordinate transforms.

The original example is expanded with four D4 views and four 45-degree-offset rotations.
Demands, routes, and ``M`` stay in the same node order. Customer relabelings are deliberately
excluded: the current models are permutation-equivariant, so those views duplicate the same
training signal instead of adding useful diversity.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance

__all__ = [
    "AUGMENT_NUM",
    "D4_NUM_TRANSFORMS",
    "augment_example",
    "augment_example_d4",
    "augment_example_rotation",
    "expand_examples",
    "sample_augment_views",
    "transform_coords_d4",
    "transform_coords_rotation",
]

D4_NUM_TRANSFORMS = 8
AUGMENT_NUM = 9  # original + 4 D4 views + 4 arbitrary-angle rotations

# Four "main" geo views (skip identity — that is variant 0).
_GEO_D4_KS = (1, 2, 3, 4)  # 90°, 180°, 270°, reflect-x
_ROTATION_ANGLES_DEGREES = (45.0, 135.0, 225.0, 315.0)


def transform_coords_d4(coords: npt.NDArray[np.floating], k: int) -> npt.NDArray[np.float64]:
    if k < 0 or k >= D4_NUM_TRANSFORMS:
        raise ValueError(f"k must be in 0..{D4_NUM_TRANSFORMS - 1}, got {k}")
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(f"coords must have shape [N, 2], got {xy.shape}")
    x, y = xy[:, 0], xy[:, 1]
    if k == 0:
        out = np.stack([x, y], axis=-1)
    elif k == 1:
        out = np.stack([y, 1.0 - x], axis=-1)
    elif k == 2:
        out = np.stack([1.0 - x, 1.0 - y], axis=-1)
    elif k == 3:
        out = np.stack([1.0 - y, x], axis=-1)
    elif k == 4:
        out = np.stack([1.0 - x, y], axis=-1)
    elif k == 5:
        out = np.stack([x, 1.0 - y], axis=-1)
    elif k == 6:
        out = np.stack([y, x], axis=-1)
    else:
        out = np.stack([1.0 - y, 1.0 - x], axis=-1)
    return out


def augment_example_d4(example: CVRPExample, k: int) -> CVRPExample:
    inst = example.instance
    new_instance = CVRPInstance(
        coords=transform_coords_d4(inst.coords, k),
        demands=np.asarray(inst.demands, dtype=np.float64).copy(),
        capacity=float(inst.capacity),
        depot_index=int(inst.depot_index),
        instance_id=f"{inst.instance_id}_d4{k}",
        n_customers=int(inst.n_customers),
        seed=inst.seed,
        generator_settings={**inst.generator_settings, "d4": int(k)},
    )
    return CVRPExample(
        instance=new_instance,
        solution=example.solution,
        constraint_matrix=np.asarray(example.constraint_matrix, dtype=np.int64).copy(),
    )


def transform_coords_rotation(
    coords: npt.NDArray[np.floating], angle_degrees: float
) -> npt.NDArray[np.float64]:
    """Rotate coordinates around the unit-square center without changing distances."""
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(f"coords must have shape [N, 2], got {xy.shape}")
    if not np.isfinite(angle_degrees):
        raise ValueError(f"angle_degrees must be finite, got {angle_degrees}")
    radians = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    center = np.array([0.5, 0.5], dtype=np.float64)
    return (xy - center) @ rotation.T + center


def augment_example_rotation(example: CVRPExample, angle_degrees: float) -> CVRPExample:
    """Rotate all nodes while preserving distances, feasibility, routes, and ``M``."""
    inst = example.instance
    new_instance = CVRPInstance(
        coords=transform_coords_rotation(inst.coords, angle_degrees),
        demands=np.asarray(inst.demands, dtype=np.float64).copy(),
        capacity=float(inst.capacity),
        depot_index=int(inst.depot_index),
        instance_id=f"{inst.instance_id}_rot{angle_degrees:g}",
        n_customers=int(inst.n_customers),
        seed=inst.seed,
        generator_settings={
            **inst.generator_settings,
            "rotation_degrees": float(angle_degrees),
        },
    )
    return CVRPExample(
        instance=new_instance,
        solution=example.solution,
        constraint_matrix=np.asarray(example.constraint_matrix, dtype=np.int64).copy(),
    )


def augment_example(example: CVRPExample, variant: int) -> CVRPExample:
    """Return one of nine distance- and label-preserving geometric views."""
    if variant < 0 or variant >= AUGMENT_NUM:
        raise ValueError(f"variant must be in 0..{AUGMENT_NUM - 1}, got {variant}")
    if variant == 0:
        return example
    if variant <= 4:
        return augment_example_d4(example, _GEO_D4_KS[variant - 1])
    return augment_example_rotation(example, _ROTATION_ANGLES_DEGREES[variant - 5])


def expand_examples(examples: list[CVRPExample]) -> list[CVRPExample]:
    return [augment_example(ex, v) for ex in examples for v in range(AUGMENT_NUM)]


def sample_augment_views(
    example: CVRPExample,
    k: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[CVRPExample]:
    if k < 1 or k > AUGMENT_NUM:
        raise ValueError(f"k must be in 1..{AUGMENT_NUM}, got {k}")
    if rng is None:
        seed = 0 if example.instance.seed is None else int(example.instance.seed)
        rng = np.random.default_rng(seed)
    variants = rng.choice(AUGMENT_NUM, size=k, replace=False)
    return [augment_example(example, int(v)) for v in variants]
