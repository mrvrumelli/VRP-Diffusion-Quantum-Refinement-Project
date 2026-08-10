"""CVRP x9 augmentation: original + 4 geometric + 4 customer-relabeling views.

Geo views transform coordinates (and keep ``M``/routes aligned in node order).
Customer relabeling permutes coordinates, demands, routes, and ``M`` together, so every
augmented example is exactly equivalent to its source and remains feasible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution

__all__ = [
    "AUGMENT_NUM",
    "D4_NUM_TRANSFORMS",
    "augment_example",
    "augment_example_d4",
    "augment_example_node_shuffle",
    "expand_examples",
    "sample_augment_views",
    "transform_coords_d4",
]

D4_NUM_TRANSFORMS = 8
AUGMENT_NUM = 9  # original + 4 geometric + 4 customer-relabeling views

# Four "main" geo views (skip identity — that is variant 0).
_GEO_D4_KS = (1, 2, 3, 4)  # 90°, 180°, 270°, reflect-x


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


def augment_example_node_shuffle(
    example: CVRPExample,
    *,
    rng: np.random.Generator | None = None,
    variant: int = 0,
) -> CVRPExample:
    """Permute customer ids globally; remap coords, demands, routes, and ``M``."""
    if variant < 0:
        raise ValueError(f"variant must be non-negative, got {variant}")
    inst = example.instance
    n = int(inst.n_customers)
    if n <= 1:
        return example
    if rng is None:
        seed = 0 if inst.seed is None else int(inst.seed)
        rng = np.random.default_rng(seed + 91_001 + 1_009 * variant)
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)

    cust_nodes = inst.customer_node_indices()
    new_coords = np.asarray(inst.coords, dtype=np.float64).copy()
    new_demands = np.asarray(inst.demands, dtype=np.float64).copy()
    old_c = new_coords[cust_nodes].copy()
    old_d = new_demands[cust_nodes].copy()
    for new_c, old_idx in enumerate(perm):
        new_coords[cust_nodes[new_c]] = old_c[old_idx]
        new_demands[cust_nodes[new_c]] = old_d[old_idx]

    new_routes = [[int(inv[c]) for c in route] for route in example.solution.routes]
    m = np.asarray(example.constraint_matrix, dtype=np.int64)
    new_m = m[np.ix_(perm, perm)]

    new_instance = CVRPInstance(
        coords=new_coords,
        demands=new_demands,
        capacity=float(inst.capacity),
        depot_index=int(inst.depot_index),
        instance_id=f"{inst.instance_id}_nsh{variant}",
        n_customers=n,
        seed=inst.seed,
        generator_settings={
            **inst.generator_settings,
            "node_shuffle": 1,
            "node_shuffle_variant": variant,
        },
    )
    new_solution = LabeledSolution(
        routes=new_routes,
        cost=float(example.solution.cost),
        num_vehicles=int(example.solution.num_vehicles),
        feasible=bool(example.solution.feasible),
        solver_name=example.solution.solver_name,
        time_budget=example.solution.time_budget,
        seed=example.solution.seed,
        runtime_seconds=float(example.solution.runtime_seconds),
    )
    return CVRPExample(instance=new_instance, solution=new_solution, constraint_matrix=new_m)


def augment_example(example: CVRPExample, variant: int) -> CVRPExample:
    """Return one of nine label-preserving geometric or customer-relabeling views."""
    if variant < 0 or variant >= AUGMENT_NUM:
        raise ValueError(f"variant must be in 0..{AUGMENT_NUM - 1}, got {variant}")
    if variant == 0:
        return example
    if variant <= 4:
        return augment_example_d4(example, _GEO_D4_KS[variant - 1])
    return augment_example_node_shuffle(example, variant=variant - 5)


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
