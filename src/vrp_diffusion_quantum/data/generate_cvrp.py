from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "CAPACITY_FLOOR_BY_SIZE",
    "CLUSTER_SEEDS_RANGE",
    "DEFAULT_CAPACITY_BY_SIZE",
    "DEMAND_HIGH",
    "DEMAND_LOW",
    "DEMAND_RANGES",
    "NORMALIZED_CLUSTER_DECAY",
    "ROUTE_SIZE_RANGES",
    "CVRPDataset",
    "CVRPInstance",
    "CustomerMode",
    "DemandMode",
    "DepotMode",
    "capacity_floor",
    "demand_bounds_for_mode",
    "generate_dataset",
    "generate_instance",
    "load_dataset",
    "resolve_capacity",
    "sample_capacity_from_weights",
    "save_dataset",
    "suggested_capacity",
]

DepotMode = Literal["random", "center", "corner"]
CustomerMode = Literal["random", "clustered", "random_clustered"]
DemandMode = Literal[
    "uniform",
    "unitary",
    # Named <demand magnitude>_demand_<spread>_spread: e.g. "small_demand_wide_spread" means
    # small demand values drawn over a wide range (high coefficient of variation).
    "small_demand_wide_spread",
    "small_demand_narrow_spread",
    "large_demand_wide_spread",
    "large_demand_narrow_spread",
    "quadrant",
    "many_small_few_large",
    # Explicit [demand_low, demand_high] range (1..100); set via demand_low/demand_high.
    "custom",
]

# Classic size anchors for capacity floors (n → floor). Any other size is interpolated
# / extrapolated by :func:`capacity_floor`. Suggested capacity is max(high_demand, floor).
CAPACITY_FLOOR_BY_SIZE: dict[int, int] = {20: 20, 50: 30, 100: 50}
DEFAULT_CAPACITY_BY_SIZE: dict[int, int] = CAPACITY_FLOOR_BY_SIZE

DEMAND_LOW: int = 1
DEMAND_HIGH: int = 9

DEMAND_RANGES: dict[str, tuple[int, int]] = {
    "small_demand_wide_spread": (1, 10),
    "small_demand_narrow_spread": (5, 10),
    "large_demand_wide_spread": (1, 100),
    "large_demand_narrow_spread": (50, 100),
}
_QUADRANT_EVEN_RANGE: tuple[int, int] = (1, 50)
_QUADRANT_ODD_RANGE: tuple[int, int] = (51, 100)
_SL_SMALL_RANGE: tuple[int, int] = (1, 10)
_SL_LARGE_RANGE: tuple[int, int] = (50, 100)
_SL_SMALL_FRACTION_RANGE: tuple[float, float] = (0.70, 0.95)

CLUSTER_SEEDS_RANGE: tuple[int, int] = (3, 8)

NORMALIZED_CLUSTER_DECAY: float = 40.0 / 1000.0

ROUTE_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "very_short": (3.0, 5.0),
    "short": (5.0, 8.0),
    "medium": (8.0, 12.0),
    "long": (12.0, 16.0),
    "very_long": (16.0, 25.0),
    "ultra_long": (25.0, 50.0),
}


def capacity_floor(n_customers: int) -> int:
    """Return a size-based capacity floor for any customer count.

    Anchors match the classic defaults ``{20: 20, 50: 30, 100: 50}``. Values between
    anchors are linearly interpolated; below 20 the floor tracks ``n``; above 100 the
    slope from the 50→100 segment (``0.4`` per customer) continues.
    """
    n = int(n_customers)
    if n <= 0:
        raise ValueError(f"n_customers must be positive, got {n_customers}")
    if n in CAPACITY_FLOOR_BY_SIZE:
        return int(CAPACITY_FLOOR_BY_SIZE[n])
    if n < 20:
        return max(1, n)
    if n < 50:
        return int(round(20 + (30 - 20) * (n - 20) / 30))
    if n < 100:
        return int(round(30 + (50 - 30) * (n - 50) / 50))
    return int(round(50 + 0.4 * (n - 100)))


def suggested_capacity(n_customers: int, high_demand: int) -> int:
    """Return a suitable default capacity for ``n_customers`` given ``high_demand``.

    Uses :func:`capacity_floor` (anchors ``20→20``, ``50→30``, ``100→50``, interpolated
    elsewhere), then clamps up to ``high_demand`` so capacity is never below the demand
    ceiling.
    """
    if high_demand < 1:
        raise ValueError(f"high_demand must be >= 1, got {high_demand}")
    return max(int(high_demand), capacity_floor(n_customers))


def resolve_capacity(n_customers: int, capacity: int | None) -> int:
    """Return the vehicle capacity for an instance of the given size.

    If ``capacity`` is provided it is used directly; otherwise :func:`capacity_floor`
    supplies a default for any positive customer count.
    """
    if capacity is not None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        return capacity
    return capacity_floor(n_customers)


def sample_capacity_from_weights(
    rng: np.random.Generator,
    min_cap: int,
    max_cap: int,
    weights: list[tuple[int, float]] | None = None,
) -> int:
    """Sample an integer capacity in ``[min_cap, max_cap]``.

    When ``weights`` is ``None`` or empty, sampling is uniform. Otherwise ``weights`` is a
    list of ``(capacity, relative_weight)`` control points; a piecewise-linear density is
    interpolated over every integer in the range, normalized, and sampled.
    """
    lo = int(min_cap)
    hi = int(max_cap)
    if hi < lo:
        raise ValueError(f"max_cap ({hi}) must be >= min_cap ({lo})")
    if lo < 1:
        raise ValueError(f"min_cap must be >= 1, got {lo}")

    values = np.arange(lo, hi + 1, dtype=np.int64)
    if not weights:
        return int(rng.integers(lo, hi + 1))

    xs: list[int] = []
    ws: list[float] = []
    for raw_x, raw_w in sorted(weights, key=lambda t: int(t[0])):
        x = int(raw_x)
        w = float(raw_w)
        if x < lo or x > hi or w < 0:
            continue
        if xs and xs[-1] == x:
            ws[-1] = max(ws[-1], w)
            continue
        xs.append(x)
        ws.append(w)

    if not xs:
        return int(rng.integers(lo, hi + 1))

    dens = np.interp(values.astype(np.float64), np.asarray(xs, dtype=np.float64), np.asarray(ws))
    dens = np.maximum(dens, 0.0)
    total = float(dens.sum())
    if total <= 0.0:
        return int(rng.integers(lo, hi + 1))
    probs = dens / total
    return int(rng.choice(values, p=probs))


def demand_bounds_for_mode(demand_mode: DemandMode) -> tuple[int, int]:
    """Return ``(low, high)`` demand bounds associated with ``demand_mode``.

    For structural modes (``quadrant``, ``many_small_few_large``) this is the envelope
    ``(1, 100)``. For ``custom`` callers must supply ``demand_low`` / ``demand_high``.
    """
    if demand_mode == "uniform":
        return DEMAND_LOW, DEMAND_HIGH
    if demand_mode == "unitary":
        return 1, 1
    if demand_mode in DEMAND_RANGES:
        return DEMAND_RANGES[demand_mode]
    if demand_mode in ("quadrant", "many_small_few_large"):
        return 1, 100
    if demand_mode == "custom":
        raise ValueError("custom demand_mode requires demand_low and demand_high")
    raise ValueError(f"unknown demand_mode: {demand_mode!r}")


def _validate_demand_bounds(demand_low: int, demand_high: int) -> tuple[int, int]:
    low = int(demand_low)
    high = int(demand_high)
    if not (1 <= low <= 100 and 1 <= high <= 100):
        raise ValueError(
            f"demand_low/demand_high must be in 1..100, got low={low}, high={high}"
        )
    if low > high:
        raise ValueError(f"demand_low ({low}) cannot exceed demand_high ({high})")
    return low, high


@dataclass(frozen=True)
class CVRPInstance:
    """A single CVRP instance with the depot stored as node ``depot_index``.

    Attributes:
        coords: ``(n_nodes, 2)`` float array in ``[0, 1]``; row ``depot_index`` is the depot.
        demands: ``(n_nodes,)`` integer array; the depot entry is ``0``.
        capacity: vehicle capacity shared by all routes.
        depot_index: index of the depot node inside ``coords``/``demands`` (default ``0``).
    """

    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    depot_index: int = 0

    @property
    def n_nodes(self) -> int:
        """Number of nodes including the depot."""
        return int(self.coords.shape[0])

    @property
    def n_customers(self) -> int:
        """Number of customer nodes (all nodes except the depot)."""
        return self.n_nodes - 1

    @property
    def depot_coords(self) -> np.ndarray:
        """The depot coordinate, shape ``(2,)``."""
        return self.coords[self.depot_index]

    @property
    def customer_coords(self) -> np.ndarray:
        """Customer coordinates, shape ``(n_customers, 2)``."""
        return np.delete(self.coords, self.depot_index, axis=0)

    @property
    def customer_demands(self) -> np.ndarray:
        """Customer demands, shape ``(n_customers,)``."""
        return np.delete(self.demands, self.depot_index)

    def normalized_demands(self) -> np.ndarray:
        """Demands scaled by capacity (``demand / capacity``), a model-input convention.

        The returned array covers all nodes; the depot entry is ``0``. Values lie in
        ``(0, 1]`` for customers because every demand is at most the capacity.
        """
        return self.demands.astype(np.float64) / self.capacity

    def validate(self) -> None:
        """Raise ``ValueError`` if the instance violates shape or capacity invariants."""
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (n_nodes, 2), got {self.coords.shape}")
        if self.demands.shape != (self.n_nodes,):
            raise ValueError(
                f"demands must have shape ({self.n_nodes},), got {self.demands.shape}"
            )
        if not (0 <= self.depot_index < self.n_nodes):
            raise ValueError(f"depot_index {self.depot_index} out of range for {self.n_nodes}")
        if self.coords.min() < 0.0 or self.coords.max() > 1.0:
            raise ValueError("coords must be normalized within [0, 1]")
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.demands[self.depot_index] != 0:
            raise ValueError("depot demand must be 0")
        customer_demands = self.customer_demands
        if customer_demands.min() <= 0:
            raise ValueError("every customer demand must be strictly positive")
        if customer_demands.max() > self.capacity:
            raise ValueError(
                "no single demand may exceed the vehicle capacity; pass a larger capacity or "
                "route_size for this demand_mode"
            )


def _sample_clustered_coords(
    n_customers: int, rng: np.random.Generator, decay: float
) -> np.ndarray:
    """Sample clustered customer coordinates via Uchoa-style exponential attraction.

    A number of seed customers (``UD[3, 8]``) are placed uniformly; the remaining customers are
    drawn from the mixture density proportional to ``sum_s exp(-dist(p, s) / decay)``. Each
    non-seed customer picks a seed uniformly and samples a radial offset from ``Gamma(2, decay)``
    (the exact 2D continuous form of a single exponential kernel), then a uniform angle.
    """
    n_seeds = int(rng.integers(CLUSTER_SEEDS_RANGE[0], CLUSTER_SEEDS_RANGE[1] + 1))
    n_seeds = min(n_seeds, n_customers)
    seeds = rng.random((n_seeds, 2))

    remaining = n_customers - n_seeds
    if remaining <= 0:
        coords = seeds[:n_customers]
    else:
        seed_choice = rng.integers(0, n_seeds, size=remaining)
        radius = rng.gamma(2.0, decay, size=remaining)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=remaining)
        offsets = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        attracted = seeds[seed_choice] + offsets
        coords = np.vstack([seeds, attracted])

    np.clip(coords, 0.0, 1.0, out=coords)
    return coords


def _sample_customer_coords(
    n_customers: int, rng: np.random.Generator, customer_mode: CustomerMode, decay: float
) -> np.ndarray:
    """Sample ``n_customers`` customer coordinates according to ``customer_mode``."""
    if customer_mode == "random":
        return rng.random((n_customers, 2))
    if customer_mode == "clustered":
        return _sample_clustered_coords(n_customers, rng, decay)
    if customer_mode == "random_clustered":
        n_random = n_customers // 2
        n_clustered = n_customers - n_random
        random_coords = rng.random((n_random, 2))
        clustered_coords = _sample_clustered_coords(n_clustered, rng, decay)
        return np.vstack([random_coords, clustered_coords])
    raise ValueError(f"unknown customer_mode: {customer_mode!r}")


def _sample_demands(
    customer_coords: np.ndarray,
    rng: np.random.Generator,
    demand_mode: DemandMode,
    *,
    demand_low: int | None = None,
    demand_high: int | None = None,
) -> np.ndarray:
    """Sample integer customer demands according to ``demand_mode`` / bounds."""
    n = int(customer_coords.shape[0])

    if demand_mode == "unitary":
        return np.ones(n, dtype=np.int64)
    if demand_mode == "quadrant":
        right = customer_coords[:, 0] >= 0.5
        top = customer_coords[:, 1] >= 0.5
        even_quadrant = right != top
        demands = np.empty(n, dtype=np.int64)
        even_count = int(np.count_nonzero(even_quadrant))
        demands[even_quadrant] = rng.integers(
            _QUADRANT_EVEN_RANGE[0], _QUADRANT_EVEN_RANGE[1] + 1, size=even_count
        )
        demands[~even_quadrant] = rng.integers(
            _QUADRANT_ODD_RANGE[0], _QUADRANT_ODD_RANGE[1] + 1, size=n - even_count
        )
        return demands
    if demand_mode == "many_small_few_large":
        small_fraction = float(rng.uniform(*_SL_SMALL_FRACTION_RANGE))
        is_small = rng.random(n) < small_fraction
        demands = np.empty(n, dtype=np.int64)
        small_count = int(np.count_nonzero(is_small))
        demands[is_small] = rng.integers(
            _SL_SMALL_RANGE[0], _SL_SMALL_RANGE[1] + 1, size=small_count
        )
        demands[~is_small] = rng.integers(
            _SL_LARGE_RANGE[0], _SL_LARGE_RANGE[1] + 1, size=n - small_count
        )
        return demands

    # Range-based modes: uniform, named DEMAND_RANGES, or custom.
    if demand_low is not None and demand_high is not None:
        low, high = _validate_demand_bounds(demand_low, demand_high)
    elif demand_mode == "custom":
        raise ValueError("custom demand_mode requires demand_low and demand_high")
    elif demand_mode == "uniform":
        low, high = DEMAND_LOW, DEMAND_HIGH
    elif demand_mode in DEMAND_RANGES:
        low, high = DEMAND_RANGES[demand_mode]
    else:
        raise ValueError(f"unknown demand_mode: {demand_mode!r}")
    return rng.integers(low, high + 1, size=n)


def _sample_route_size(route_size: float | str, rng: np.random.Generator) -> float:
    """Resolve a target average route size ``r`` from a number or a named preset."""
    if isinstance(route_size, bool):  # guard: bool is an int subclass
        raise ValueError(f"route_size must be a number or preset name, got {route_size!r}")
    if isinstance(route_size, int | float):
        if route_size <= 0:
            raise ValueError(f"route_size must be positive, got {route_size}")
        return float(route_size)
    if route_size in ROUTE_SIZE_RANGES:
        low, high = ROUTE_SIZE_RANGES[route_size]
        return float(rng.uniform(low, high))
    raise ValueError(
        f"unknown route_size preset {route_size!r}; use a number or one of "
        f"{sorted(ROUTE_SIZE_RANGES)}"
    )


def _resolve_instance_capacity(
    n_customers: int,
    demands: np.ndarray,
    capacity: int | None,
    route_size: float | str | None,
    rng: np.random.Generator,
    *,
    random_capacity: bool = False,
    capacity_max: int | None = None,
    capacity_weights: list[tuple[int, float]] | None = None,
    demand_high_floor: int | None = None,
) -> int:
    """Resolve capacity; never below ``max(demands)`` or the high-demand floor."""
    min_cap = max(int(demands.max()), int(demand_high_floor or 0), 1)

    if random_capacity:
        cap_hi = int(capacity_max) if capacity_max is not None else max(min_cap * 3, min_cap + 50)
        if cap_hi < min_cap:
            raise ValueError(
                f"capacity_max ({cap_hi}) must be >= high-demand floor ({min_cap})"
            )
        return sample_capacity_from_weights(rng, min_cap, cap_hi, capacity_weights)

    if capacity is not None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if capacity < min_cap:
            raise ValueError(
                f"capacity ({capacity}) cannot be lower than high demand / max demand "
                f"({min_cap})"
            )
        return int(capacity)

    if route_size is not None:
        r = _sample_route_size(route_size, rng)
        derived = round(r * float(demands.mean()))
        return max(derived, min_cap)

    default = resolve_capacity(n_customers, None)
    if default < min_cap:
        raise ValueError(
            f"default capacity ({default}) is below high-demand floor ({min_cap}); "
            "set capacity >= high demand, enable random_capacity, or use route_size"
        )
    return default


def generate_instance(
    n_customers: int,
    rng: np.random.Generator,
    *,
    capacity: int | None = None,
    depot_mode: DepotMode = "random",
    customer_mode: CustomerMode = "random",
    demand_mode: DemandMode = "uniform",
    demand_low: int | None = None,
    demand_high: int | None = None,
    random_demand_bounds: bool = False,
    route_size: float | str | None = None,
    random_capacity: bool = False,
    capacity_max: int | None = None,
    capacity_weights: list[tuple[int, float]] | None = None,
    cluster_decay: float = NORMALIZED_CLUSTER_DECAY,
) -> CVRPInstance:
    """Generate one CVRP instance of ``n_customers`` customers using ``rng``.

    Args:
        n_customers: number of customer nodes (excluding the depot).
        rng: the random generator to draw coordinates and demands from.
        capacity: explicit vehicle capacity; must be >= high demand / max demand.
        depot_mode: ``random`` (uniform), ``center`` (0.5, 0.5), or ``corner`` (0, 0).
        customer_mode: ``random``, ``clustered``, or ``random_clustered``.
        demand_mode: ``uniform`` (default U{1..9}), Uchoa regimes, or ``custom``.
        demand_low / demand_high: optional bounds in ``1..100`` (required for ``custom``).
        random_demand_bounds: if True, sample a fresh ``[low, high]`` in ``1..100`` for
            this instance, then draw demands inside it (overrides preset bounds).
        route_size: target average customers-per-route; derives capacity when ``capacity``
            is ``None`` and ``random_capacity`` is False.
        random_capacity: if True, sample capacity from
            ``[high_demand_floor, capacity_max]`` (default max is about ``3 * floor``).
        capacity_max: upper bound when ``random_capacity`` is True.
        capacity_weights: optional ``(capacity, weight)`` control points shaping the
            random-capacity density (``None`` = uniform).
        cluster_decay: exponential-attraction decay for clustered customers (normalized units).

    Returns:
        A validated :class:`CVRPInstance` with the depot as node ``0``.
    """
    if n_customers <= 0:
        raise ValueError(f"n_customers must be positive, got {n_customers}")

    customer_coords = _sample_customer_coords(n_customers, rng, customer_mode, cluster_decay)

    if depot_mode == "random":
        depot_coords = rng.random((1, 2))
    elif depot_mode == "center":
        depot_coords = np.full((1, 2), 0.5)
    elif depot_mode == "corner":
        depot_coords = np.zeros((1, 2))
    else:  # pragma: no cover - guarded by the DepotMode type
        raise ValueError(f"unknown depot_mode: {depot_mode!r}")

    coords = np.vstack([depot_coords, customer_coords])

    # Resolve per-instance demand bounds.
    mode_for_sample: DemandMode = demand_mode
    low_for_sample = demand_low
    high_for_sample = demand_high
    if random_demand_bounds:
        # Independent low/high in 1..100 with low <= high.
        a = int(rng.integers(1, 101))
        b = int(rng.integers(1, 101))
        low_for_sample, high_for_sample = (a, b) if a <= b else (b, a)
        mode_for_sample = "custom"
    elif demand_mode == "custom":
        if demand_low is None or demand_high is None:
            raise ValueError("custom demand_mode requires demand_low and demand_high")
        low_for_sample, high_for_sample = _validate_demand_bounds(demand_low, demand_high)
    elif demand_mode in ("uniform", *DEMAND_RANGES):
        if demand_low is None or demand_high is None:
            low_for_sample, high_for_sample = demand_bounds_for_mode(demand_mode)
        else:
            low_for_sample, high_for_sample = _validate_demand_bounds(demand_low, demand_high)

    customer_demands = _sample_demands(
        customer_coords,
        rng,
        mode_for_sample,
        demand_low=low_for_sample,
        demand_high=high_for_sample,
    )
    demands = np.zeros(n_customers + 1, dtype=np.int64)
    demands[1:] = customer_demands

    if high_for_sample is not None:
        demand_high_floor = int(high_for_sample)
    elif demand_mode in ("quadrant", "many_small_few_large"):
        demand_high_floor = 100
    else:
        demand_high_floor = int(customer_demands.max())

    resolved_capacity = _resolve_instance_capacity(
        n_customers,
        customer_demands,
        capacity,
        route_size,
        rng,
        random_capacity=random_capacity,
        capacity_max=capacity_max,
        capacity_weights=capacity_weights,
        demand_high_floor=demand_high_floor,
    )

    instance = CVRPInstance(coords=coords, demands=demands, capacity=float(resolved_capacity))
    instance.validate()
    return instance


@dataclass(frozen=True)
class CVRPDataset:
    """A batch of same-size CVRP instances stored as stacked arrays.

    Attributes:
        coords: ``(num_instances, n_nodes, 2)`` float array in ``[0, 1]``.
        demands: ``(num_instances, n_nodes)`` integer array; depot column is ``0``.
        capacity: ``(num_instances,)`` float array of per-instance capacities.
        n_customers: number of customers per instance (excluding the depot).
        seed: the seed that produced this dataset.
        depot_mode: depot placement strategy used.
        customer_mode: customer placement strategy used.
        demand_mode: demand distribution used.
        route_size: route-size spec used to derive capacity ("none" if a fixed capacity was used).
    """

    coords: np.ndarray
    demands: np.ndarray
    capacity: np.ndarray
    n_customers: int
    seed: int
    depot_mode: DepotMode
    customer_mode: CustomerMode = "random"
    demand_mode: DemandMode = "uniform"
    route_size: str = "none"

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __getitem__(self, index: int) -> CVRPInstance:
        return CVRPInstance(
            coords=self.coords[index],
            demands=self.demands[index],
            capacity=float(self.capacity[index]),
        )

    def __iter__(self) -> Iterator[CVRPInstance]:
        for index in range(len(self)):
            yield self[index]


def generate_dataset(
    n_customers: int,
    num_instances: int,
    *,
    seed: int,
    capacity: int | None = None,
    depot_mode: DepotMode = "random",
    customer_mode: CustomerMode = "random",
    demand_mode: DemandMode = "uniform",
    demand_low: int | None = None,
    demand_high: int | None = None,
    random_demand_bounds: bool = False,
    route_size: float | str | None = None,
    random_capacity: bool = False,
    capacity_max: int | None = None,
    capacity_weights: list[tuple[int, float]] | None = None,
    cluster_decay: float = NORMALIZED_CLUSTER_DECAY,
) -> CVRPDataset:
    """Generate ``num_instances`` CVRP instances of a fixed size, reproducibly from ``seed``.

    Instances are drawn sequentially from a single ``numpy.random.default_rng(seed)``, so the
    same generation parameters always yield identical data. See :func:`generate_instance` for
    the meaning of the distribution parameters.
    """
    if num_instances <= 0:
        raise ValueError(f"num_instances must be positive, got {num_instances}")
    rng = np.random.default_rng(seed)

    n_nodes = n_customers + 1
    coords = np.empty((num_instances, n_nodes, 2), dtype=np.float64)
    demands = np.empty((num_instances, n_nodes), dtype=np.int64)
    capacities = np.empty(num_instances, dtype=np.float64)

    for index in range(num_instances):
        instance = generate_instance(
            n_customers,
            rng,
            capacity=capacity,
            depot_mode=depot_mode,
            customer_mode=customer_mode,
            demand_mode=demand_mode,
            demand_low=demand_low,
            demand_high=demand_high,
            random_demand_bounds=random_demand_bounds,
            route_size=route_size,
            random_capacity=random_capacity,
            capacity_max=capacity_max,
            capacity_weights=capacity_weights,
            cluster_decay=cluster_decay,
        )
        coords[index] = instance.coords
        demands[index] = instance.demands
        capacities[index] = instance.capacity

    logger.info(
        "generated dataset n_customers=%d num_instances=%d seed=%d depot=%s customer=%s "
        "demand=%s route_size=%s",
        n_customers,
        num_instances,
        seed,
        depot_mode,
        customer_mode,
        demand_mode,
        route_size,
    )
    return CVRPDataset(
        coords=coords,
        demands=demands,
        capacity=capacities,
        n_customers=n_customers,
        seed=seed,
        depot_mode=depot_mode,
        customer_mode=customer_mode,
        demand_mode=demand_mode,
        route_size="none" if route_size is None else str(route_size),
    )


def _dataset_csv_paths(path: str | Path) -> tuple[Path, Path]:
    """Resolve the (nodes, instances) CSV paths for a dataset base path.

    Accepts a bare stem (``cvrp20``), a stem with any suffix (``cvrp20.csv``), or one of the
    two CSV files themselves (``cvrp20_nodes.csv`` / ``cvrp20_instances.csv``).
    """
    resolved = Path(path)
    name = resolved.name
    if name.endswith("_nodes.csv"):
        stem_name = name[: -len("_nodes.csv")]
    elif name.endswith("_instances.csv"):
        stem_name = name[: -len("_instances.csv")]
    else:
        stem_name = resolved.with_suffix("").name
    nodes_path = resolved.parent / f"{stem_name}_nodes.csv"
    instances_path = resolved.parent / f"{stem_name}_instances.csv"
    return nodes_path, instances_path


def save_dataset(dataset: CVRPDataset, path: str | Path) -> Path:
    """Save a dataset as two normalized CSV files and return the nodes-CSV path.

    Writes ``<stem>_nodes.csv`` (one row per node) and ``<stem>_instances.csv`` (one row per
    instance), joined on the ``instance`` column. Demands are stored **normalized** as
    ``demand / capacity`` (in ``(0, 1]``), and the instances' ``capacity`` is therefore ``1``.
    This is the model-ready form and is not reversible to the original integer demands.
    Coordinates use full ``repr`` precision so they round-trip exactly.
    """
    nodes_path, instances_path = _dataset_csv_paths(path)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)

    with nodes_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "node_id", "is_depot", "x", "y", "demand"])
        for instance_idx in range(len(dataset)):
            coords = dataset.coords[instance_idx]
            demands = dataset.demands[instance_idx]
            capacity = float(dataset.capacity[instance_idx])
            for node_id in range(coords.shape[0]):
                writer.writerow(
                    [
                        instance_idx,
                        node_id,
                        int(node_id == 0),
                        repr(float(coords[node_id, 0])),
                        repr(float(coords[node_id, 1])),
                        repr(float(demands[node_id]) / capacity),
                    ]
                )

    with instances_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance",
                "n_customers",
                "capacity",
                "seed",
                "depot_mode",
                "customer_mode",
                "demand_mode",
                "route_size",
            ]
        )
        for instance_idx in range(len(dataset)):
            writer.writerow(
                [
                    instance_idx,
                    dataset.n_customers,
                    # Demands are stored normalized (demand / capacity), so capacity is 1.
                    repr(1.0),
                    dataset.seed,
                    dataset.depot_mode,
                    dataset.customer_mode,
                    dataset.demand_mode,
                    dataset.route_size,
                ]
            )

    logger.info(
        "saved dataset to %s and %s (%d instances)", nodes_path, instances_path, len(dataset)
    )
    return nodes_path


def load_dataset(path: str | Path) -> CVRPDataset:
    """Load a dataset previously written by :func:`save_dataset`."""
    nodes_path, instances_path = _dataset_csv_paths(path)

    with instances_path.open(newline="") as handle:
        instance_rows = list(csv.DictReader(handle))
    if not instance_rows:
        raise ValueError(f"no instances found in {instances_path}")

    num_instances = len(instance_rows)
    first = instance_rows[0]
    n_customers = int(first["n_customers"])
    seed = int(first["seed"])
    depot_mode = first["depot_mode"]
    customer_mode = first["customer_mode"]
    demand_mode = first["demand_mode"]
    route_size = first["route_size"]

    capacity = np.zeros(num_instances, dtype=np.float64)
    for row in instance_rows:
        capacity[int(row["instance"])] = float(row["capacity"])

    n_nodes = n_customers + 1
    coords = np.zeros((num_instances, n_nodes, 2), dtype=np.float64)
    # Demands are stored normalized (demand / capacity), so they load as floats in (0, 1].
    demands = np.zeros((num_instances, n_nodes), dtype=np.float64)
    with nodes_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            i = int(row["instance"])
            node_id = int(row["node_id"])
            coords[i, node_id, 0] = float(row["x"])
            coords[i, node_id, 1] = float(row["y"])
            demands[i, node_id] = float(row["demand"])

    return CVRPDataset(
        coords=coords,
        demands=demands,
        capacity=capacity,
        n_customers=n_customers,
        seed=seed,
        depot_mode=depot_mode,  # type: ignore[arg-type]
        customer_mode=customer_mode,  # type: ignore[arg-type]
        demand_mode=demand_mode,  # type: ignore[arg-type]
        route_size=route_size,
    )


def _derive_size_seed(base_seed: int, n_customers: int) -> int:
    """Deterministically derive an independent seed for one instance size."""
    seed_sequence = np.random.SeedSequence([base_seed, n_customers])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "data" / "cvrp.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    """Load a generation config YAML into a dict (empty if the file is missing or empty)."""
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _coerce_route_size(raw: float | int | str | None) -> float | str | None:
    """Interpret a route_size value (CLI or config) as a number, a preset name, or ``None``."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"route_size must be a number or preset name, got {raw!r}")
    if isinstance(raw, int | float):
        return float(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. All values default to ``None`` so the config supplies the fallback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="generation config YAML (source of truth; the flags below override it)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=None, help="customer counts to generate"
    )
    parser.add_argument("--num", type=int, default=None, help="instances to generate per size")
    parser.add_argument("--seed", type=int, default=None, help="base seed for all sizes")
    parser.add_argument(
        "--depot-mode",
        choices=list(get_args(DepotMode)),
        default=None,
        help="depot placement strategy",
    )
    parser.add_argument(
        "--customer-mode",
        choices=list(get_args(CustomerMode)),
        default=None,
        help="customer placement strategy",
    )
    parser.add_argument(
        "--demand-mode",
        choices=list(get_args(DemandMode)),
        default=None,
        help="demand distribution",
    )
    parser.add_argument(
        "--route-size",
        type=str,
        default=None,
        help="target avg customers-per-route (number) or preset name; derives capacity",
    )
    parser.add_argument("--capacity", type=int, default=None, help="override vehicle capacity")
    parser.add_argument(
        "--cluster-decay",
        type=float,
        default=None,
        help="exponential-attraction decay for clustered customers (normalized units)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory to write cvrp{size}_nodes.csv / cvrp{size}_instances.csv into",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: generate one dataset per size, driven by config with CLI overrides."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    config = _load_config(args.config)

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    sizes = args.sizes if args.sizes is not None else config.get("sizes", [20, 50, 100])
    num_instances = int(args.num if args.num is not None else config.get("num_instances", 10000))
    depot_mode: DepotMode = (
        args.depot_mode if args.depot_mode is not None else config.get("depot_mode", "random")
    )
    customer_mode: CustomerMode = (
        args.customer_mode
        if args.customer_mode is not None
        else config.get("customer_mode", "random")
    )
    demand_mode: DemandMode = (
        args.demand_mode if args.demand_mode is not None else config.get("demand_mode", "uniform")
    )
    capacity = args.capacity if args.capacity is not None else config.get("capacity")
    cluster_decay = float(
        args.cluster_decay
        if args.cluster_decay is not None
        else config.get("cluster_decay", NORMALIZED_CLUSTER_DECAY)
    )
    route_size = _coerce_route_size(
        args.route_size if args.route_size is not None else config.get("route_size")
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else config.get("output_dir", "data/raw/cvrp")
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    logger.info("using config_name=%s path=%s", config.get("config_name", "cvrp"), args.config)

    for raw_size in sizes:
        size = int(raw_size)
        size_seed = _derive_size_seed(seed, size)
        dataset = generate_dataset(
            size,
            num_instances,
            seed=size_seed,
            capacity=capacity,
            depot_mode=depot_mode,
            customer_mode=customer_mode,
            demand_mode=demand_mode,
            route_size=route_size,
            cluster_decay=cluster_decay,
        )
        target = output_dir / f"cvrp{size}"
        save_dataset(dataset, target)


if __name__ == "__main__":
    main()
