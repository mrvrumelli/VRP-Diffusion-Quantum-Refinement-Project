"""CVRP example JSON I/O, collate, and size-homogeneous batching."""

from __future__ import annotations

import base64
import json
import re
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch

from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix

_METADATA_FILES = {"subset_manifest.json", "training_label_manifest.json"}


class IndexedJSONDataset(Sequence[CVRPExample]):
    """Path-indexed JSON dataset with a bounded in-process LRU cache.

    Construction indexes filenames only. Examples are parsed on first access, so callers can
    inspect, select, or batch a large corpus without retaining every matrix in memory.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        sizes: Sequence[int] | None = None,
        cache_size: int = 0,
    ) -> None:
        if cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}")
        root = Path(dataset_dir)
        patterns = ["*.json"] if sizes is None else [f"cvrp{int(size)}_*.json" for size in sizes]
        paths = {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.name not in _METADATA_FILES
        }
        self.root = root
        self.paths = tuple(sorted(paths))
        self.cache_size = cache_size
        self._cache: OrderedDict[int, CVRPExample] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    @overload
    def __getitem__(self, index: int) -> CVRPExample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[CVRPExample]: ...

    def __getitem__(self, index: int | slice) -> CVRPExample | Sequence[CVRPExample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = index if index >= 0 else len(self) + index
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        if normalized in self._cache:
            example = self._cache.pop(normalized)
            self._cache[normalized] = example
            return example
        example = load_example(self.paths[normalized])
        if self.cache_size:
            self._cache[normalized] = example
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return example

    def clear_cache(self) -> None:
        self._cache.clear()

    def indices_by_size(self) -> dict[int, list[int]]:
        """Return indexed positions grouped by CVRP size, parsing conventional filenames."""
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, path in enumerate(self.paths):
            match = re.match(r"cvrp(\d+)_", path.name)
            size = int(match.group(1)) if match else self[index].instance.n_customers
            grouped[size].append(index)
        return dict(grouped)


def make_example(instance: CVRPInstance, solution: LabeledSolution) -> CVRPExample:
    """Build a `CVRPExample`, deriving `constraint_matrix` from `solution.routes` (task P1.3)."""
    constraint_matrix = build_constraint_matrix(solution.routes, instance.n_customers)
    return CVRPExample(instance=instance, solution=solution, constraint_matrix=constraint_matrix)


def save_example(
    example: CVRPExample,
    path: str | Path,
    *,
    compact_matrix: bool = True,
) -> None:
    """Save one `CVRPExample` as a single JSON file, creating parent directories as needed."""
    payload = {
        "instance": {
            "coords": example.instance.coords.tolist(),
            "demands": example.instance.demands.tolist(),
            "capacity": example.instance.capacity,
            "depot_index": example.instance.depot_index,
            "instance_id": example.instance.instance_id,
            "n_customers": example.instance.n_customers,
            "seed": example.instance.seed,
            "generator_settings": example.instance.generator_settings,
        },
        "solution": {
            "routes": example.solution.routes,
            "cost": example.solution.cost,
            "num_vehicles": example.solution.num_vehicles,
            "feasible": example.solution.feasible,
            "solver_name": example.solution.solver_name,
            "time_budget": example.solution.time_budget,
            "seed": example.solution.seed,
            "runtime_seconds": example.solution.runtime_seconds,
        },
    }
    if compact_matrix:
        packed = np.packbits(example.constraint_matrix.reshape(-1), bitorder="little")
        payload["constraint_matrix_packed"] = {
            "encoding": "numpy-packbits-little-base64-v1",
            "shape": list(example.constraint_matrix.shape),
            "data": base64.b64encode(packed.tobytes()).decode("ascii"),
        }
    else:
        payload["constraint_matrix"] = example.constraint_matrix.tolist()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def load_example(path: str | Path) -> CVRPExample:
    """Load one `CVRPExample` previously written by `save_example`."""
    payload = json.loads(Path(path).read_text())

    instance_payload = payload["instance"]
    instance = CVRPInstance(
        coords=np.array(instance_payload["coords"], dtype=np.float64),
        demands=np.array(instance_payload["demands"], dtype=np.float64),
        capacity=float(instance_payload["capacity"]),
        depot_index=int(instance_payload["depot_index"]),
        instance_id=str(instance_payload["instance_id"]),
        n_customers=int(instance_payload["n_customers"]),
        seed=instance_payload["seed"],
        generator_settings=instance_payload["generator_settings"],
    )

    solution_payload = payload["solution"]
    solution = LabeledSolution(
        routes=[list(route) for route in solution_payload["routes"]],
        cost=float(solution_payload["cost"]),
        num_vehicles=int(solution_payload["num_vehicles"]),
        feasible=bool(solution_payload["feasible"]),
        solver_name=str(solution_payload["solver_name"]),
        time_budget=solution_payload["time_budget"],
        seed=solution_payload["seed"],
        runtime_seconds=float(solution_payload["runtime_seconds"]),
    )

    if "constraint_matrix_packed" in payload:
        matrix_payload = payload["constraint_matrix_packed"]
        if matrix_payload.get("encoding") != "numpy-packbits-little-base64-v1":
            raise ValueError(f"unsupported matrix encoding: {matrix_payload.get('encoding')}")
        shape = tuple(int(value) for value in matrix_payload["shape"])
        if len(shape) != 2:
            raise ValueError("packed constraint matrix shape must have two dimensions")
        raw = base64.b64decode(matrix_payload["data"], validate=True)
        bit_count = shape[0] * shape[1]
        unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        if unpacked.size < bit_count or len(raw) != (bit_count + 7) // 8:
            raise ValueError("packed constraint matrix byte length does not match shape")
        constraint_matrix = unpacked[:bit_count].reshape(shape).astype(np.uint8)
    else:
        # Preserve the serialized dtype through validation so malformed floats are rejected.
        constraint_matrix = np.array(payload["constraint_matrix"])
    return CVRPExample(instance=instance, solution=solution, constraint_matrix=constraint_matrix)


def load_dataset(dataset_dir: str | Path) -> list[CVRPExample]:
    """Load example JSON files, excluding dataset metadata, in deterministic order."""
    return list(IndexedJSONDataset(dataset_dir))


def load_examples_by_size(dataset_dir: str | Path, sizes: list[int]) -> list[CVRPExample]:
    """Load only ``cvrp{size}_*.json`` files for the given customer sizes."""
    return list(IndexedJSONDataset(dataset_dir, sizes=sizes))


@dataclass
class CVRPBatch:
    """A batch of `CVRPExample`s, zero-padded to the batch's max size, plus padding masks."""

    coords: torch.Tensor  # [batch, max_n_nodes, 2] float32
    demands: torch.Tensor  # [batch, max_n_nodes] float32
    node_mask: torch.Tensor  # [batch, max_n_nodes] bool, True where the node is real
    depot_index: torch.Tensor  # [batch] int64
    customer_node_indices: torch.Tensor  # [batch, max_n_customers] int64, -1 where padded
    capacity: torch.Tensor  # [batch] float32
    constraint_matrix: torch.Tensor  # [batch, max_n_customers, max_n_customers] float32
    customer_mask: torch.Tensor  # [batch, max_n_customers] bool, True where the customer is real
    cost: torch.Tensor  # [batch] float32
    num_vehicles: torch.Tensor  # [batch] int64
    feasible: torch.Tensor  # [batch] bool
    metadata: list[dict[str, Any]]  # per-example instance_id, seed, solver_name, routes, ...


def collate_batch(examples: list[CVRPExample]) -> CVRPBatch:
    """Pad and stack a list of `CVRPExample` into a `CVRPBatch`.

    Examples may have different `n_customers` (e.g. mixed CVRP20/CVRP50 in one batch); shorter
    examples are zero-padded up to the batch max, with `node_mask`/`customer_mask` marking real
    (non-padded) entries.
    """
    if not examples:
        raise ValueError("cannot collate an empty list of examples")

    max_n_nodes = max(example.instance.coords.shape[0] for example in examples)
    max_n_customers = max(example.instance.n_customers for example in examples)
    batch_size = len(examples)

    coords = torch.zeros((batch_size, max_n_nodes, 2), dtype=torch.float32)
    demands = torch.zeros((batch_size, max_n_nodes), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_n_nodes), dtype=torch.bool)
    depot_index = torch.zeros((batch_size,), dtype=torch.int64)
    customer_node_indices = torch.full((batch_size, max_n_customers), -1, dtype=torch.int64)
    capacity = torch.zeros((batch_size,), dtype=torch.float32)
    constraint_matrix = torch.zeros(
        (batch_size, max_n_customers, max_n_customers), dtype=torch.float32
    )
    customer_mask = torch.zeros((batch_size, max_n_customers), dtype=torch.bool)
    cost = torch.zeros((batch_size,), dtype=torch.float32)
    num_vehicles = torch.zeros((batch_size,), dtype=torch.int64)
    feasible = torch.zeros((batch_size,), dtype=torch.bool)
    metadata: list[dict[str, Any]] = []

    for i, example in enumerate(examples):
        n_nodes = example.instance.coords.shape[0]
        n_customers = example.instance.n_customers

        coords[i, :n_nodes] = torch.from_numpy(example.instance.coords).to(torch.float32)
        demands[i, :n_nodes] = torch.from_numpy(example.instance.demands).to(torch.float32)
        node_mask[i, :n_nodes] = True
        depot_index[i] = example.instance.depot_index
        customer_nodes = example.instance.customer_node_indices()
        customer_node_indices[i, :n_customers] = torch.tensor(customer_nodes, dtype=torch.int64)
        capacity[i] = example.instance.capacity
        constraint_matrix[i, :n_customers, :n_customers] = torch.from_numpy(
            example.constraint_matrix
        ).to(torch.float32)
        customer_mask[i, :n_customers] = True
        cost[i] = example.solution.cost
        num_vehicles[i] = example.solution.num_vehicles
        feasible[i] = example.solution.feasible
        metadata.append(
            {
                "instance_id": example.instance.instance_id,
                "n_customers": n_customers,
                "depot_index": example.instance.depot_index,
                "customer_node_indices": customer_nodes,
                "seed": example.instance.seed,
                "generator_settings": example.instance.generator_settings,
                "solver_name": example.solution.solver_name,
                "routes": example.solution.routes,
            }
        )

    return CVRPBatch(
        coords=coords,
        demands=demands,
        node_mask=node_mask,
        depot_index=depot_index,
        customer_node_indices=customer_node_indices,
        capacity=capacity,
        constraint_matrix=constraint_matrix,
        customer_mask=customer_mask,
        cost=cost,
        num_vehicles=num_vehicles,
        feasible=feasible,
        metadata=metadata,
    )


def size_homogeneous_chunks(
    examples: list[CVRPExample],
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    shuffle: bool = True,
    augmentation: bool = False,
) -> Iterator[list[CVRPExample]]:
    """Group by ``n_customers``; optionally expand to nine geometric views."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not examples:
        return

    from vrp_diffusion_quantum.data.augment import AUGMENT_NUM, augment_example

    by_size: dict[int, list[CVRPExample]] = defaultdict(list)
    for example in examples:
        by_size[int(example.instance.n_customers)].append(example)

    for n_customers in sorted(by_size):
        bucket = by_size[n_customers]
        if augmentation:
            factor = AUGMENT_NUM
            pair_count = len(bucket) * factor
            if shuffle:
                gen = generator if generator is not None else torch.Generator().manual_seed(0)
                size_gen = torch.Generator().manual_seed(
                    int(gen.initial_seed()) + 1_000_003 * int(n_customers)
                )
                order: list[int] = torch.randperm(pair_count, generator=size_gen).tolist()
            else:
                order = list(range(pair_count))
            for start in range(0, pair_count, batch_size):
                chunk: list[CVRPExample] = []
                for flat in order[start : start + batch_size]:
                    ex_i = int(flat) // factor
                    variant = int(flat) % factor
                    chunk.append(augment_example(bucket[ex_i], variant))
                yield chunk
            continue

        if shuffle:
            gen = generator if generator is not None else torch.Generator().manual_seed(0)
            size_gen = torch.Generator().manual_seed(
                int(gen.initial_seed()) + 1_000_003 * int(n_customers)
            )
            order_tensor = torch.randperm(len(bucket), generator=size_gen)
            bucket = [bucket[int(i)] for i in order_tensor]
        for start in range(0, len(bucket), batch_size):
            yield bucket[start : start + batch_size]


def size_homogeneous_batches(
    examples: list[CVRPExample],
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    shuffle: bool = True,
    augmentation: bool = False,
) -> Iterator[CVRPBatch]:
    """Yield collated batches where every example shares the same ``n_customers``."""
    for chunk in size_homogeneous_chunks(
        examples,
        batch_size,
        generator=generator,
        shuffle=shuffle,
        augmentation=augmentation,
    ):
        yield collate_batch(chunk)
