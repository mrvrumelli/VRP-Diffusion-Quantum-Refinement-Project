"""Dataset save/load format and batching for CVRP examples (task P1.4).

Each `CVRPExample` (coords, demands, capacity, routes, cost, constraint matrix, metadata) is
saved as one JSON file. `load_dataset` reads every example file in a directory, and
`collate_batch` pads and stacks a list of examples into batched tensors plus a padding mask,
for instances of varying `n_customers` in the same batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix


def make_example(instance: CVRPInstance, solution: LabeledSolution) -> CVRPExample:
    """Build a `CVRPExample`, deriving `constraint_matrix` from `solution.routes` (task P1.3)."""
    constraint_matrix = build_constraint_matrix(solution.routes, instance.n_customers)
    return CVRPExample(instance=instance, solution=solution, constraint_matrix=constraint_matrix)


def save_example(example: CVRPExample, path: str | Path) -> None:
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
        "constraint_matrix": example.constraint_matrix.tolist(),
    }
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

    constraint_matrix = np.array(payload["constraint_matrix"])
    return CVRPExample(instance=instance, solution=solution, constraint_matrix=constraint_matrix)


def load_dataset(dataset_dir: str | Path) -> list[CVRPExample]:
    """Load every `*.json` example file in `dataset_dir`, sorted by filename for determinism."""
    example_paths = sorted(Path(dataset_dir).glob("*.json"))
    return [load_example(example_path) for example_path in example_paths]


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
