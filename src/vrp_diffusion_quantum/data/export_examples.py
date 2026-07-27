"""Bridge generated datasets + labels into the modeling ``CVRPExample`` JSON format.

The generation/solving pipeline (:mod:`vrp_diffusion_quantum.data.generate_cvrp`,
:mod:`vrp_diffusion_quantum.data.solve_cvrp`) writes ``cvrp{N}_nodes.csv`` /
``cvrp{N}_instances.csv`` pairs (raw integer demands and the real vehicle capacity) plus
``labels/cvrp{N}_labels.json``. The modeling side (:mod:`vrp_diffusion_quantum.data.dataset`,
:mod:`vrp_diffusion_quantum.data.types`) instead consumes one ``CVRPExample`` JSON *per
instance*, loadable with :func:`vrp_diffusion_quantum.data.dataset.load_dataset`.

This module converts a run folder into those per-example JSON files. One subtlety is handled
here:

* **Route index base.** Solver routes store *node* indices (depot excluded, depot is node 0),
  while ``CVRPExample`` / ``build_constraint_matrix`` expect 0-based *customer ids* in
  ``[0, n_customers)``. We remap ``node -> customer id`` before building the example.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.generate_cvrp import CVRPDataset, load_dataset
from vrp_diffusion_quantum.data.generate_cvrp import _load_config as _load_config_file
from vrp_diffusion_quantum.data.types import CVRPExample, LabeledSolution
from vrp_diffusion_quantum.data.types import CVRPInstance as ExampleCVRPInstance
from vrp_diffusion_quantum.utils.feasibility import validate_labeled_solution

__all__ = [
    "build_examples",
    "export_run",
    "export_size",
    "routes_nodes_to_customer_ids",
]


def routes_nodes_to_customer_ids(
    routes: list[list[int]], depot_index: int, n_customers: int
) -> list[list[int]]:
    """Remap solver routes from node indices to 0-based customer ids.

    Solver routes hold node array indices (with the depot omitted). ``CVRPExample`` and
    :func:`~vrp_diffusion_quantum.utils.constraint_matrix.build_constraint_matrix` address
    customers by a 0-based id in ``[0, n_customers)``. With the depot at ``depot_index`` the
    non-depot nodes, in ascending order, map to customer ids ``0..n_customers - 1``.
    """
    customer_nodes = [i for i in range(n_customers + 1) if i != depot_index]
    node_to_customer = {node: customer_id for customer_id, node in enumerate(customer_nodes)}
    remapped: list[list[int]] = []
    for route in routes:
        remapped.append([node_to_customer[int(node)] for node in route])
    return remapped


def build_examples(
    dataset: CVRPDataset,
    solutions: list[dict[str, Any]],
    *,
    run_name: str,
    generator_settings: dict[str, Any],
) -> list[CVRPExample]:
    """Build one :class:`CVRPExample` per solved instance.

    Args:
        dataset: the instances loaded from the run's CSVs (raw integer demands, real capacity).
        solutions: label dicts as written by ``save_labels`` (``asdict`` of ``CVRPSolution``).
        run_name: used to make a stable, human-readable ``instance_id``.
        generator_settings: metadata dict stored on every ``CVRPInstance``.
    """
    if len(solutions) != len(dataset):
        raise ValueError(
            f"solutions ({len(solutions)}) and dataset ({len(dataset)}) length mismatch"
        )

    n_customers = dataset.n_customers
    examples: list[CVRPExample] = []
    for instance_idx, label in enumerate(solutions):
        label_instance_id = int(label["instance_id"])
        if label_instance_id != instance_idx:
            raise ValueError(
                f"label at position {instance_idx} has instance_id={label_instance_id}; "
                "labels must be ordered and aligned with dataset instances"
            )
        label_n_customers = int(label["n_customers"])
        if label_n_customers != n_customers:
            raise ValueError(
                f"label instance_id={label_instance_id} has n_customers={label_n_customers}, "
                f"expected {n_customers}"
            )

        coords = np.asarray(dataset.coords[instance_idx], dtype=np.float64)
        demands = np.asarray(dataset.demands[instance_idx], dtype=np.float64)
        capacity = float(dataset.capacity[instance_idx])
        depot_index = 0  # generated instances always store the depot as node 0

        instance = ExampleCVRPInstance(
            coords=coords,
            demands=demands,
            capacity=capacity,
            depot_index=depot_index,
            instance_id=f"{run_name}_cvrp{n_customers}_{instance_idx:04d}",
            n_customers=n_customers,
            seed=label.get("seed"),
            generator_settings=generator_settings,
        )
        routes = routes_nodes_to_customer_ids(
            [list(route) for route in label["routes"]], depot_index, n_customers
        )
        solution = LabeledSolution(
            routes=routes,
            cost=float(label["cost"]),
            num_vehicles=int(label["num_vehicles"]),
            feasible=bool(label["feasible"]),
            solver_name=str(label["solver_name"]),
            time_budget=label.get("time_budget"),
            seed=label.get("seed"),
            runtime_seconds=float(label.get("runtime_seconds", 0.0)),
        )
        if not solution.feasible:
            raise ValueError(f"label instance_id={label_instance_id} is marked infeasible")
        feasibility = validate_labeled_solution(instance, solution)
        if not feasibility.feasible:
            details = "; ".join(feasibility.violations)
            raise ValueError(f"invalid label instance_id={label_instance_id}: {details}")

        examples.append(make_example(instance, solution))
    return examples


def export_size(
    run_dir: Path,
    size: int,
    out_dir: Path,
    *,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Write one ``CVRPExample`` JSON per instance for a single size; return the paths written."""
    config = config if config is not None else _load_config(run_dir)
    labels_path = run_dir / "labels" / f"cvrp{size}_labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"no labels for cvrp{size}: {labels_path}")
    solutions = json.loads(labels_path.read_text())

    dataset = load_dataset(run_dir / f"cvrp{size}")
    run_name = str(config.get("run_name") or run_dir.name)
    generator_settings = {
        "run_name": run_name,
        "n_customers": size,
        "depot_mode": config.get("depot_mode"),
        "customer_mode": config.get("customer_mode"),
        "demand_mode": config.get("demand_mode"),
        "source": "export_examples",
    }
    examples = build_examples(
        dataset,
        solutions,
        run_name=run_name,
        generator_settings=generator_settings,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for instance_idx, example in enumerate(examples):
        path = out_dir / f"cvrp{size}_{instance_idx:04d}.json"
        save_example(example, path)
        written.append(path)
    return written


def export_run(
    run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    sizes: list[int] | None = None,
) -> dict[int, list[Path]]:
    """Export every labeled size in ``run_dir`` to ``CVRPExample`` JSON files.

    Args:
        run_dir: a generated run folder containing ``cvrp{N}_*.csv`` and ``labels/``.
        out_dir: where the example JSONs go; defaults to ``run_dir/examples``.
        sizes: restrict to these sizes; defaults to every size that has a labels file.

    Returns:
        Mapping of size to the list of example paths written for it.
    """
    run_path = Path(run_dir)
    output_path = Path(out_dir) if out_dir is not None else run_path / "examples"
    config = _load_config(run_path)

    if sizes is None:
        sizes = _discover_labeled_sizes(run_path)
    if not sizes:
        raise ValueError(f"no labeled sizes found under {run_path / 'labels'}")

    written: dict[int, list[Path]] = {}
    for size in sorted(sizes):
        written[size] = export_size(run_path, size, output_path, config=config)
    return written


def _load_config(run_dir: Path) -> dict[str, Any]:
    """Load a run's ``config.yaml`` snapshot (reuses the generator's parser)."""
    return _load_config_file(run_dir / "config.yaml")


def _discover_labeled_sizes(run_dir: Path) -> list[int]:
    labels_dir = run_dir / "labels"
    if not labels_dir.is_dir():
        return []
    sizes: list[int] = []
    for path in labels_dir.glob("cvrp*_labels.json"):
        stem = path.name[len("cvrp") : -len("_labels.json")]
        if stem.isdigit():
            sizes.append(int(stem))
    return sorted(sizes)
