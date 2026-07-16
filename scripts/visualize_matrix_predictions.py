"""Visualize predicted M against ground truth M for held-out examples (task P2.3).

Trains MatrixPredictor on one split of examples and saves a ground-truth/predicted/error
comparison plot for each held-out example in the other split, per AGENTS.md Phase 2's done
criterion (qualitative predicted-vs-ground-truth heatmaps, at least 20 examples).

There is no real dataset yet -- P1.1 (CVRP generator) and P1.2 (PyVRP/OR-Tools labels) are not
implemented. So this script generates its own small synthetic CVRP instances in-memory, solved
with a simple greedy nearest-neighbor heuristic defined below. That heuristic is NOT the P1.2
OR-solver: it exists only to produce enough feasible, internally-consistent examples to exercise
this visualization pipeline at its target scale. Treat the resulting plots as a pipeline sanity
check, not a measurement of real model quality -- re-run this against real data once P1.1/P1.2
exist.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor, train_matrix_predictor
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

logger = logging.getLogger(__name__)


def configure_plot_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vrp_diffusion_quantum_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


_MIN_CUSTOMERS = 5
_MAX_CUSTOMERS = 10
_MAX_DEMAND = 5
_CAPACITY = 12.0  # > _MAX_DEMAND, so greedy construction always makes progress


def _greedy_nearest_neighbor_routes(instance: CVRPInstance) -> list[list[int]]:
    """Placeholder route construction: repeatedly visit the nearest unvisited customer that
    still fits under capacity, closing the route and starting a new one when none fit.

    Not an OR-quality solver (see module docstring) -- only used to synthesize feasible example
    data for this visualization demo.
    """
    customer_coords = instance.customer_coords()
    customer_demands = instance.customer_demands()
    depot_coords = instance.coords[instance.depot_index]

    unvisited = set(range(instance.n_customers))
    routes: list[list[int]] = []
    while unvisited:
        route: list[int] = []
        load = 0.0
        current_coords = depot_coords
        while True:
            candidates = [c for c in unvisited if load + customer_demands[c] <= instance.capacity]
            if not candidates:
                break
            next_customer = min(
                candidates,
                key=lambda c: float(np.linalg.norm(current_coords - customer_coords[c])),
            )
            route.append(next_customer)
            load += customer_demands[next_customer]
            current_coords = customer_coords[next_customer]
            unvisited.remove(next_customer)
        routes.append(route)
    return routes


def _generate_synthetic_examples(num_examples: int, seed: int) -> list[CVRPExample]:
    """Generate `num_examples` small synthetic, feasible CVRP examples (see module docstring)."""
    rng = np.random.default_rng(seed)
    examples = []
    for index in range(num_examples):
        n_customers = int(rng.integers(_MIN_CUSTOMERS, _MAX_CUSTOMERS + 1))
        coords: npt.NDArray[np.float64] = rng.uniform(0.0, 10.0, size=(n_customers + 1, 2))
        demands = np.zeros(n_customers + 1)
        demands[1:] = rng.integers(1, _MAX_DEMAND + 1, size=n_customers).astype(np.float64)

        instance = CVRPInstance(
            coords=coords,
            demands=demands,
            capacity=_CAPACITY,
            depot_index=0,
            instance_id=f"synthetic_{index:03d}",
            n_customers=n_customers,
            seed=seed,
            generator_settings={
                "kind": "synthetic_placeholder_for_P2.3",
                "note": "not P1.1/P1.2; greedy nearest-neighbor heuristic, not an OR solver",
            },
        )
        routes = _greedy_nearest_neighbor_routes(instance)
        report = validate_routes(instance, routes)
        if not report.feasible:
            raise RuntimeError(f"greedy construction produced an infeasible route: {report}")

        solution = LabeledSolution(
            routes=routes,
            cost=route_cost(instance, routes),
            num_vehicles=len(routes),
            feasible=True,
            solver_name="greedy_nearest_neighbor_placeholder",
            time_budget=None,
            seed=seed,
            runtime_seconds=0.0,
        )
        examples.append(make_example(instance, solution))
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=40)
    parser.add_argument("--held-out-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-plots", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "matrix_prediction_examples",
    )
    return parser.parse_args()


def main() -> None:
    configure_plot_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrp_diffusion_quantum.eval.visualize import plot_matrix_comparison

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    examples = _generate_synthetic_examples(args.num_examples, args.seed)
    split_index = round(args.num_examples * (1.0 - args.held_out_fraction))
    train_examples = examples[:split_index]
    held_out_examples = examples[split_index:]
    if not train_examples or not held_out_examples:
        raise ValueError(
            f"--held-out-fraction={args.held_out_fraction} leaves an empty split for "
            f"--num-examples={args.num_examples}"
        )
    if args.max_plots is not None:
        held_out_examples = held_out_examples[: args.max_plots]

    torch.manual_seed(args.seed)
    model = MatrixPredictor(hidden_dim=args.hidden_dim)
    train_matrix_predictor(
        model, train_examples, num_epochs=args.epochs, learning_rate=args.learning_rate
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for example in held_out_examples:
            m_prob = model(
                torch.from_numpy(example.instance.customer_coords()).float(),
                torch.from_numpy(example.instance.customer_demands()).float(),
                example.instance.capacity,
            ).numpy()
            instance_id = example.instance.instance_id
            n_customers = example.instance.n_customers
            fig = plot_matrix_comparison(
                example.constraint_matrix,
                m_prob,
                title=f"{instance_id} (n_customers={n_customers})",
            )
            output_path = args.output_dir / f"{instance_id}.png"
            fig.savefig(output_path)
            plt.close(fig)
            logger.info("saved comparison plot instance_id=%s", instance_id)

    print(
        f"saved {len(held_out_examples)} predicted-vs-ground-truth plot(s) to {args.output_dir} "
        f"({len(train_examples)} synthetic examples used for training, "
        f"{len(held_out_examples)} held out)"
    )


if __name__ == "__main__":
    main()
