"""Unattended generate -> OR-label -> export pipeline, safe to leave running overnight.

Runs the same three stages as the Streamlit dataset generator
(``app_cvrp_dataset_generator.py``: Generate / Solve / Convert), driven end to end from
``configs/data/cvrp.yaml`` and ``configs/data/solve_labels.yaml``, but headless and resumable.

Solving is the slow part (minutes to hours per size), so progress is checkpointed after every
solved instance. Killing the process (or a crash, or the machine sleeping) loses at most the
instance in flight — rerun the exact same command and it picks up where it left off instead of
re-solving from scratch. A run is identified by its output folder
(``<data-root>/<run-name>/``); reuse ``--run-name`` to resume a specific run.

Examples:
    # Use the defaults in configs/data/cvrp.yaml + configs/data/solve_labels.yaml.
    python scripts/run_data_pipeline.py

    # Bigger run, 4 solver processes in parallel, resumable under a fixed name.
    python scripts/run_data_pipeline.py --run-name big_run --num 20000 --workers 4

    # Left running overnight and interrupted -- just run the same command again:
    python scripts/run_data_pipeline.py --run-name big_run --num 20000 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

import numpy as np
import yaml

from vrp_diffusion_quantum.data import generate_cvrp as gen_mod
from vrp_diffusion_quantum.data.export_examples import export_run
from vrp_diffusion_quantum.data.generate_cvrp import CVRPDataset, CVRPInstance, DemandMode
from vrp_diffusion_quantum.data.solve_cvrp import FleetMode, SolverName, solve_instance
from vrp_diffusion_quantum.utils.experiment import hash_dataset

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEN_CONFIG = ROOT / "configs" / "data" / "cvrp.yaml"
DEFAULT_SOLVE_CONFIG = ROOT / "configs" / "data" / "solve_labels.yaml"
LOG_EVERY = 25


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gen-config", type=Path, default=DEFAULT_GEN_CONFIG)
    parser.add_argument("--solve-config", type=Path, default=DEFAULT_SOLVE_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="workspace root; default: gen config output_dir",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="run folder name; pass the same name again to resume that run",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--num", type=int, default=None, help="instances to generate per size")
    parser.add_argument("--seed", type=int, default=None, help="base seed for generation")
    parser.add_argument("--depot-mode", choices=["random", "center", "corner"], default=None)
    parser.add_argument(
        "--customer-mode", choices=["random", "clustered", "random_clustered"], default=None
    )
    parser.add_argument("--demand-mode", choices=list(get_args(DemandMode)), default=None)
    parser.add_argument("--route-size", type=str, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--cluster-decay", type=float, default=None)

    parser.add_argument("--solver", choices=["pyvrp", "ortools"], default=None)
    parser.add_argument("--solve-seed", type=int, default=None, help="base seed for OR labeling")
    parser.add_argument(
        "--time-limit", type=float, default=None, help="solver wall-clock seconds per instance"
    )
    parser.add_argument(
        "--no-improvement-seconds",
        type=float,
        default=None,
        help="stop an instance early once its best cost stalls this long",
    )
    parser.add_argument("--fleet-mode", choices=["unlimited", "up_to", "exact"], default=None)
    parser.add_argument("--fleet-size", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="solve this many instances in parallel per size (PyVRP/OR-Tools release the GIL "
        "during solve, so this is real speedup; start with a small run before trusting a big one)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="stop after labeling; skip building CVRPExample JSON files",
    )
    return parser.parse_args(argv)


def _requested_solver_settings(args: argparse.Namespace, generation_seed: int) -> dict[str, Any]:
    """Resolve and validate the complete solver configuration for a new run."""
    solve_config = gen_mod._load_config(args.solve_config)
    solver_defaults = solve_config.get("solver", {})

    raw_time_limit = (
        args.time_limit if args.time_limit is not None else solver_defaults.get("time_limit", 1.0)
    )
    time_limit = None if raw_time_limit is None else float(raw_time_limit)
    raw_no_improvement = (
        args.no_improvement_seconds
        if args.no_improvement_seconds is not None
        else solver_defaults.get("no_improvement_seconds")
    )
    no_improvement_seconds = None if raw_no_improvement is None else float(raw_no_improvement)
    fleet_mode: FleetMode = args.fleet_mode or solver_defaults.get("fleet_mode", "unlimited")
    fleet_size = (
        args.fleet_size if args.fleet_size is not None else solver_defaults.get("fleet_size")
    )
    if fleet_mode in ("up_to", "exact") and fleet_size is None:
        raise ValueError(f"--fleet-size is required when --fleet-mode is {fleet_mode!r}")
    if time_limit is None and no_improvement_seconds is None:
        raise ValueError("solver config must set time_limit and/or no_improvement_seconds")

    return {
        "name": args.solver or solver_defaults.get("name", "pyvrp"),
        "seed": int(
            args.solve_seed
            if args.solve_seed is not None
            else solve_config.get("seed", generation_seed)
        ),
        "time_limit": time_limit,
        "no_improvement_seconds": no_improvement_seconds,
        "fleet_mode": fleet_mode,
        "fleet_size": None if fleet_size is None else int(fleet_size),
        "workers": max(1, int(args.workers)),
    }


def _resolve_run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Resolve the run folder and its effective generation config.

    If ``<run_dir>/config.yaml`` already exists, it is authoritative (a resumed run): CLI /
    gen-config overrides are ignored so resuming can't silently change generation parameters
    partway through a dataset. Otherwise a fresh effective config is built and snapshotted.
    """
    gen_config = gen_mod._load_config(args.gen_config)

    data_root = args.data_root
    if data_root is None:
        data_root = ROOT / gen_config.get("output_dir", "data/raw/cvrp")
    elif not data_root.is_absolute():
        data_root = ROOT / data_root

    config_name = str(gen_config.get("config_name", "cvrp"))
    run_name = args.run_name or f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = data_root / run_name
    snapshot_path = run_dir / "config.yaml"

    if snapshot_path.is_file():
        logger.info("run_dir=%s has an existing config.yaml; resuming with its settings", run_dir)
        existing = gen_mod._load_config(snapshot_path)
        if "solver" not in existing:
            labels_dir = run_dir / "labels"
            has_labels = labels_dir.is_dir() and any(labels_dir.iterdir())
            if has_labels:
                raise ValueError(
                    f"cannot safely resume legacy run {run_dir}: config.yaml has no solver "
                    "settings but label checkpoints already exist"
                )
            existing["solver"] = _requested_solver_settings(args, int(existing["seed"]))
            snapshot_path.write_text(yaml.safe_dump(existing, sort_keys=False))
        return run_dir, existing

    seed = int(args.seed if args.seed is not None else gen_config.get("seed", 42))
    sizes = [
        int(s)
        for s in (args.sizes if args.sizes is not None else gen_config.get("sizes", [20, 50, 100]))
    ]
    effective = {
        "config_name": config_name,
        "seed": seed,
        "sizes": sizes,
        "num_instances": int(
            args.num if args.num is not None else gen_config.get("num_instances", 10000)
        ),
        "capacity": args.capacity if args.capacity is not None else gen_config.get("capacity"),
        "depot_mode": args.depot_mode or gen_config.get("depot_mode", "random"),
        "customer_mode": args.customer_mode or gen_config.get("customer_mode", "random"),
        "demand_mode": args.demand_mode or gen_config.get("demand_mode", "uniform"),
        "route_size": gen_mod._coerce_route_size(
            args.route_size if args.route_size is not None else gen_config.get("route_size")
        ),
        "cluster_decay": float(
            args.cluster_decay
            if args.cluster_decay is not None
            else gen_config.get("cluster_decay", gen_mod.NORMALIZED_CLUSTER_DECAY)
        ),
        "run_name": run_name,
        "output_dir": str(data_root.relative_to(ROOT))
        if data_root.is_relative_to(ROOT)
        else str(data_root),
        "solver": _requested_solver_settings(args, seed),
    }
    return run_dir, effective


def _generate_size(run_dir: Path, size: int, config: dict[str, Any]) -> CVRPDataset:
    target = run_dir / f"cvrp{size}"
    nodes_path, instances_path = gen_mod._dataset_csv_paths(target)
    if nodes_path.is_file() and instances_path.is_file():
        logger.info("cvrp%d: dataset already generated, loading from disk", size)
        return gen_mod.load_dataset(target)

    size_seed = gen_mod._derive_size_seed(int(config["seed"]), size)
    logger.info(
        "cvrp%d: generating %d instances (seed=%d)", size, config["num_instances"], size_seed
    )
    dataset = gen_mod.generate_dataset(
        size,
        int(config["num_instances"]),
        seed=size_seed,
        capacity=config.get("capacity"),
        depot_mode=config.get("depot_mode", "random"),
        customer_mode=config.get("customer_mode", "random"),
        demand_mode=config.get("demand_mode", "uniform"),
        route_size=config.get("route_size"),
        cluster_decay=float(config.get("cluster_decay", gen_mod.NORMALIZED_CLUSTER_DECAY)),
    )
    gen_mod.save_dataset(dataset, target)
    return dataset


def _labels_path(labels_dir: Path, size: int) -> Path:
    return labels_dir / f"cvrp{size}_labels.json"


def _partial_path(labels_dir: Path, size: int) -> Path:
    return labels_dir / f".cvrp{size}_labels.partial.jsonl"


def _load_partial(path: Path) -> dict[int, dict[str, Any]]:
    """Load already-solved instances from a checkpoint file, keyed by instance_id.

    A trailing line truncated by a crash mid-write is silently dropped; that one instance is
    simply re-solved.
    """
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[int(record["instance_id"])] = record
    return records


def _solve_one(
    instance: CVRPInstance,
    *,
    instance_id: int,
    base_seed: int,
    solver: SolverName,
    time_limit: float | None,
    no_improvement_seconds: float | None,
    fleet_mode: FleetMode,
    fleet_size: int | None,
) -> dict[str, Any]:
    instance_seed = int(np.random.SeedSequence([base_seed, instance_id]).generate_state(1)[0])
    solution = solve_instance(
        instance,
        solver=solver,
        time_limit=time_limit,
        no_improvement_seconds=no_improvement_seconds,
        fleet_mode=fleet_mode,
        fleet_size=fleet_size,
        seed=instance_seed,
        instance_id=instance_id,
    )
    return asdict(solution)


def _solve_size(
    dataset: CVRPDataset,
    size: int,
    labels_dir: Path,
    *,
    solver: SolverName,
    seed: int,
    time_limit: float | None,
    no_improvement_seconds: float | None,
    fleet_mode: FleetMode,
    fleet_size: int | None,
    workers: int,
) -> dict[str, Any]:
    final_path = _labels_path(labels_dir, size)
    if final_path.is_file():
        logger.info("cvrp%d: already labeled, skipping solve", size)
        payload = json.loads(final_path.read_text())
        return _summarize(size, payload, dataset)

    labels_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_path(labels_dir, size)
    done = _load_partial(partial_path)
    remaining = [i for i in range(len(dataset)) if i not in done]
    total = len(dataset)
    if done:
        logger.info("cvrp%d: resuming, %d/%d already solved", size, len(done), total)

    def _kwargs(instance_id: int) -> dict[str, Any]:
        return dict(
            instance=dataset[instance_id],
            instance_id=instance_id,
            base_seed=seed,
            solver=solver,
            time_limit=time_limit,
            no_improvement_seconds=no_improvement_seconds,
            fleet_mode=fleet_mode,
            fleet_size=fleet_size,
        )

    if remaining:
        with partial_path.open("a") as handle:

            def _checkpoint(payload: dict[str, Any]) -> None:
                done[int(payload["instance_id"])] = payload
                handle.write(json.dumps(payload) + "\n")
                handle.flush()

            completed = 0
            start = time.perf_counter()
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_solve_one, **_kwargs(i)): i for i in remaining}
                    for future in as_completed(futures):
                        _checkpoint(future.result())
                        completed += 1
                        if completed % LOG_EVERY == 0 or completed == len(remaining):
                            elapsed = time.perf_counter() - start
                            logger.info(
                                "cvrp%d: %d/%d solved this run (%.1fs elapsed, %.2fs/instance avg)",
                                size,
                                completed,
                                len(remaining),
                                elapsed,
                                elapsed / completed,
                            )
            else:
                for i in remaining:
                    _checkpoint(_solve_one(**_kwargs(i)))
                    completed += 1
                    if completed % LOG_EVERY == 0 or completed == len(remaining):
                        elapsed = time.perf_counter() - start
                        logger.info(
                            "cvrp%d: %d/%d solved this run (%.1fs elapsed, %.2fs/instance avg)",
                            size,
                            completed,
                            len(remaining),
                            elapsed,
                            elapsed / completed,
                        )

    ordered = [done[i] for i in range(total)]
    final_path.write_text(json.dumps(ordered, indent=2) + "\n")
    partial_path.unlink(missing_ok=True)
    logger.info("cvrp%d: wrote %s", size, final_path)
    return _summarize(size, ordered, dataset)


def _summarize(
    size: int,
    solutions: list[dict[str, Any]],
    dataset: CVRPDataset,
) -> dict[str, Any]:
    costs = [s["cost"] for s in solutions]
    vehicles = [s["num_vehicles"] for s in solutions]
    runtimes = [s["runtime_seconds"] for s in solutions]
    feasible = sum(1 for s in solutions if s["feasible"])
    customer_demands = dataset.demands[:, 1:].astype(np.float64)
    total_demands = customer_demands.sum(axis=1)
    return {
        "size": size,
        "n": len(solutions),
        "feasible": feasible,
        "feasible_rate": round(feasible / len(solutions), 4) if solutions else None,
        "mean_cost": round(float(np.mean(costs)), 4) if costs else None,
        "mean_vehicles": round(float(np.mean(vehicles)), 3) if vehicles else None,
        "mean_runtime_seconds": round(float(np.mean(runtimes)), 4) if runtimes else None,
        "median_runtime_seconds": round(float(np.median(runtimes)), 4) if runtimes else None,
        "p95_runtime_seconds": round(float(np.quantile(runtimes, 0.95)), 4) if runtimes else None,
        "total_runtime_seconds": round(float(np.sum(runtimes)), 4) if runtimes else None,
        "mean_customer_demand": round(float(np.mean(customer_demands)), 4),
        "min_customer_demand": round(float(np.min(customer_demands)), 4),
        "max_customer_demand": round(float(np.max(customer_demands)), 4),
        "mean_total_demand": round(float(np.mean(total_demands)), 4),
        "mean_capacity": round(float(np.mean(dataset.capacity)), 4),
        "min_capacity": round(float(np.min(dataset.capacity)), 4),
        "max_capacity": round(float(np.max(dataset.capacity)), 4),
    }


def _write_dataset_hashes(run_dir: Path, sizes: list[int]) -> str:
    """Write component and aggregate hashes for the generated modeling dataset."""
    component_hashes: dict[str, str] = {}
    for size in sizes:
        for suffix in ("nodes", "instances"):
            path = run_dir / f"cvrp{size}_{suffix}.csv"
            component_hashes[f"cvrp{size}_{suffix}"] = hash_dataset(path)
        labels_path = run_dir / "labels" / f"cvrp{size}_labels.json"
        component_hashes[f"cvrp{size}_labels"] = hash_dataset(labels_path)

    examples_dir = run_dir / "examples"
    if examples_dir.is_dir():
        component_hashes["examples"] = hash_dataset(examples_dir)

    serialized = json.dumps(component_hashes, sort_keys=True).encode()
    aggregate_hash = hashlib.sha256(serialized).hexdigest()
    (run_dir / "dataset_hashes.json").write_text(
        json.dumps(component_hashes, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "dataset_hash.txt").write_text(aggregate_hash + "\n")
    return aggregate_hash


def _write_summary_csv(run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    """Write the Phase 1 baseline statistics as a machine-readable table."""
    if not summary_rows:
        return
    with (run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dir, config = _resolve_run(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "run.log")

    snapshot_path = run_dir / "config.yaml"
    if not snapshot_path.is_file():
        snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False))
    commit_hash = _git_commit_hash()
    if commit_hash:
        (run_dir / "commit_hash.txt").write_text(commit_hash + "\n")

    sizes = [int(s) for s in config["sizes"]]
    solver_config = config["solver"]
    solver: SolverName = solver_config["name"]
    solve_seed = int(solver_config["seed"])
    time_limit = solver_config["time_limit"]
    no_improvement_seconds = solver_config["no_improvement_seconds"]
    fleet_mode: FleetMode = solver_config["fleet_mode"]
    fleet_size = solver_config["fleet_size"]
    workers = int(solver_config["workers"])

    logger.info(
        "run_dir=%s sizes=%s num_instances=%d solver=%s workers=%d",
        run_dir,
        sizes,
        config["num_instances"],
        solver,
        workers,
    )

    wall_start = time.perf_counter()
    datasets: dict[int, CVRPDataset] = {}
    for size in sizes:
        datasets[size] = _generate_size(run_dir, size, config)

    labels_dir = run_dir / "labels"
    summary_rows: list[dict[str, Any]] = []
    for size in sizes:
        summary_rows.append(
            _solve_size(
                datasets[size],
                size,
                labels_dir,
                solver=solver,
                seed=solve_seed,
                time_limit=time_limit,
                no_improvement_seconds=no_improvement_seconds,
                fleet_mode=fleet_mode,
                fleet_size=fleet_size,
                workers=workers,
            )
        )

    examples_written = 0
    if not args.skip_convert:
        logger.info("exporting CVRPExample JSON files")
        written = export_run(run_dir)
        examples_written = sum(len(paths) for paths in written.values())
        logger.info("wrote %d example files under %s/examples", examples_written, run_dir)

    wall_seconds = time.perf_counter() - wall_start
    dataset_hash = _write_dataset_hashes(run_dir, sizes)
    _write_summary_csv(run_dir, summary_rows)
    summary = {
        "run_dir": str(run_dir),
        "wall_seconds": round(wall_seconds, 1),
        "examples_written": examples_written,
        "dataset_hash": dataset_hash,
        "solver": solver_config,
        "per_size": summary_rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset_hash": dataset_hash,
                "wall_seconds": round(wall_seconds, 1),
                "per_size": summary_rows,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nDone in {wall_seconds / 3600:.2f}h -- run_dir={run_dir}")
    for row in summary_rows:
        print(
            f"  cvrp{row['size']}: {row['n']} instances, "
            f"{row['feasible']}/{row['n']} feasible, mean_cost={row['mean_cost']}"
        )


if __name__ == "__main__":
    main()
