"""Experiment-tracking template: config, dataset hash, seed, metric table, plots, logs.

Every training or evaluation run should open one `ExperimentTracker` and use it for the
lifetime of the run. See docs/coding_standards.md section 6 for the output-directory contract
this module implements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

import yaml

_HASH_CHUNK_BYTES = 1 << 20


class _HashDigest(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _update_length_prefixed(digest: _HashDigest, data: bytes) -> None:
    digest.update(len(data).to_bytes(8, byteorder="big", signed=False))
    digest.update(data)


def hash_dataset(path: str | Path) -> str:
    """Return a deterministic sha256 hex digest for a dataset file or directory.

    Directory hashes include relative file paths, so renaming or removing a file changes the
    hash even if remaining file contents are unchanged.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")

    if dataset_path.is_file():
        files = [dataset_path]
        base = dataset_path.parent
    else:
        files = sorted(p for p in dataset_path.rglob("*") if p.is_file())
        base = dataset_path

    digest = hashlib.sha256()
    for file_path in files:
        relative_path = file_path.relative_to(base).as_posix().encode("utf-8")
        _update_length_prefixed(digest, relative_path)
        digest.update(file_path.stat().st_size.to_bytes(8, byteorder="big", signed=False))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    """Return the current git HEAD commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


class ExperimentTracker:
    """Creates a reproducible experiment output directory and logs a run into it.

    Layout written under `self.run_dir`:

    ```text
    config.yaml       the exact config used for this run (seed and experiment_name included)
    seed.txt          the seed used for this run
    commit_hash.txt   git HEAD commit hash, when available
    dataset_hash.txt  sha256 of the dataset used, when a dataset_path is provided
    run.log           full log stream for the run
    metrics.json      final/summary metrics, updated via log_metrics()
    summary.csv       per-step/epoch metric table, appended via log_metric_row()
    plots/            directory for saved plots
    ```
    """

    def __init__(
        self,
        output_root: str | Path,
        experiment_name: str,
        config: dict[str, Any],
        seed: int,
        dataset_path: str | Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.seed = seed

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = Path(output_root) / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir()

        resolved_config = dict(config)
        resolved_config.setdefault("seed", seed)
        resolved_config.setdefault("experiment_name", experiment_name)
        with (self.run_dir / "config.yaml").open("w") as handle:
            yaml.safe_dump(resolved_config, handle, sort_keys=False)

        (self.run_dir / "seed.txt").write_text(f"{seed}\n")

        commit_hash = git_commit_hash()
        if commit_hash is not None:
            (self.run_dir / "commit_hash.txt").write_text(f"{commit_hash}\n")

        self.dataset_hash: str | None = None
        if dataset_path is not None:
            self.dataset_hash = hash_dataset(dataset_path)
            (self.run_dir / "dataset_hash.txt").write_text(f"{self.dataset_hash}\n")

        self._metrics: dict[str, Any] = {}
        self._summary_path = self.run_dir / "summary.csv"
        self._summary_fieldnames: list[str] | None = None

        self.logger = logging.getLogger(f"experiment.{experiment_name}.{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._file_handler: logging.Handler = logging.FileHandler(self.run_dir / "run.log")
        self._stream_handler: logging.Handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._file_handler.setFormatter(formatter)
        self._stream_handler.setFormatter(formatter)
        self.logger.addHandler(self._file_handler)
        self.logger.addHandler(self._stream_handler)

        self.logger.info(
            "started experiment=%s seed=%d dataset_hash=%s commit=%s run_dir=%s",
            experiment_name,
            seed,
            self.dataset_hash,
            commit_hash,
            self.run_dir,
        )

    def log_metric_row(self, row: dict[str, Any]) -> None:
        """Append one row (e.g. one epoch or eval instance) to summary.csv."""
        if self._summary_fieldnames is None:
            self._summary_fieldnames = list(row.keys())
        is_new_file = not self._summary_path.exists()
        with self._summary_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._summary_fieldnames)
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
        self.logger.info("metric_row %s", row)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge and persist final/summary metrics to metrics.json."""
        self._metrics.update(metrics)
        with (self.run_dir / "metrics.json").open("w") as handle:
            json.dump(self._metrics, handle, indent=2, sort_keys=True)
        self.logger.info("metrics %s", metrics)

    def close(self) -> None:
        handlers: tuple[logging.Handler, logging.Handler] = (
            self._file_handler,
            self._stream_handler,
        )
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self) -> ExperimentTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
