# VRP Diffusion Quantum Refinement Project

Reproducible research code for a Constraints Matrix Diffusion based CVRP solver, with
quantum and quantum-inspired refinement scoped to small local neighborhoods after the
classical baseline is stable.

## Setup

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

The base environment includes PyTorch, PyVRP, OR-Tools, MLflow, TensorBoard,
scientific Python packages, and plotting libraries.

Optional extras:

```bash
# notebooks
python -m pip install -e ".[dev,notebooks]"

# W&B tracking support
python -m pip install -e ".[dev,tracking]"

# local quantum and quantum-inspired experiments
python -m pip install -e ".[dev,quantum]"
```

For CUDA-specific PyTorch builds, install the appropriate PyTorch wheel for the target
machine before running the editable install.

## Development Commands

```bash
ruff check .
ruff format .
pytest
mypy src
```

Run the current smoke tests with:

```bash
pytest tests
```

## Phase 1 — Generate and label CVRP data

Run the reproducible headless pipeline:

```bash
python scripts/run_data_pipeline.py \
  --run-name baseline_data_v1 \
  --sizes 20 50 100 \
  --num 10000 \
  --workers 4
```

The command generates instances, labels them with the configured PyVRP/OR-Tools solver, validates
and exports modeling examples, and writes `config.yaml`, dataset hashes, labels, `metrics.json`,
`summary.csv`, `summary.json`, `run.log`, and the commit hash. Re-run the identical command with
the same `--run-name` to resume; saved generation and solver settings remain authoritative.

For a small check:

```bash
python scripts/run_data_pipeline.py \
  --data-root /tmp/vrp_phase1 \
  --run-name smoke \
  --sizes 20 \
  --num 2 \
  --time-limit 0.1
```

Visualize exported routes and route-membership matrices:

```bash
python scripts/visualize_dataset.py \
  --dataset-dir data/raw/cvrp/baseline_data_v1/examples \
  --output-dir outputs/phase1_sanity_plots \
  --max-examples 20
```

## Phase 2 — Supervised matrix predictor

The committed sanity configuration exercises a genuine held-out split:

```bash
python scripts/train_matrix_predictor.py \
  --config configs/train/matrix_predictor_sanity.yaml
```

For a real experiment, copy that config and set `dataset.path` to the Phase 1 run's `examples/`
directory. The run logs held-out matrix metrics and nearest-neighbor, demand-aware, seeded-random,
and all-zero baselines, saves `model.pt`, and writes predicted-versus-ground-truth plots.

Plots can also be generated directly from any Phase 1 export:

```bash
python scripts/visualize_matrix_predictions.py \
  --dataset-dir data/raw/cvrp/baseline_data_v1/examples \
  --validation-fraction 0.2 \
  --max-plots 20
```

See `docs/report_m2.md` for the Phase 2 method, limitations, and reporting contract.

After the Phase 1 and Phase 2 runs finish, build the data-quality, labeling-speed, and supervised
predictor result tables from their logged artifacts:

```bash
python scripts/build_phase2_report.py \
  --phase1-run data/raw/cvrp/baseline_data_v1 \
  --phase2-run outputs/train/<phase2-run>
```
