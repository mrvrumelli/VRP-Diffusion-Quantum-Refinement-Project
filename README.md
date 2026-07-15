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
