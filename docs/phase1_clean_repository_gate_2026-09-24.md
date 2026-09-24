# Phase 1 clean repository gate

**Verified:** 2026-09-24  
**Platform:** Windows, Python 3.12.10, CPU-only clean test environment

## Fresh installation

The gate was reproduced in a newly created `.phase1-venv`, without access to packages from the
existing project environment:

```powershell
python -m venv .phase1-venv
.phase1-venv\Scripts\python.exe -m pip install --upgrade pip
.phase1-venv\Scripts\python.exe -m pip install -e ".[dev]"
.phase1-venv\Scripts\python.exe -m pip check
```

Installed gate versions:

- Python 3.12.10
- PyVRP 0.14.0
- PyTorch 2.14.0+cpu
- Ruff 0.16.8
- pytest 9.1.1

`pip check` reported no broken requirements. The editable package build and dependency resolution
both completed successfully.

## Required checks

With `CUDA_VISIBLE_DEVICES=-1`:

```powershell
.phase1-venv\Scripts\ruff.exe check .
.phase1-venv\Scripts\ruff.exe format --check .
.phase1-venv\Scripts\pytest.exe -q -m paper_smoke
.phase1-venv\Scripts\pytest.exe -q
```

Results:

- Ruff lint: passed.
- Ruff formatting: 152 files already formatted.
- Explicit `paper_cmd` CPU smoke: 1 passed.
- Complete CPU suite: 489 passed, 4 CUDA-only tests skipped.

The paper smoke runs the frozen pretrained-global-GAT path, masked local GAT, sum-MLP fusion,
dual-pointer decoder, tiny RL optimization, and feasibility checks entirely on CPU. It asserts
that optimization lowers validation cost, feasibility remains 100%, and the frozen GAT does not
change.

## Repository hygiene changes

- Ruff-formatted the Python source, tests, scripts, and evaluation entry points.
- Added LF rules in `.gitattributes` to make formatting stable across Windows and Unix checkouts.
- Excluded repository-local pytest scratch directories and the verification environment from Ruff.
- Registered the `paper_smoke` pytest marker.

The only warnings in the CPU suite were three third-party SWIG deprecations and the existing
pytest-cache permission warning. They do not affect test results.

