# Coding Standards

Task ID: P0.3. This document is the accepted coding standard for the **VRP Diffusion + Quantum
Refinement** project. It exists so that every contributor and every coding agent produces code
that is formatted, typed, logged, seeded, and organized the same way. Follow it in every PR; the
PR checklist in `AGENTS.md` assumes it.

Scope: formatting, type hints, logging, seeds, config naming, output folders. For the broader
project rules (pipeline order, testing, quantum guardrails), see `AGENTS.md`.

---

## 1. Formatting and linting

- Python version: 3.11+ (pinned in `pyproject.toml` as `>=3.11,<3.13`).
- Formatter and linter: `ruff`, configured in `pyproject.toml`. Do not hand-format around it or
  add a second formatter.
- Line length: 100 characters.
- Quote style: double quotes.
- Indentation: spaces, LF line endings.
- Lint rule sets enabled: `E`, `F`, `I` (import sorting), `B` (bugbear), `UP` (pyupgrade), `N`
  (naming), `ANN` (annotations), `RUF`.

Run before every commit:

```bash
ruff check .
ruff format .
```

CI/PR gate (see `AGENTS.md` §9):

```bash
ruff check .
ruff format --check .
pytest
mypy src
```

Do not disable a rule inline (`# noqa`) to silence a real issue. If a rule is wrong for the whole
project, change it in `pyproject.toml` and explain why in the PR description.

---

## 2. Type hints

- `mypy` runs in `strict` mode against `src/vrp_diffusion_quantum` (`[tool.mypy]` in
  `pyproject.toml`). New code must pass it.
- Every public function, method, and dataclass field must be typed. Private helpers should be
  typed too unless the type is obvious and purely local.
- Prefer `dataclasses` or `TypedDict` for structured CVRP objects (instances, solutions,
  constraint matrices) instead of bare dicts or tuples. See `AGENTS.md` §7 for the expected fields
  of each object.
- Use precise container types (`list[int]`, `dict[str, float]`, `np.ndarray`, `torch.Tensor`)
  rather than `Any`. If a type genuinely cannot be narrowed, annotate `Any` explicitly and say why
  in a comment.
- Tests are exempt from return-type annotations (`ANN201` is ignored under `tests/**/*.py`) but
  should still type fixtures and helpers where practical.

```python
def route_cost(routes: list[list[int]], coords: np.ndarray) -> float:
    ...

@dataclass
class CVRPInstance:
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    depot_index: int = 0
```

---

## 3. Naming

### Domain variable names

Use the established domain vocabulary consistently across modules, configs, and logs:

```text
coords
customer_coords
demands
capacity
routes
route_cost
constraint_matrix
m_true
m_prob
m_hat
visited_mask
capacity_mask
neighborhood
qubo
solution
```

Avoid ambiguous names such as `data`, `thing`, `mat`, `tmp`, or `result` in public code — say what
the object actually is (`m_hat`, not `mat`; `route_cost`, not `result`).

### Modules, files, functions

- Modules and files: `snake_case.py`.
- Functions and variables: `snake_case`.
- Classes and dataclasses: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Package layout follows `AGENTS.md` §4 (`src/vrp_diffusion_quantum/{data,models,diffusion,
  inference,train,eval,quantum,local_search,utils}`). Put a module under the subpackage that
  matches its pipeline stage, not under `utils/` by default.

### Config naming

- Config files live under `configs/{data,train,eval,quantum}/` and are YAML.
- Name a config file after what it configures plus the variant, e.g.
  `configs/data/cvrp20.yaml`, `configs/train/diffusion_base.yaml`,
  `configs/eval/cvrp50_ablation_no_m.yaml`, `configs/quantum/qaoa_depth2.yaml`.
- Inside a config, use the same domain names as in code (`capacity`, `seed`, `n_customers`), not
  abbreviations invented for the config only.
- Every config must include a `seed` field and enough fields to fully reproduce the run (dataset
  choice, model hyperparameters, solver/method name). Do not rely on code defaults for anything
  that affects a reported result — put it in the config explicitly.
- A `config_name` (the file's stem) is logged with every experiment's metrics so results can be
  traced back to the exact config used.

---

## 4. Logging

- Library code (anything importable under `src/vrp_diffusion_quantum/`) uses the `logging`
  module, never `print`. Get a module-level logger with `logging.getLogger(__name__)`.
- Scripts under `scripts/` may `print` final human-readable summaries, but should still use
  `logging` for progress and diagnostics so verbosity is controllable.
- Do not log secrets, absolute local paths that vary by machine, or large tensors/arrays.
- Every experiment run must produce a log file (`stdout.log` or `run.log`) inside its output
  directory — see §6.

```python
import logging

logger = logging.getLogger(__name__)

def train_epoch(...) -> float:
    logger.info("epoch=%d train_loss=%.4f", epoch, loss)
    ...
```

---

## 5. Seeds and reproducibility

Every stochastic component must accept an explicit seed or generator — never rely on global,
unseeded randomness. This applies to at least:

- CVRP instance generation
- OR label generation, when the solver is stochastic
- constraint-matrix predictor training
- diffusion noising and sampling
- policy (RL) training
- sampling / multi-start inference
- QAOA / annealing / classical local-search experiments

Rules:

- Accept `seed: int` (or an explicit `np.random.Generator` / `torch.Generator`) as a function or
  config parameter. Do not call global seeding functions (`np.random.seed`, `torch.manual_seed`)
  deep inside library code where it can silently affect unrelated callers — seed once, at the
  script/entry-point level, from the config's `seed` field, and thread generators down from there.
- The seed used for a run must be logged (`seed.txt` or the `seed` field in `config.yaml`/
  `metrics.json`) so the run can be reproduced exactly.
- Tests that use randomness must fix the seed so they are deterministic.

---

## 6. Output folders

Every experiment (training or evaluation run) writes to its own output directory containing:

```text
config.yaml            # the exact config used, including seed
metrics.json           # final metrics
summary.csv / summary.json
stdout.log / run.log
seed.txt               # or seed recorded inside config.yaml
commit_hash.txt        # when available
plots/                 # when plots are produced
```

Minimum fields to log, per `AGENTS.md` §10:

- Training runs: `train_loss`, `validation_loss`, matrix metrics when relevant, route cost when
  relevant, `feasibility_rate`, `runtime`, `learning_rate`, `seed`.
- Evaluation runs: `instance_id`, `n_customers`, `method`, `cost`, `gap_to_reference`,
  `runtime_seconds`, `num_vehicles`, `feasible`, `seed`, `config_name`.
- Quantum refinement runs additionally log: `neighborhood_type`, `neighborhood_size`,
  `qubo_num_variables`, `qubo_num_terms`, `penalty_weights`, `solver_name`,
  `num_samples`/`shots`, `qaoa_depth` when relevant, `raw_energy`, `accepted_improvement`,
  `post_repair_feasible`.

Do not commit generated output directories, datasets, checkpoints, logs, or plots to git unless
explicitly requested — keep them under `data/raw/`, `data/processed/`, or an experiment output
root that is gitignored. Only small, hand-checkable samples belong in `data/samples/`.

---

## 7. Enforcement

This file is required reading before opening a PR (see the PR checklist in `AGENTS.md` §13). A
reviewer should reject a PR that:

- fails `ruff check`, `ruff format --check`, or `mypy src`,
- adds an untyped public function or an untyped dataclass field,
- uses `print` inside library code,
- introduces a stochastic component without a seed parameter,
- adds a config without a `seed` field,
- writes experiment outputs without the files listed in §6.
