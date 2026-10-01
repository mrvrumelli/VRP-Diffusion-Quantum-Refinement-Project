# Quantum refinement scope

Task Q1.1. This document fixes what the quantum and quantum-inspired stage may do, what it may
claim, and how it is compared. It is a proposal for team agreement; nothing here is a result.

## Position in the pipeline

The learned CVRP solver produces a complete feasible solution. Refinement then improves that
solution locally:

1. Select a small neighborhood of the current solution (`quantum/neighborhoods.py`).
2. Extract a self-contained subproblem with fixed boundary conditions.
3. Solve the subproblem with a QUBO solver or a classical solver on exactly the same subproblem.
4. Decode, repair if needed, and evaluate the true route cost with the project's `route_cost`.
5. Accept the change only if the full solution stays feasible and its true cost strictly falls.

Step 5 makes refinement monotone: it can never make a solution worse. Refinement never replaces
the constructive solver and never solves a whole CVRP instance as one QUBO.

Dependency: initial solutions come from the frozen classical baseline (task P5.6). Until that
freeze is committed (see [the freeze proposal](classical_baseline_freeze_proposal_2026-09-30.md)),
formulations and tests may proceed, but no refinement result may be reported as final.

## Allowed subproblems

| Subproblem | QUBO module | Decision | Variables |
|---|---|---|---|
| Reorder one route or route segment between fixed endpoints | `qubo_reorder.py` | Visiting order of k customers | k² (one-hot position encoding) |
| Exchange or reassign customers between two routes | `qubo_exchange.py` | Route membership of m movable customers | m plus capacity slack bits (at most about 12 for capacity 50) |

Neighborhood types that feed them (`neighborhoods.py`):

- **High-cost route.** Routes with the highest cost per customer; long routes are cut into a
  segment around their most expensive edge.
- **Uncertain matrix entries.** Customers whose predicted same-route probability to their own
  route barely exceeds, or falls below, that to a neighbouring route. Requires `m_prob`.
- **Low-confidence edges.** Consecutive customers whose predicted same-route probability is low;
  the neighborhood is a segment around that edge.
- **Two-route exchange.** Spatially adjacent route pairs; the movable customers are those of
  each route that are closest to the other route.

## Size limits

| Use | Reorder segment length k | Exchange movable customers m | Binary variables |
|---|---:|---:|---:|
| Exhaustive checks in tests | ≤ 4 | ≤ 8 | ≤ 20 |
| Statevector simulation (QAOA and similar) | ≤ 5 | ≤ 14 | ≤ 25–28 |
| Simulated annealing and other classical QUBO heuristics | ≤ 10 | ≤ 24 | ≤ 100 |
| Hardware | Decided only after simulation screening | | |

These limits are configuration defaults, not claims about any device.

## Comparison protocol

- Every quantum or quantum-inspired run is paired with classical solvers on the **identical**
  subproblem, from the identical initial solution (`local_search/baselines.py`). The classical
  references are an exact solver where feasible (dynamic programming or enumeration) and a
  standard local search (2-opt; relocate and swap).
- Simulated annealing on the same QUBO is the quantum-inspired control. It separates the effect of
  the formulation from the effect of the quantum solver.
- Budgets are matched and stated explicitly: wall-clock time, number of samples or shots, and
  number of objective evaluations.
- Report four configurations from identical starting solutions: learned solver only; plus
  classical refinement; plus quantum refinement; quantum refinement followed by classical polish.
- Development uses neighborhoods from the development panel. The reserved test set stays untouched
  until the method and budgets are fixed.

## Claims

Allowed, when supported by the protocol above:

- Solution quality and runtime of each method on the same set of small subproblems.
- Whether diffusion-biased QUBO terms change solution quality, with the bias toggled off as control.
- Feasibility and repair rates of decoded QUBO solutions.

Not allowed:

- Any claim of quantum advantage, speedup, or scalability from these experiments.
- Any statement about full-instance CVRP solving by quantum methods.
- Hardware conclusions drawn from simulation, or simulation conclusions drawn from a few runs.
- Reporting only improvements: subproblems where no method improved, and runs whose decoded
  output was infeasible, are reported too.

## Formulation conventions

- A QUBO is a matrix `Q` and constant `offset`, with energy `xᵀQx + offset` over binary `x`.
  Linear terms sit on the diagonal. Variable labels are kept with the matrix.
- Constraint penalties are explicit and logged. The reorder QUBO's permutation penalty and the
  exchange QUBO's capacity penalty both scale with the subproblem's largest distance.
- The exchange QUBO uses a surrogate cost (insertion cost to each route plus within-route
  dispersion), because true route cost depends on ordering. Its decoded solutions are always
  re-evaluated with true route cost before acceptance.
- Diffusion-biased terms (`qubo_bias.py`) are optional, off by default, and weighted by a single
  configurable `alpha`.

## Logging

Per the coding standards, every refinement run logs `neighborhood_type`, `neighborhood_size`,
`qubo_num_variables`, `qubo_num_terms`, `penalty_weights`, `solver_name`, `num_samples` or `shots`,
`qaoa_depth` when relevant, `raw_energy`, `accepted_improvement` and `post_repair_feasible`, in
addition to the standard evaluation fields.

## Decisions for the team

1. Agree the size limits above, or replace them with limits from a specific simulator or device.
2. Agree that the exchange QUBO's surrogate objective is acceptable, given true-cost acceptance.
3. Agree the matched-budget unit for the first experiments: wall-clock time or number of samples.
