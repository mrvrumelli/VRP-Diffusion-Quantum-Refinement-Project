Q1.1-Q1.6 implementation review, 2026-10-01
==========================================

**Fix status: all 11 findings resolved.** The final targeted run passed **217 tests, with no
failures or skips**, including the 23 originally failing acceptance cases and both real dimod
conversion checks. The former audit checks are now in the normal test suite:
[independent checks](../tests/test_quantum_audit.py) and
[acceptance regressions](../tests/test_quantum_acceptance.py). The latter also adds 24 boundary
cases around the fixes. Existing tests were retained unchanged.

| Finding | Implemented resolution |
|---|---|
| 1: missing BQM variables | Export includes every label, even when its linear and quadratic coefficients are zero. |
| 2: invalid binary states | A shared validator checks exact vector shape and binary values before loads, decoding, or repair; slack bits are checked too. |
| 3: stale reorder application | Application checks route bounds, original segment order, and both fixed endpoints. |
| 4: fixed exchange membership | Application checks each route's fixed customers and the original route snapshot before replacing routes. |
| 5: prior dimensions | Extracted subproblems retain the full customer count; active bias validates exact dimensions and referenced customer indices. |
| 6: starting cost | Exchange subproblems retain immutable original routes; `cost_before` now includes their original visiting order. The reproduced case reports 12 to 10, improvement 2. |
| 7: repair tie-breaker | Subset dynamic programming maximises sample agreement first, then minimises full path cost including endpoints. |
| 8: overlapping windows | The selector greedily retains the weakest edge's window and suppresses every overlapping window. |
| 9: constant QUBO annealing | The solver returns the requested number of empty samples at the constant energy. |
| 10: fractional large demands | Integral checks use absolute tolerance only (`rel_tol=0`). |
| 11: unknown singleton solver | Method validation now runs before the size shortcut. |

Exact repair of an invalid sample is bounded at 12 customers; valid permutations pass through
at any size. The documented heuristic scope is at most 10 customers. Manually constructed
exchange subproblems lacking original routes retain a documented reconstructed-cost fallback;
the extraction APIs preserve the actual starting solution for matched comparisons.

Final validation: 217 tests passed; Ruff lint and formatting passed; strict mypy passed for all
nine quantum/local-search source files. The implementation has 95.5% statement coverage and
85.8% branch coverage (92.9% combined), including the real dimod run. Evidence:
[JUnit results](evidence/quantum_fixes_20261001/targeted.xml),
[coverage](evidence/quantum_fixes_20261001/coverage.json), and
[fix manifest](evidence/quantum_fixes_20261001/manifest.json).

Current reproduction command (with the isolated dimod installation available):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'outputs/quantum_review_dependencies')
.\.venv\Scripts\python.exe -m pytest tests/test_quantum_neighborhoods.py tests/test_qubo_reorder.py tests/test_qubo_exchange.py tests/test_qubo_bias.py tests/test_local_search_baselines.py tests/test_quantum_audit.py tests/test_quantum_acceptance.py -q -p no:cacheprovider
```

**Original audit, before fixes.** The remaining findings and line references record the source
as initially reviewed. The original XML, coverage, manifest, and reproduction files are preserved
as historical evidence; the passing results above supersede their failure status.

All six requested tasks were reviewed. Each existing executable task suite was run separately.
The 66 existing tests passed. Independent formula, enumeration, boundary, integration, and
acceptance tests produced **170 passing cases and 23 failing cases in total**, including the
existing tests. The failures demonstrate **11 grouped findings** below; several are input
validation gaps or documentation/reporting mismatches rather than failures on valid normal inputs.

During the initial review, production source and existing tests were not changed. Reproductions
were kept outside normal `tests/` discovery and failed against that initial implementation.

| Task | Existing tests | Additional passing cases | Failing acceptance cases | Assessment |
|---|---:|---:|---:|---|
| Q1.1: scope document | Manual review | Document/link checks | 0 | Requested scope, limits, protocol, claims, and three decisions are present. |
| Q1.2: neighborhoods | 9 | 12 | 6 | Selection/extraction invariants pass; application guards, assignment validation, and overlap documentation have gaps. |
| Q1.3: core and reorder | 14 | 26 | 5 | Energy/exact/annealing checks pass; BQM export, invalid decoding, repair wording, and empty annealing have issues. |
| Q1.4: exchange | 13 | 24 | 7 | Capacity formulation passes exhaustive checks; decode/repair validation and large fractional input rejection have gaps. |
| Q1.5: diffusion bias | 10 | 20 | 2 | Both formulas and feasibility penalties pass; prior dimensions are insufficiently validated. |
| Q1.6: classical baselines | 20 | 14 | 3 | Optimality and acceptance checks pass; starting-cost interpretation and singleton method validation need attention. |
| Cross-module anneal/decode/repair checks | 0 | 8 | 0 | Feasible repaired outputs and matched classical references agree. |
| Total executable cases | 66 | 104 | 23 | 193 distinct cases, including two real dimod tests. |

The combined run without dimod was 169 passed / 22 failed / 2 deselected. The separate run with
real dimod was 1 passed / 1 failed / 117 deselected; it preceded addition of the eight integration
cases. Deselections are accounted for by the other run, not counted as skipped coverage.

Combined coverage for the six implementation modules was 94.3% of statements and 83.1% of
branches (91.4% combined). The real dimod run was separate and is not included in this coverage.
Ruff checks and formatting checks passed for the reviewed source and both new audit files.

**Findings, ordered by practical impact**

1. **Q1.3: BQM conversion drops zero-coefficient decision variables.**
   Location: [qubo.py](../src/vrp_diffusion_quantum/quantum/qubo.py), `to_dicts` at line 85 and
   `to_bqm` at line 99. `to_dicts` deliberately omits zero coefficients, but `to_bqm` never
   restores variables absent from both coefficient dictionaries. A normal one-customer exchange
   between symmetric routes has one assignment variable and a constant objective. Its QUBO has
   label `y[c=2]`; the actual dimod BQM has **zero variables**. A backend therefore cannot return
   the assignment required by `decode_exchange`. Tested with dimod 0.12.22, not a mock.
   Preserve every QUBO label in the BQM, including zero-bias variables.
   Reproduction: `test_q13_bqm_retains_zero_coefficient_variables`.

2. **Q1.2/Q1.3/Q1.4: invalid states can decode or be certified feasible.**
   Locations: [neighborhoods.py](../src/vrp_diffusion_quantum/quantum/neighborhoods.py),
   `loads`/`is_feasible` at lines 174/180;
   [qubo_reorder.py](../src/vrp_diffusion_quantum/quantum/qubo_reorder.py), `decode_reorder`
   at line 121; [qubo_exchange.py](../src/vrp_diffusion_quantum/quantum/qubo_exchange.py),
   `decode_exchange`/`repair_exchange` at lines 216/223.
   With loose capacity, assignments `(2,)`, `(-1,)`, and `(0.5,)` all pass `is_feasible`.
   Exchange repair returns `(2,)` and `(-1,)` unchanged, and silently truncates `(0.5,)` to `(0,)`.
   Reorder decoding accepts the nonbinary matrix `[[2,-1],[-1,2]]`, since its row and column sums
   are one; fractional values are also truncated before checking. This matters when integrating
   a backend that returns spins, relaxed values, or a malformed state. Native tested solvers
   return binary states, so these failures require invalid external input. Validate shape and
   binary values before casts, loads, decoding, or repair. Eleven parametrized failing cases
   cover this group.

3. **Q1.2: applying a stale reorder subproblem can lose and duplicate customers.**
   Location: [neighborhoods.py](../src/vrp_diffusion_quantum/quantum/neighborhoods.py),
   `apply_reorder` at line 500. Extract from `[[0,1],[2]]`, then apply that saved subproblem to
   updated routes `[[0,2],[1]]` with order `(1,0)`: the result is `[[1,0],[1]]`. Customer 2 is lost
   and customer 1 appears twice. Extraction checks staleness, but application checks only the
   proposed permutation. Revalidate the target segment and boundaries at application time.
   The current synchronous `refine` path extracts immediately before solving and did not exhibit
   this failure; batching or caching subproblems can expose it.
   Reproduction: `test_q12_apply_reorder_rejects_stale_routes`.

4. **Q1.2: exchange application does not preserve fixed route membership.**
   Location: [neighborhoods.py](../src/vrp_diffusion_quantum/quantum/neighborhoods.py),
   `apply_exchange` at line 566. For routes `[[0],[1,2]]` with only customer 2 movable, applying
   `[[1,2],[0]]` succeeds. Both fixed customers changed routes, outside the declared subproblem.
   The union-of-customers check is insufficient; check each fixed set against its own route.
   Existing solvers honor the fixed sets, so this is a public application-boundary validation gap.
   Reproduction: `test_q12_apply_exchange_rejects_moving_fixed_customers`.

5. **Q1.5: active bias accepts a prior with the wrong customer dimension.**
   Location: [qubo_bias.py](../src/vrp_diffusion_quantum/quantum/qubo_bias.py),
   `_validate_pair_prob` at line 83 and `build_biased_exchange_qubo` at line 220. For a six-customer
   instance, a 7x7 matrix is silently accepted; a 5x5 matrix reaches an `IndexError`. Only
   squareness and probability range are checked. A depot-inclusive matrix can consequently be
   accepted with incorrectly aligned customer indices. Require exactly `(n_customers, n_customers)`
   where the instance is available, and validate referenced customer indices for reorder-only
   APIs. Disabled bias correctly leaves the QUBO unchanged and does not need a usable prior.
   Reproduction: `test_q15_exchange_prior_shape_matches_instance` (two cases).

6. **Q1.6: `ExchangeSolution.cost_before` is the reconstructed cost, not necessarily the
   supplied solution's starting cost.**
   Location: [baselines.py](../src/vrp_diffusion_quantum/local_search/baselines.py), line 260;
   [neighborhoods.py](../src/vrp_diffusion_quantum/quantum/neighborhoods.py), `evaluate_exchange`.
   Customers at `(1,0),(2,0),(3,0),(4,0),(-1,0)`, capacity 4, routes `[[0,2,1,3],[4]]`, and only
   customer 4 movable give an actual starting cost of **12**. Both exchange methods report
   `cost_before=10`, `cost_after=10`, and zero improvement because nearest-neighbor/2-opt has
   already polished the original assignment during evaluation. `refine` correctly reports 12 to
   10 and improvement 2. This is a **reporting/contract ambiguity**, not incorrect enumeration:
   the module explicitly defines its assignment objective using reconstructed routes. Do not use
   its `improvement` as the full-solution improvement in matched comparisons. Preserve the actual
   original cost in the subproblem, or name/document the metric as the reconstructed starting cost.
   The two failing acceptance tests enforce the scope protocol's interpretation of a supplied
   starting solution; an explicit narrower metric contract is another resolution.

7. **Q1.3: reorder repair does not implement its stated true-path-cost tie-breaker.**
   Location: [qubo_reorder.py](../src/vrp_diffusion_quantum/quantum/qubo_reorder.py),
   `repair_reorder` at line 132. It minimizes an endpoint-only surrogate after maximizing bit
   agreement. With depot `(0,0)`, customers `(3,1),(0,-2),(-2,-4),(-4,-4)`, and an all-zero sample,
   every permutation has equal agreement. Repair returns cost **18.7054814270**, while an equally
   agreeing permutation costs **17.8901997215**. The output is valid and maximum-agreement;
   the narrower endpoint heuristic should be documented, or the advertised tie-break implemented.
   Reproduction: `test_q13_repair_uses_documented_true_path_cost_tiebreak`.

8. **Q1.2: the low-confidence selector emits overlapping windows despite its docstring.**
   Location: [neighborhoods.py](../src/vrp_diffusion_quantum/quantum/neighborhoods.py),
   `select_low_confidence_edges` at line 326, deduplication at line 353. Six customers with uniformly
   weak edges and segment size 3 yield `(0,3)`, `(1,4)`, `(2,5)`, `(3,6)`. The implementation only
   removes identical windows. The docstring says overlapping windows keep the lowest edge.
   This can use selection budgets on strongly overlapping work and requires reselection after
   accepted changes. Either change overlap handling or narrow the documentation to identical
   windows. Reproduction: `test_q12_low_confidence_windows_obey_documented_overlap_rule`.

9. **Q1.3: annealing crashes on a valid constant, zero-variable QUBO.**
   Location: [qubo.py](../src/vrp_diffusion_quantum/quantum/qubo.py), line 214. The builder and exact
   solver support this case, but annealing's default temperature calculation takes `max()` of an
   empty matrix. Return constant empty samples or explicitly reject this case consistently.
   Current reorder/exchange builders reject zero free customers, so this is a shared-core edge
   case. Reproduction: `test_q13_annealing_supports_valid_constant_qubo`.

10. **Q1.4: the integral-demand check accepts material fractional parts at large magnitudes.**
    Location: [qubo_exchange.py](../src/vrp_diffusion_quantum/quantum/qubo_exchange.py), `_integral`
    at line 84. `math.isclose` retains its default relative tolerance, so demand `1000000000.25`
    passes the integral check and is rounded. The encoded load then differs from the original.
    Use `rel_tol=0` with the chosen absolute tolerance. This does not affect the tested small
    integer-capacity use cases and is low priority for the documented experiments.
    Reproduction: `test_q14_exact_slack_rejects_materially_fractional_large_demands`.

11. **Q1.6: singleton reorder subproblems silently accept unknown method names.**
    Location: [baselines.py](../src/vrp_diffusion_quantum/local_search/baselines.py), lines 166-180.
    `solve_reorder(singleton, "typo")` succeeds and records the invalid solver name because the
    size shortcut precedes method validation. Validate the method before that shortcut.
    `refine` separately validates method names and is unaffected.
    Reproduction: `test_q16_singleton_rejects_unknown_solver_name`.

**What was verified**

Q1.1 has the requested refinement-only pipeline, fixed-boundary/shared-subproblem requirement,
strict feasible-improvement acceptance rule, three size-limit rows plus hardware deferral,
matched initial solutions and budgets, allowed/disallowed claims, and exactly three team decisions.
Its local link resolves. It explicitly remains a proposal; its referenced baseline-freeze proposal
also remains open. This audit supplies formulation/test evidence, not final experimental results
or quantum-advantage claims. Size/budget tables are policy guidance rather than an implemented
matched-budget execution harness; the six requested modules do not provide such a harness.

Q1.2 was exercised across 12 seeded geometries, nonzero depot indices, empty route slots, all four
selectors, selector caps/determinism, permutation cost deltas, assignment loads, complete customer
coverage, fixed membership, and unchanged input routes.

Q1.3 was checked against independent scalar quadratic sums; exact enumeration and seeded
annealing; every binary state for reorder sizes 1-4; zero, small, normal, and large coordinate
scales; all valid permutations; maximum-agreement repair; the exact-size guard; and actual dimod
energy round trips. Heuristic annealing was not assumed to find the exact optimum on every case.

Q1.4 was checked by enumerating assignments and slack states. The minimum energy over slack bits
equals the surrogate plus the exact squared positive capacity violations. Tests cover slack
representability for every upper bound 0-128, both/one/no binding constraints, zero residual
capacity, zero demands, impossible partitions, multiple capacities/geometries, feasible-state
preservation in repair, and infeasible-energy separation. Greedy repair can legitimately return
`None` on a feasible problem when it cannot find an admissible move sequence; no completeness
guarantee was assumed.

Q1.5's probability and confidence modes were checked directly against independent formulas for
all assignments/permutations in the fixtures, including probabilities 0/0.5/1, alpha 0.01/1/100,
disabled/zero-alpha identity, zero-confidence contribution, and feasible ground states under
strong bias.

Q1.6's Held-Karp outputs were compared with all six-customer permutations; exhaustive exchange
with all feasible assignments; local search with exact lower bounds and its starting cost;
full-solution refinement with true route cost and capacity/coverage validation. A deliberately
injected cheaper but overloaded candidate was rejected. Exact size guards also passed.

**Evidence and reproduction**

- [Independent checks](evidence/quantum_review_20261001/test_quantum_audit.py)
- [Failing acceptance cases](evidence/quantum_review_20261001/test_quantum_regressions.py)
- [Final targeted JUnit results](evidence/quantum_review_20261001/final_targeted.xml)
- [Coverage data](evidence/quantum_review_20261001/coverage.json)
- [Environment, hashes, and result manifest](evidence/quantum_review_20261001/manifest.json)

From the repository root, on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_quantum_neighborhoods.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_qubo_reorder.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_qubo_exchange.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_qubo_bias.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_local_search_baselines.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest docs/evidence/quantum_review_20261001/test_quantum_audit.py -q -p no:cacheprovider -k 'not bqm'
.\.venv\Scripts\python.exe -m pytest docs/evidence/quantum_review_20261001/test_quantum_regressions.py -q -p no:cacheprovider -k 'not bqm'
```

The last command failed on the original source and now passes after the fixes. To select a task within either new
file, use `-k q12`, `-k q13`, etc.; use `-k integration` for the cross-module checks.

The optional dependency was installed into `outputs/quantum_review_dependencies` with `--no-deps`,
leaving the project's virtual environment unchanged. For the two real BQM checks:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'outputs/quantum_review_dependencies')
.\.venv\Scripts\python.exe -m pytest docs/evidence/quantum_review_20261001/test_quantum_audit.py docs/evidence/quantum_review_20261001/test_quantum_regressions.py -q -p no:cacheprovider -k bqm
```

Network installation and these two tests required the approved elevated tool context because
Windows sandbox permissions blocked downloading/reading the isolated dependency. The actual
conversion test completed. No QAOA, hardware, or full model-training benchmarks were run; those
are outside the requested six implementations.
