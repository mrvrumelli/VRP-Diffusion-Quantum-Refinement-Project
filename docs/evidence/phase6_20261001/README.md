# Phase 6 evidence — 2026-10-01 to 2026-10-02

Small result files behind [the Phase 6 write-up](../../phase6_refinement_experiments_2026-10-01.md)
and [the quantum results interpretation](../../quantum_results.md), copied from the gitignored
`outputs/phase6_20261001/` by `scripts/collect_evidence.py` in this folder. `manifest.json` lists the SHA-256 of
every copied file and of the two frozen neighbourhood sets.

| Folder or file | Contents |
|---|---|
| `experiment_*` | Single-subproblem comparisons at a 0.25 s budget (`*_sqa`: second run with SQA) |
| `loop_*` | Three-round refinement loops |
| `extension/` | Rounds and neighbourhood grid, Q5 polish, QAOA in the loop, seeds, strong bias, time-matched control, generated tables |
| `selection/` | Neighbourhood-selection ablations, both selectors and each alone |
| `bias_sweep/` | Bias strength and mode sweep |
| `calibration/` | Calibration and uncertainty analysis of the prior |
| `qaoa_*`, `exchange_capacity_violations_*` | QAOA screening reports, random control, warm start, capacity violations |
| `final/` | The once-only final test on the reserved test and R/C/RC cells, under the declared protocol |
| `timing/` | The clean single-process timing pass |
| `scripts/` | The analysis scripts and queue drivers used, as run |

Full per-graph rows (`rows.jsonl`) stay in `outputs/`; they are large and regenerable with the
commands in the write-up's reproduction table.
