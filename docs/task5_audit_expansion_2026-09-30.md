# Task 5 — audited-pool expansion audit — 2026-09-30

Track A Task 5 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
expand the audited IID source pool so the 500/1,000/2,000-per-size learning curve becomes
possible, then materialize it.

## Frozen inputs

- Expansion pool: `data/processed/s7799_audit_expansion_1500` — 1,566/1,582/1,597 examples
  (N20/N50/N100), drawn from `cvrp_s7799_n20-50-100_x66667/splits/train`, verified zero
  instance-id overlap against `label_audit_s7799`, `paper_cmd_hgs_pilot_5k`, and
  `paper_cmd_hgs_full_50k` before labeling.
- Audit config: `configs/data/label_audit_s7799_expansion.yaml` — same protocol as the original
  strong-reference audit: 4 PyVRP base seeds `[73021-73024]`, 10/20/40s time budgets by size,
  OR-Tools challenge on every unstable case plus a 50/size stable control sample.
- Output: `outputs/label_audit/s7799_expansion_1500/`.

## Run history

Launched 2026-09-29 (`outputs/logs/task5_audit_expansion_20260929.log`). Killed intentionally
partway through (user requested a pause) at roughly 1,120-4,200+/18,980 PyVRP tasks done — by that
point the run had actually progressed further than the last live check showed, since PyVRP had
in fact finished completely (18,980/18,980 cached) and moved into the OR-Tools challenge phase
(1,175/~1,414 done) before being stopped. Every individual solve result is cached to disk
atomically as it completes and re-checked by task signature on restart
(`_read_cached_candidate`/`_atomic_write_json` in `label_audit.py`), so nothing already computed
was lost.

Resumed 2026-09-30 (`outputs/logs/task5_audit_expansion_20260930.log`): correctly detected
18,741/18,980 valid cached PyVRP candidates and only recomputed the 239 that were in-flight at
kill time, then ran the remaining OR-Tools challenges to completion. Confirms the audit's
resume path is genuinely reliable, not just resilient in theory.

## Result

**Complete, 0 errors.**

| Metric | Value |
|---|---:|
| Instances | 4,745 (1,566/1,582/1,597 by size) |
| PyVRP runs | 18,980/18,980 |
| OR-Tools challenges | 1,414/1,414 |
| Solver errors | 0 |
| Reference accepted | 3,823 / 4,745 |
| Matrix target accepted | 3,480 / 4,745 |
| OR-Tools beats PyVRP | 1 (of 1,414 checked) |
| Needs review | 1,265 |
| Mean matrix disagreement from original | 3.85% |

Same acceptance shape as the original 500/size audit — the vast majority of instances land a
stable, near-best PyVRP reference across all 4 seeds; a small minority need OR-Tools review, and
essentially none are cases where OR-Tools actually beats PyVRP's best seed. No new failure mode
introduced by the larger, more size-imbalanced pool.

## What this unblocks

The combined original + expansion pool (materialized via
[`outputs/_scratch/build_task5_curve.py`](../outputs/_scratch/build_task5_curve.py)) now has:

| Size | Original (train-profile) | + Expansion (train-profile) | Combined |
|---|---:|---:|---:|
| N20 | 500 | 1,566 | 2,066 |
| N50 | 536 | 1,899 | 2,435 |
| N100 | 1,268 | 3,829 | 5,097 |

Comfortably above the 2,000/size ceiling needed for the learning curve's top point at every size.
Nested 500/1,000/2,000-per-size subsets (`data/processed/s7799_task5_curve_n{20,50,100}_{500,1000,2000}`,
same seed per size across all three per-size draws so 500⊂1,000⊂2,000) are built and the 9
training configs are staged. Learning-curve training and the stop-rule evaluation (C.2: stop once
frozen-panel decoded route-gap gain flattens between consecutive curve points) follow next.
