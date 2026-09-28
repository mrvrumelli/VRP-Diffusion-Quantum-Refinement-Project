# paper_cmd full-scale label audit — completed 2026-09-28

Track B Task 9, step 3 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
the full-scale labeling campaign, launched after the pilot
([`task9_paper_cmd_pilot_audit_2026-09-24.md`](task9_paper_cmd_pilot_audit_2026-09-24.md)) proved
the pipeline and Task 10's pilot-scale training
([`task10_paper_cmd_diffusion_pilot_2026-09-25.md`](task10_paper_cmd_diffusion_pilot_2026-09-25.md))
showed a clean, undersized-data negative result (F1 0.505 vs. the paper's ≈0.823 target) that
justified committing to it.

## Frozen inputs

- Source: 50,580 instances (16,870/16,840/16,870 by size), a fresh draw from the s7799 train
  split, verified zero overlap against both `s7799_strong_reference` and the pilot pool before
  labeling.
- Config: `configs/data/label_audit_paper_cmd_full.yaml` — 2 PyVRP seeds `[50937, 50938]` (the
  tooling floor, not the paper's theoretical 1-seed protocol), 10/20/40s budgets, OR-Tools
  challenger, 11 workers.

## Results

**101,160/101,160 PyVRP runs** (2,365,695 CPU-seconds) and **13,415/13,415 OR-Tools challenges**
(468,171 CPU-seconds), **0 solver errors**. Wall time: ~72 hours (2026-09-25 ~03:16 to
2026-09-28), close to the ~59-hour estimate given the OR-Tools phase added real additional time
beyond the PyVRP-only projection.

| Size | Instances | Cost-stable | Matrix-stable | Needs review |
|---|---:|---:|---:|---:|
| N20 | 16,870 | 16,869 (99.99%) | **16,756 (99.3%)** | 114 (0.7%) |
| N50 | 16,840 | 14,303 (84.9%) | **13,555 (80.5%)** | 3,285 (19.5%) |
| N100 | 16,870 | 8,394 (49.8%) | **6,854 (40.6%)** | 10,016 (59.4%) |
| **Total** | **50,580** | **39,566** | **37,165** | **13,415** |

Artifacts: `outputs/label_audit/paper_cmd_full_50k/{metrics.json,summary.csv,accepted_matrix_examples/,candidates/}`.

## Interpretation

Acceptance rates track the pilot's almost exactly (pilot: 99.6%/80.6%/38.6%; full scale:
99.3%/80.5%/40.6%) — confirming the pilot's smaller sample was already a reliable estimate of the
full-scale yield, not a fluke. **37,165 usable matrix-stable examples**, about 74% of the paper's
50,000-instance nominal target and over 10x the pilot's 3,606 — this is the real test of whether
Task 10's F1 0.505 shortfall was genuinely a data-scale problem.

## Next: Task 10 at full scale

Split `accepted_matrix_examples/` into train/val (following the same pattern as the pilot), rerun
GAT pretrain then the paper-config diffusion training, and check the F1≈0.823-at-50-steps gate for
real this time.
