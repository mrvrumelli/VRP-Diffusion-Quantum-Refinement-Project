# rc_full_eval OOD audit — completed 2026-09-25

**Correction, 2026-09-30 (R1):** Only N100 has higher acceptance than IID: N20 98.4% < 98.6%; N50 86.8% < 94.4%; N100 56.8% > 47.8%. Claims of higher acceptance at every size are superseded. See the [review](autonomous_work_review_2026-09-30.md) and [execution log](corrective_execution_2026-09-30.md). Historical run values below are retained.

Track A Task 1 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
finish the fresh R/C/RC out-of-distribution eval-batch audit (seeds 9901-9903) that had been
paused since 2026-08-26 at 18,219/36,000 candidates. This pool exists specifically to give the
`rcfull`-merged checkpoints a genuine held-out OOD test that the original `rc_full` audit pool
could not, since that one went into training.

## Frozen inputs

- Config: `configs/data/label_audit_rc_full_eval.yaml` (4 PyVRP seeds `[9911, 9912, 9913, 9914]`,
  10/20/40s budgets, OR-Tools challenger, `stable_sample_per_size: 50`, 11 workers).
- Source: 9,000 instances, 1,000 each in every `{R,C,RC} x {N20,N50,N100}` cell, disjoint from the
  original `rc_full` pool (seeds 8801-8803) that was folded into `rcfull_audit_policy_v1` training
  labels and from the `s7799` training corpus.

## Results

36,000/36,000 PyVRP runs (841,589 CPU-seconds) and 1,889/1,889 OR-Tools challenges (63,764
CPU-seconds), **0 solver errors**.

| Size | Instances | Cost-stable | Matrix-stable | Needs review |
|---|---:|---:|---:|---:|
| N20 | 3,000 | 2,999 (99.97%) | **2,952 (98.4%)** | 48 (1.6%) |
| N50 | 3,000 | 2,782 (92.7%) | **2,604 (86.8%)** | 396 (13.2%) |
| N100 | 3,000 | 2,209 (73.6%) | **1,705 (56.8%)** | 1,295 (43.2%) |
| **Total** | **9,000** | **7,990** | **7,261** | **1,739** |

Artifacts: `outputs/label_audit/rc_full_eval/{metrics.json,summary.csv,accepted_matrix_examples/,candidates/}`.

## Interpretation

Acceptance rates follow the same size-ordering every audit in this project has shown (N20 near-perfect,
N100 hardest), and are actually *higher* than the original 4-seed `s7799_strong_reference` audit's
rates at every size (98.4%/86.8%/56.8% here vs. 98.6%/94.4%/47.8% there for N20/N50, and notably
better at N100: 56.8% vs. 47.8%) — plausibly because R/C/RC spatial-stress instances have more
structured (less ambiguous) optimal partitions than uniform-random IID instances at the same size,
though this was not tested directly and shouldn't be over-interpreted from one comparison.

## Next: materialize as the OOD panel for Task 3

This is the real arbiter for whether the `rcfull`-merged training experiment
(`diffusion_denoiser_s7799_rcfull_persize_n{20,50,100}_cuda`) is a genuine win on R/C/RC data
despite its earlier regression on the s7799 IID panel, or just a strictly worse model — see
[[project_status_2026-08]] memory and `docs/archive/plans/3060ti_training_todo.md`. **Do not fold
this into training** — that would defeat the entire purpose of auditing it with fresh seeds.
