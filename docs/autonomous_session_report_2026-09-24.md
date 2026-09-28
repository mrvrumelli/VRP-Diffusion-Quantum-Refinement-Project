# Autonomous work session report — 2026-09-24 onward

Full account of an ongoing autonomous session working through
[`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)'s
two-track long-compute queue: Track A (`ours_robust`, this project's own dual-pointer policy
extensions) and Track B (`paper_cmd`, a faithful reproduction of the paper this project is based
on). This report covers everything from session start through the current moment; individual
tasks each also have their own detailed report, linked below rather than repeated here.

## Starting point

The session began by auditing `docs/` (23 markdown files, no navigation, mixed current/stale
status), organizing it into `docs/archive/{plans,session_reports,evidence}/` with a curated index
at `docs/reports/index.md`, then merging the `feat/paper-cmd-reproduction` branch (Track B's
scaffolding) onto the working branch and building the long-compute guide that everything since has
followed.

## Track A (`ours_robust`) — completed work

### Task 1 — Finish the paused `rc_full_eval` OOD audit

A fresh out-of-distribution audit (R/C/RC spatial-stress instances, seeds 9901-9903) had been
paused since 2026-08-26 at 18,219/36,000 candidates. Resumed and completed cleanly: **36,000/36,000
PyVRP + 1,889/1,889 OR-Tools runs, 0 errors**, yielding 7,261 matrix-stable examples out of 9,000
audited. Full detail: [`task1_rc_full_eval_audit_2026-09-25.md`](task1_rc_full_eval_audit_2026-09-25.md).

### Task 2 — Large-panel policy-vs-diffusion evaluation

Re-confirmed the dual-pointer policy's K=1 gap against the frozen diffusion-only champions on the
full 100/92/45-example panel (larger than the slices used for the original comparison):

| Size | Policy K=1 gap | Diffusion-only champion | Delta |
|---|---:|---:|---:|
| N20 | 17.86% | 21.11% | **-3.25pp** |
| N50 | 27.81% | 29.39% | **-1.58pp** |
| N100 | 31.80% | 30.57% | +1.23pp (behind) |

N20/N50 wins confirmed at scale; N100 confirmed behind — this result is what motivated Task 6.
Full detail: [`task2_large_panel_eval_2026-09-24.md`](task2_large_panel_eval_2026-09-24.md).

### Task 6 — N100 hyperparameter sweep: gap closed

Systematically tested four hypotheses against the N100 base config, isolating one variable per
run:

| Variant | Change | K=1 gap | Verdict |
|---|---|---:|---|
| 1 | `entropy_weight` 0→0.01 | 32.44% | negative |
| 2 | `learning_rate` 1e-4→5e-5 | 34.72% | negative (worse than variant 1) |
| 3 | `num_starts` 8→16 | **29.80%** (3-seed mean) | **confirmed win** |

Variant 3 was replicated on 3 independent seeds (42: 30.18%, 4332: 28.74%, 4333: 30.47%) before
being trusted — this project has direct precedent for single-seed wins reversing on replication,
so this was not skipped. **Result: all three sizes now beat their diffusion-only counterparts**,
not just N20/N50. Full detail:
[`task6_n100_hyperparameter_sweep_2026-09-24.md`](task6_n100_hyperparameter_sweep_2026-09-24.md).

### Task 3 — rcfull-merged vs. champion, IID vs. OOD (in progress)

The real arbiter for whether merging the R/C/RC spatial-stress data into training
(`rcfull`-merged checkpoints) is a genuine win on that harder data despite regressing on the
original IID panel, or just a strictly worse model. Using Task 1's fresh OOD panel (never seen in
training) against both the champions and the rcfull-merged checkpoints:

| Size | Model | IID gap | OOD gap |
|---|---|---:|---:|
| N20 | Champion | 21.11% | 17.89% |
| N20 | rcfull-merged | 36.38% | 35.81% |
| N50 | Champion | 29.39% | 25.77% |
| N50 | rcfull-merged | 37.35% | *(running)* |

**Emerging verdict**: at both sizes checked so far, rcfull-merged is worse on *both* panels, not a
genuine OOD-specific win — the "strictly worse model" outcome, not "real tradeoff." N100 still to
run. Also notable: the champions generalize *better* to OOD than to their own IID panel at every
size checked — unexpected, not yet explained, worth a closer look once the full table exists.

## Track B (`paper_cmd`) — completed and in-progress work

### Phase 4 code (dual encoder-decoder architecture)

Confirmed complete and tested (`build_policy_from_config` no longer raises `NotImplementedError`
for `paper_cmd`; `tests/test_paper_cmd_policy.py` covers checkpoint round-trip, frozen-gradient,
tiny-RL-training, and ablation-mode gates) — this had already landed via concurrent work before
this session's compute queue started, verified rather than assumed.

### Task 9 pilot — labeling test before committing to 50,000 instances

Rather than committing blind to the paper's full 50,000-instance labeling campaign, labeled a
4,949-instance pilot first (2 PyVRP seeds — the closest reachable approximation to the paper's
1-seed protocol, since the tooling has a hard floor of 2). **Completed: 9,898/9,898 PyVRP +
1,343/1,343 OR-Tools, 0 errors**, yielding 3,606 usable matrix-stable examples. Full detail:
[`task9_paper_cmd_pilot_audit_2026-09-24.md`](task9_paper_cmd_pilot_audit_2026-09-24.md).

### Task 10 pilot — the actual gate check, result: not met

Trained the exact paper-config GAT + diffusion denoiser (T=1000, BatchNorm, frozen GAT, the real
architecture, not a shortcut) on the pilot's 2,886 training examples. **Best sample F1: 0.505**
against the paper's ≈0.823 target — not a borderline miss, about 60% of target. The failure
pattern (train loss kept improving while F1 peaked at epoch 1 and decayed) is the signature of
overfitting an undersized pool, not an implementation bug — Phase 3's engineering gates had
already passed 81+507 tests independently. Full detail:
[`task10_paper_cmd_diffusion_pilot_2026-09-25.md`](task10_paper_cmd_diffusion_pilot_2026-09-25.md).

### Task 9 full-scale — currently running

The pilot's clean negative result justified the full campaign. Drew an oversized 52,500-instance
pool, checked and pruned overlap against both already-audited pools (1,920 removed), reverified
zero overlap, and launched the same 2-seed protocol at scale on the resulting 50,580 instances
(16,870/16,840/16,870 by size). **Currently ~80% through** (see Live Status below). Once complete,
Task 10 reruns on the full data and gets a real shot at the paper's target.

## Bugs found and fixed along the way

Real code issues, distinct from routine training/eval work:

1. **mypy regression**: `constraint_denoiser.py`'s new BatchNorm support
   (`_normalize_features`) returned `Any` from a `Tensor`-typed function. Fixed with explicit
   `cast(Tensor, ...)`, matching this file's existing pattern.
2. **Hardcoded seed-count check**: `scripts/run_strong_label_audit.py` required exactly 4 PyVRP
   seeds, blocking the paper's cheaper 2-seed protocol even though the underlying policy only
   requires `>= minimum_near_best_seeds` (2). Relaxed the check; verified backward-compatible with
   every existing 4-seed config (507/507 tests still pass).
3. **Regime-prefixed filename bug**: `IndexedJSONDataset`/`load_examples_by_size` filtered by a
   glob requiring the size token as a literal filename *prefix* (`cvrp{size}_*.json`), so it
   silently found zero examples in any R/C/RC pooled directory, where files are prefixed by
   regime (`c_cvrp100_0000.json`) to avoid collisions. This affects every caller of that function
   against a regime-prefixed directory, not just this session's own scripts. Fixed the glob to
   match the token anywhere (`*cvrp{size}_*.json`), added a regression test.

## Incidents: 3x exit-127 failures under concurrent CPU+GPU load

The guide's own "CPU-only and GPU-only jobs may run concurrently" exception held up in steady
state but proved to have a real, reproducible failure mode: a GPU job starting (or a fresh
subprocess launching) while a CPU audit's workers are actively churning has a genuine chance of
dying immediately with a bare `exit 127` and no traceback — confirmed 3 times, at 3 different
points in the task sequence (never the same spot twice), ruling out a fixed bug location.
Diagnosed via direct GPU-utilization checks each time (0% right after a failure vs. real
utilization on immediate retry). Mitigation adopted: retry the failed step once before treating it
as a real bug, and — since re-running a whole multi-leg script wastes already-succeeded legs — run
remaining legs individually rather than restarting the batch script.

## Live status (at time of writing)

- **Task 9 full-scale audit (CPU)**: **80,526/101,160 candidates (~80%)**, ~49h19m elapsed.
  Progress accelerated substantially in the last stretch (39.6%→80% in about 13.5 hours),
  consistent with having moved past the slowest chunk (N100 at the 40s-per-solve budget).
- **Task 3 (GPU)**: N20 fully compared (rcfull-merged worse on both panels). N50: champion legs
  done, rcfull-merged IID leg done (37.35%, worse than champion as expected), rcfull-merged OOD
  leg currently running — the one remaining number needed to close out N50. N100 (4 legs) not yet
  started.

## What's left

**Track A**: finish Task 3's remaining N50/N100 legs and write its final verdict; Tasks 4
(matched-budget + CVRPLIB comparison), 5 (expand IID audit + learning curve — the largest
remaining Track A item), 7-8 (quantum refinement, gated behind everything) not started.

**Track B**: once the full-scale audit finishes, rerun Task 10 (the real gate check at full
scale); if it clears the F1≈0.823 gate, Task 11 (full Phase 5 baseline training — the plan's own
estimate is **1-3 elapsed weeks**) becomes the next task, followed by Task 12 (evaluation suite,
including all 10,000 XML100 instances), Task 13 (ablations, **1-2 weeks**), and Task 14
(robustness study, **4-7 days**). The paper-alignment plan's own total estimate for full
reproduction was **5-9 elapsed weeks** — this session's progress is real and substantial, but is
early within that scope, not close to the end of it.
