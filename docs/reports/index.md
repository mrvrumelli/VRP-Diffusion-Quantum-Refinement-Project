# docs/ file index — audited 2026-09-24, organized 2026-09-24

`docs/` accumulated 23 markdown files across roughly six weeks of autonomous work sessions
(2026-08-14 through 2026-09-24) with no navigation aid and no marked status, so old and current
documents read as equally authoritative. This index audits every file, states what's actually
current vs. superseded vs. stale, and gives a reading order. The 10 historical/outdated files this
audit identified have been moved into `docs/archive/{plans,session_reports,evidence}/` (via `git
mv`, history preserved) so `docs/`'s top level now holds only current-status documents; every
cross-reference between docs was updated to the new paths.

## Read these first (current source of truth, live in `docs/`)

| File | What it is |
|---|---|
| [`project_findings_2026-09-24.md`](../project_findings_2026-09-24.md) | Current status + ordered action plan for the `ours_robust` track. Start here. |
| [`cmd_paper_alignment_plan.md`](../cmd_paper_alignment_plan.md) | Current status + 9-phase plan for the `paper_cmd` faithful-reproduction track. |
| [`cmd_paper_comparison_contract.md`](../cmd_paper_comparison_contract.md) | The two tracks' Phase 0 operating contract (what's frozen, what's still an open author question). |
| [`autonomous_long_compute_guide_2026-09-24.md`](../autonomous_long_compute_guide_2026-09-24.md) | The executable long-compute task queue for both tracks — commands, gates, completion signals. |
| [`stochastic_reference_probe.md`](../stochastic_reference_probe.md) | **The authoritative evidence trail for the currently active diffusion recipe** (three per-size stochastic-reference denoisers). Long, but every current default traces back to a specific table here. |
| [`failure_analysis.md`](../failure_analysis.md) | Current representative failure modes (long-horizon over-split, vehicle inflation) for the active recipe. |
| [`or_baseline_comparison.md`](../or_baseline_comparison.md) | Current fair OR-Tools/PyVRP matched-budget comparison (smoke scale, explicitly labeled as such). |
| [`policy_batch_profile_2026-09-24.md`](../policy_batch_profile_2026-09-24.md) | Current finding: batching gives ~6x policy-inference throughput; baseline stays batch-1 for comparability. |
| [`rc_full_artifact_reconstruction_2026-09-24.md`](../rc_full_artifact_reconstruction_2026-09-24.md) | The true, current provenance state of the `rc_full`/`rc_full_eval` audits — supersedes any earlier claim about what those directories contain. |
| [`label_audit_runbook.md`](../label_audit_runbook.md) | Still-accurate operational how-to for running/resuming a strong-label audit. |
| [`coding_standards.md`](../coding_standards.md) | Evergreen project coding standard — not time-sensitive, always current. |

## `docs/archive/evidence/` — foundational, keep, but read as history not current status

Cited by the files above for hashes, seeds, and provenance. Their *narrative* conclusions have
mostly been superseded by later work (noted per file); their *evidence* (hashes, counts,
per-instance numbers) remains the authoritative record and should not be recomputed casually.

| File | Superseded by / current relevance |
|---|---|
| [`label_audit_s7799_decision.md`](../archive/evidence/label_audit_s7799_decision.md) | Foundational 1,500-instance audit decision (policy-v2). Its own recommended label policy was later superseded by stochastic-reference targets (see `stochastic_reference_probe.md`) — keep for the audit numbers, not for "what to train." |
| [`route_partition_ambiguity_plan.md`](../archive/plans/route_partition_ambiguity_plan.md) | Filed under `archive/plans/` (see below) rather than here — cross-listed because it's also foundational evidence of methodology. |
| [`reverse_sampling_diagnosis.md`](../archive/evidence/reverse_sampling_diagnosis.md) | Diagnostic that picked exact-stochastic sampling over deterministic/stride-7 alternatives. Conclusion still holds for the frozen diffusion-only recipe; superseded only in the sense that the paper_cmd track's `skipped_posterior` sampler (merged 2026-09-24) is now a separate, additional option, not a replacement. |
| [`spatial_stress_validation.md`](../archive/evidence/spatial_stress_validation.md) | Foundational R/C/RC generation validation. Hashes and metrics are unchanged and still accurate. |

## `docs/archive/session_reports/` — dated work-session changelogs

| File | Superseded by / current relevance |
|---|---|
| [`autonomous_work_report_2026-08-28.md`](../archive/session_reports/autonomous_work_report_2026-08-28.md) | Session changelog for the dual-pointer policy's first real runs. Numbers here were later refined by the K=1 methodology fix — see `[[dual_pointer_policy_branch]]` memory / `project_findings_2026-09-24.md` for the current N20/N50/N100 K=1 gaps. |
| [`autonomous_work_report_2026-08-16.md`](../archive/session_reports/autonomous_work_report_2026-08-16.md) | Session changelog. Its headline finding (per-size models beat pooled) is real and still active, but is now stated more authoritatively in `stochastic_reference_probe.md`'s "Per-size specialized models" section — read that instead for the numbers. |
| [`autonomous_work_report_2026-08-15.md`](../archive/session_reports/autonomous_work_report_2026-08-15.md) | Session changelog. Its "recommended next actions" are all completed and superseded by later sessions. |
| [`3060ti_training_report.md`](../archive/session_reports/3060ti_training_report.md) | Reports the single pooled policy-v2 model as "the selected model." **That model is no longer the active recipe** — three per-size models beat it by ~37-41% (see `stochastic_reference_probe.md`). Keep only for the CUDA/reliability-gate evidence (memory, resume, CUDA test results), not for "which model is selected." |

## `docs/archive/plans/` — outdated checklists and a completed research plan

All three files below are pre-execution planning documents that no longer reflect reality: work
that actually happened is not always checked off, and some checked/unchecked items describe a plan
that was later executed differently (e.g. the policy was eventually trained with the dual-pointer
REINFORCE approach against existing datasets, not the "100k→1-5M generated rollout episodes" scheme
originally planned). Treat all three as historical drafts superseded by
`project_findings_2026-09-24.md` and `cmd_paper_alignment_plan.md`.

| File | Why it's stale |
|---|---|
| [`3060ti_training_todo.md`](../archive/plans/3060ti_training_todo.md) | Its own header admits the original 1k/5k/10k/30k roadmap was superseded on 2026-08-15, and its "Next-stage research plan" is now fully executed (see `stochastic_reference_probe.md`). The bulk of the file (sections 1-8, the original per-item checklist) describes a sequence largely completed via different concrete steps than listed. Only the 2026-09-24 status-correction banner at the top is current. |
| [`data_readiness_todo.md`](../archive/plans/data_readiness_todo.md) | "Current inventory" and "Immediate fixes" are genuinely done (correctly checked). Everything from "RTX 3060 Ti / CUDA migration" onward mixes real gaps (Phase 2's supervised-predictor learning curve genuinely was never run beyond a sanity check) with items completed through a different path than described (Phase 4's rollout-episode counts were superseded by the dual-pointer policy's actual training approach) and items already done elsewhere but not reflected here (R/C/RC labeling, per-size training). Don't trust any individual checkbox without cross-checking `project_findings_2026-09-24.md`. |
| [`route_partition_ambiguity_plan.md`](../archive/plans/route_partition_ambiguity_plan.md) | The research plan whose 8-step execution order is now **fully executed** — every workstream item is done and documented in `stochastic_reference_probe.md`. Keep for methodology/decision-gate principles (e.g. "never select using the untouched test set"), not as an open plan. |

## Needs a rerun before citing again (still live in `docs/`, not archived)

This one stays in `docs/`'s top level rather than the archive — its content isn't superseded, it's
just stale evidence inside an otherwise-current report shape, and the fix is a rerun, not a
re-read.

| File | Why |
|---|---|
| [`constraint_matrix_diffusion_ablation_report.md`](../constraint_matrix_diffusion_ablation_report.md) | Says so itself: "a CPU-friendly smoke report... the diffusion arm trains a tiny linear-encoder denoiser from scratch... because no full trained diffusion checkpoint is present in this checkout." That's no longer true — the per-size champions now exist. The report's `diffusion_m` row (matrix F1 **0.0000**) reflects an untrained placeholder, not the real model, and should not be quoted as evidence about the actual diffusion recipe until rerun against a real checkpoint. |

## Not a report at all

`coding_standards.md` is listed above under "read first" because it's a living standard, not a
point-in-time report — it has no supersession status, it's just current project policy.
