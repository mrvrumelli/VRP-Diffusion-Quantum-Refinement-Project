# Corrective execution — 2026-09-30

The user authorized execution of the [R1–R12 queue](autonomous_work_priority_tasks_2026-09-30.md),
including the bounded Track B diagnostics previously paused. Historical artifacts are retained.

All twelve corrective items now have an implementation, completed bounded experiment, or
explicit conditional stop decision. Completion of this queue does **not** mean the paper was
reproduced. The [results report](corrective_results_2026-09-30.md) contains the measured effects
and [archived evidence](evidence/corrective_20260930/README.md) supports recalculation.

| Task | Execution status | Evidence / remaining gate |
|---|---|---|
| R1 | Complete | Dated corrections in ten task reports and guide; [archived evidence](evidence/autonomous_review_20260930/README.md). |
| R2 | Contract complete; final-reference limitation recorded | Shared graph metrics, paired stratified bootstrap, audited 24/72-source development panels; 96 test sources reserved and unscored. Strengthening their references is conditional on a qualifying finalist. |
| R3 | Complete | Explicit `posterior_mixture_v2`, legacy compatibility, independent enumeration, integration tests and checkpoint rescoring. |
| R4 | Complete | Four paper checkpoints, three Task 5 pairs and all three Task 6 seeds rescored; corrected step curve and 72-graph development expansion complete. N50 route improvement is supported on development data; no independent promotion. |
| R5 | Complete | Nine nested source-count datasets; verified single-HGS split: 50,000 train / 289 validation / 291 test. New paper training enforces provenance. |
| R6 | Passed bounded sanity gate | One-source generated F1=1.0; exact oracle recovery at 1/50/1000 steps; fixed-noise train/validation metrics saved. |
| R7 | Bounded two-seed experiment complete | Versioned depot-relative input path and legacy loading; mixed validation effects, no promotion. |
| R8 | Complete; tested calibration rejected | Weighted/unweighted two-seed runs complete; unweighting improves F1 but worsens routes. Separate fitted logit-intercept calibration worsens both generation and routing in two sampling seeds. |
| R9 | Complete; no normalization replacement selected | Mixed-size order controls, four N100-only runs and 48 size/noise/mode probes completed. Single-size stability improves; route-gain direction is sampling-sensitive. Original checkpoint buffers remain unchanged. |
| R10 | Complete; expansion stopped | Both five-epoch N100 seeds worsen at 1,000 versus 500 sources: +9.22 and +23.48 gap points, paired intervals above zero. The [predeclared stop rule](corrective_curve_protocol_2026-09-30.md) preserves the untouched test. |
| R11 | Bounded comparison complete | Policy end-to-end timing, solver time matching, six-instance CVRPLIB expansion and all nine OOD cells completed. Fleet-cap defect corrected. Search caps and setup overhead are explicit; no identical-hardware deadline claim. |
| R12 | Complete; freeze deferred | [Decision manifest](evidence/corrective_20260930/candidate_baseline.json) records hashes, evidence classifications, unresolved paper assumptions and downstream gates. No independently confirmed new winner; no paper or quantum promotion. |


Confirmed defects: soft posterior composition, absent depot input, file-versus-source learning
curve counts, inconsistent gap definitions, and uncontrolled filtered/unfiltered panels.
Task 5 N20/1000, N50/500, N100/1000 and Task 6 N100 policy remain **candidates**.
The tested calibration recipe is **rejected**; broader objective, normalization and data effects
remain **hypotheses**. The bounded original Task 4 comparison and per-regime Task 3 results are
now reported. Quantum work remains gated.

The canonical primary routing comparison is mean instance-relative gap; ratio-of-total-cost gap
is also reported. Existing policy `gap_percent` retains its historical ratio definition.
Generated F1 is pooled positive-class F1 over ordered off-diagonal pairs at threshold 0.5;
per-size F1 and their equal-size mean are separate. Confidence intervals resample graphs within
size and recompute counts; sampling seeds are reported separately, not treated as extra graphs.

All corrective runs explicitly select `posterior_mixture_v2`. Existing CLI/function defaults
still select the legacy sampler for compatibility; new manual evaluations must pass
`--sampler posterior_mixture_v2` (or set the corresponding configuration field). Depot-relative
inputs are also opt-in and stored in checkpoints. New `paper_cmd` training now validates actual
single-HGS label provenance: old training configurations pointing at filtered audit labels must
be repointed to the repaired manifests before they can satisfy that claim.

The initial common panel uses existing validation labels and is explicitly **development data**.
It is outside the union of the registered denoiser and GAT training sources, but has been used
for historical selection. No score on it is called an untouched-test result. Hashes, source
inventories, per-instance probabilities/targets, confusion counts, costs, feasibility and runtime
are recorded by [the evaluation runner](../scripts/run_corrective_evaluation.py).

Initial verification: 81 sampler/prediction/training tests and 19 comparison/routing/policy tests
passed. Pytest's cache directory was unwritable; the tests themselves completed successfully.


Verification update: the final full suite passed **543 tests**, with Ruff and strict source mypy
clean. Four warnings concern third-party deprecations and Windows CPU-count detection. The CVRPLIB
expansion discovered an additional defect: the evaluator ignored the declared fleet count.
B-n51-k7 was reported below its reference with eight vehicles instead of seven. The corrected
`at_most_declared` run uses seven vehicles and matches 1,032; the invalid earlier table is retained
only as diagnostic evidence. Six official CVRPLIB instances were obtained through the browser
because direct download encountered an expired server certificate. One unavailable `.sol` file
uses the instance-comment reference, checked against the official index.

The four curve checkpoints preserve their pretrained frozen GAT tensors exactly. The final
archive also checks reserved-test separation against registered training pools, repaired
single-HGS splits, source-count subsets and the two historical policy test panels. No overlap
was found; no reserved test predictions were generated.

The next research step is to resolve the paper's metric, objective and input ambiguities from
author artifacts if available, then preregister one matched two-seed comparison on the repaired
single-reference splits. This is a remaining research hypothesis, not an automatic continuation
of the failed scaling branch. Reference strengthening and independent finalist testing remain
conditional; paper Tasks 11-14 and quantum Tasks 7-8 retain their prerequisites.
