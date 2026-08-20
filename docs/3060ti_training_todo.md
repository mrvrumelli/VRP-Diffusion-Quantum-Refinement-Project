# RTX 3060 Ti training checklist

> **Completed 2026-08-15.** The original 1k/5k/10k/30k roadmap below was written before the
> strong-label audit established that only 500 audited training sources per size are available.
> The executed evidence-based curve was 100/250/500 per size. See
> `docs/3060ti_training_report.md` for the authoritative configs, hashes, metrics, final checkpoint,
> untouched-test result, and limitations. Remaining unchecked historical items are not prerequisites
> for the completed bounded policy-v2 run unless a future larger corpus is generated.
>
> The 2026-08-15 follow-up implementation and probe summary is in
> [`autonomous_work_report_2026-08-15.md`](autonomous_work_report_2026-08-15.md).

## Completed autonomous workflow

- [x] Preserve original train/validation/test files and create only versioned derived datasets.
- [x] Complete and validate the 1,500-instance strong-label audit with per-size analysis.
- [x] Run bounded CVRP100 80- and 120-second follow-ups; reject broad longer-time relabeling.
- [x] Materialize and hash policy-v2 labels: original CVRP20, strongest audited CVRP50/100.
- [x] Freeze independently audited validation and untouched test subsets.
- [x] Pass CUDA preflight, CUDA tests, end-to-end smoke, VRAM, numerical-stability, and resume gates.
- [x] Compare original, canonical, mixed/multi-reference, and compute-matched label policies.
- [x] Compare timestep sampling, learning rate, and positive-class weighting with fixed validation.
- [x] Replicate the winning uniform-timestep recipe across three seeds.
- [x] Run the honest 100/250/500-per-size learning curve and select the full audited pool.
- [x] Train to an early-stopped 15-epoch ceiling and retain the validation-selected checkpoint.
- [x] Run exact 700-step full-chain validation and one-time untouched-test evaluation per size.
- [x] Save environment, driver, configs, manifests, logs, checkpoints, metrics, and decision report.

## Next-stage research plan

The exact full-chain result, rather than noisy-time AUC, determines the next work. Follow
[`route_partition_ambiguity_plan.md`](route_partition_ambiguity_plan.md) in this order:

- [x] Implement capacity-aware matrix-to-route decoding plus feasibility, cost-gap, repair,
  route-size, vehicle-count, runtime, and matrix-after-decoding evaluation. Classical and quantum
  refinement runtimes remain pending until those refiners exist.
- [x] Diagnose reverse-process degradation on stable CVRP20 with fixed comparable samplers; exact
  stochastic sampling outperformed deterministic transitions and approximate stride-7 sampling.
- [x] Add a fixed, hashed full-chain validation panel and a portable configuration that selects
  checkpoints by exact decoded route-cost gap while retaining exact sample F1.
- [x] Add canonical plus frozen oracle best-of-K matrix evaluation with one declared F1 matching
  criterion and a single selected reference for every reported oracle metric.
- [x] Run a one-reference-per-source-per-epoch stochastic probe with 1,500 unique sources and 94
  matched optimizer steps per epoch; exact fixed-panel F1 improved from 0.4254 to 0.5498 and the
  decoded gap fell from 76.59% to 43.37%.
- [x] Add versioned training-only consensus targets and pair-confidence masks without weakening
  the binary `CVRPExample.constraint_matrix` invariant; hard stochastic references still supply
  valid forward diffusion states while consensus/masks affect only the clean-target loss.
- [x] Compare policy-v2, exclusion, stochastic, consensus, and masked arms under identical compute.
  A step-matched policy-v2 baseline, 3-seed stochastic replication, step-matched exclusion, and
  consensus/masked-consensus were all scored on the same 120-example large panel. Ranked by overall
  decoded gap: exclusion (42.20%) < stochastic (44.48% mean) < consensus (52.76%) <
  masked-consensus (53.11%) < baseline (56.39%). Consensus and masked-consensus are ruled out. See
  [`stochastic_reference_probe.md`](stochastic_reference_probe.md).
- [x] Require a CVRP100-specific gain without material CVRP20/CVRP50 regression. CVRP100/CVRP50
  gains are large and consistent across every seed for both exclusion and stochastic. CVRP20 shows
  a small, real regression (+2.7 to +3.6 pp mean, seed-dependent) that is an order of magnitude
  smaller than the gains elsewhere. Exclusion multi-seed replication (3 seeds, matching stochastic)
  additionally found exclusion has ~7x the seed-to-seed variance of stochastic (19.5 pp range vs.
  2.75 pp) — one exclusion seed lands worse than the baseline. Stochastic reference is recommended
  as the more reliable arm on this evidence; whether the remaining CVRP20 regression counts as
  "material" is an open call for the research owner. See
  [`stochastic_reference_probe.md`](stochastic_reference_probe.md).
- [x] **Superseded 2026-08-16.** Three independent per-size models beat the single frozen pooled
  model on both the large validation panel (46.03% -> 27.00% mean gap, ~41% relative) and the
  untouched test set (45.73% -> 28.63% mean gap, ~37% relative), confirming the CVRP20 regression
  was multi-task interference, fully recoverable by not sharing the denoiser across sizes. N20/N50
  gain enormously (-42pp / -12pp on test); N100 is noise-level indifferent to isolation (+2.5pp on
  the tiny 20-example test slice, -1.8pp on the larger panel). New recommended default: three
  per-size models (`diffusion_denoiser_s7799_stochastic_persize_n20/n50/n100_cuda`), shared frozen
  GAT encoder. See [`stochastic_reference_probe.md`](stochastic_reference_probe.md) "Per-size
  specialized models".
- [x] (Superseded, kept for history) Freeze the sampler, target, checkpoint, and decoder before one-time test evaluation. Frozen:
  stochastic-reference targets, uniform-timestep sampling, exact-stochastic reverse sampling,
  decoded-route-cost-gap checkpoint selection, capacity-aware decoder. Checkpoint
  `diffusion_denoiser_s7799_stochastic_probe_cuda_20260815T103109616491Z` (seed 4331, epoch 0,
  chosen because it predates the multi-seed comparison and so cannot be a cherry-picked seed). The
  CVRP20 regression was judged non-blocking (an order of magnitude smaller than the CVRP50/CVRP100
  gains); see the freeze-decision reasoning in
  [`stochastic_reference_probe.md`](stochastic_reference_probe.md).
- [x] One-time untouched-test evaluation. Frozen model on the 60-example test set: F1 0.5447
  (was 0.4350), decoded gap 45.73% overall (was 72.13%), with every per-size gap improved including
  CVRP20 (63.54% vs. 70.78%) relative to the actual prior production checkpoint. Results are within
  ~2pp of the large validation panel, so the recipe generalizes rather than overfitting to the
  selection panel. See [`stochastic_reference_probe.md`](stochastic_reference_probe.md).
- [ ] Generate additional strong labels only after a verified model-side gain — now unblocked by a
  verified gain, but not started; scope and cost (9,000 R/C/RC instances need audit config authoring
  and are ~6x the previous 1,500-instance audit's solver load) should be estimated before committing
  CPU time.
  - Traced the exact pipeline (2026-08-16): `generate_cvrp.load_dataset(data/raw/cvrp/spatial_stress_{r,c,rc}_s{8801,8802,8803}/cvrp{size})`
    → `solve_cvrp.solve_dataset(...)` (quick placeholder labels, same as original s7799 corpus) →
    `solve_cvrp.save_labels(...)` → `export_examples.export_run(run_dir)` (writes `CVRPExample`
    JSON) → `label_audit.run_reference_label_audit(source, output, policy=..., expected_counts_by_size=...)`
    (the strong multi-seed PyVRP/OR-Tools audit; same function `tests/test_label_audit.py` exercises).
  - **Do not run `solve_dataset`/`save_labels` against the raw `data/raw/cvrp/spatial_stress_*`
    directories directly** — those are the exact directories `spatial_stress_validation.md` hashed
    and verified as zero-overlap with training; writing labels there risks invalidating that
    evidence. Copy the instances (or just the CSVs) to a scratch/output directory first.
  - **Launched then paused, 2026-08-16 — not currently running.** A 60-instance pilot (20/regime at
    size 20, config `configs/data/label_audit_rc_pilot.yaml`) completed cleanly in 4m02s wall-clock
    (240 PyVRP runs + 11 OR-Tools challenges, zero errors, 59/60 accepted), giving real timing to
    extrapolate from: ~21-22 hours wall-clock for the full 9,000-instance audit on this 12-core
    machine at full ~11-worker parallelism (CPU-only, does not touch the GPU). With explicit user
    sign-off on that cost, the full audit was launched as a detached background chain
    (`outputs/logs/run_rc_full_audit_chain.sh`: prep, ~50 min, then the real audit,
    `configs/data/label_audit_rc_full.yaml`, same policy as the original 1,500-instance training
    audit — 4 PyVRP seeds, 10/20/40s per-size time budgets).
  - **Paused partway into prep** after the user asked whether their CPU running at its 95°C Ryzen
    5600 thermal ceiling (already reached with just the prep stage's 3 workers) was safe to run
    unattended for the full ~21-22 hours. Assessment: not dangerous — Ryzen 5000-series is designed
    to run to Tjmax under sustained boost with hardware-enforced throttling — but sustained
    full-load overnight was more thermal commitment than to make unilaterally, so all audit
    processes were killed (`taskkill`) as a precaution. The user chose to do GPU work instead; the
    audit was not resumed. **Current on-disk state**: `outputs/label_audit_full/{r,c,rc}/` contain
    only the copied raw CSVs from the killed prep run, no labels or examples yet — this partial
    state is not usable and must be deleted before restarting (`rm -rf outputs/label_audit_full`
    then rerun `outputs/logs/run_rc_full_audit_chain.sh`; prep itself is not resumable, only the
    audit stage after it is, via its candidate cache). Source raw data
    (`data/raw/cvrp/spatial_stress_*`) was never written to and is unaffected.
  - **Resumed 2026-08-16 20:25 with reduced parallelism.** User chose to cap the audit stage at 8
    workers (`workers: 8` in `configs/data/label_audit_rc_full.yaml`, down from the default 11) to
    lower sustained thermal load, accepting a longer wall-clock estimate (~30 hours instead of
    ~21-22) in exchange. Prep stage completed cleanly (9,000/9,000 examples, 100% feasible,
    ~52 minutes). The audit stage started at 21:16:51 and was progressing normally (783/36,000
    PyVRP runs after ~65 minutes) — files are processed alphabetically, so it was still working
    through the slowest size (CVRP100, 40s budget) first; the blended rate should pick up once it
    reaches CVRP20/50.
  - **Stopped by user request 2026-08-16 ~22:3x, not an error.** All 8 worker processes killed
    cleanly (`taskkill`). 218 candidate files were already saved to
    `outputs/label_audit/rc_full/candidates/` before stopping — this is the resumable cache.
    **To resume**: rerun just the audit stage (prep is already complete, do not rerun the full chain
    script or delete `outputs/label_audit_full/`):
    ```
    ./.venv/Scripts/python.exe scripts/run_strong_label_audit.py --config configs/data/label_audit_rc_full.yaml
    ```
    This picks up from the 218 cached candidates automatically rather than re-solving them (verified
    resumable behavior — see `tests/test_label_audit.py`). Adjust `workers:` in the config first if a
    different parallelism is wanted next time — but note the resume check requires an exact config
    match against the cached `config.json`, so changing `workers:` (or anything else) against the
    same `output_dir` raises `ValueError: existing audit config differs`; point `output_dir` at a
    new location instead for a different-config run.
  - **12-worker thermal test, 2026-08-16 — result: safe, cooler than expected.** At user request,
    to observe sustained CPU temperature at maximum parallelism (all 12 logical cores). Used a fresh
    `output_dir: outputs/label_audit/rc_full_12w` so the real 8-worker run's 218 cached candidates
    under `outputs/label_audit/rc_full/` stayed untouched. **Result: temperatures did not exceed
    70°C** at full 12-worker load — well below the earlier ~95°C seen during the 3-worker *prep*
    stage. This is consistent with AMD Ryzen's per-core boost behavior: fewer active cores get
    boosted to much higher individual clocks/voltage (and heat) than when all cores share the load
    under all-core boost, so full parallelism can run cooler than partial parallelism for this chip.
    Practical takeaway: this machine's cooling comfortably handles sustained full-CPU-load work: the
    earlier caution was reasonable to raise but the actual risk was low. The 12-worker test run was
    stopped after confirming this (its `rc_full_12w` output can be deleted or ignored — it holds no
    unique progress; the real progress is in `rc_full`'s 218 cached candidates).
  - **Open decision**: given confirmed thermal safety, resuming the real `rc_full` run (218 cached
    candidates) at `workers: 12` or `workers: 11` (auto-default) instead of 8 would finish faster
    (~21-22hr instead of ~30hr) with no demonstrated downside. Resuming at `workers: 8` still works
    too via the exact command above; just edit `workers:` in
    `configs/data/label_audit_rc_full.yaml` back to `outputs/label_audit/rc_full` as `output_dir`
    first if it was left pointed at `rc_full_12w`.

This is the operational checklist for the Windows 11 / RTX 3060 Ti machine. The first
1k-per-size run is a pipeline and throughput pilot, not final model training.

## 1. Environment

- [x] Install Python 3.12 and create `.venv`.
- [x] Activate the environment in PowerShell:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

- [x] Install the project and development dependencies:

  ```powershell
  python -m pip install -e ".[dev]"
  ```

- [x] Replace the CPU-only Torch package with the CUDA 12.8 build:

  ```powershell
  python -m pip uninstall -y torch
  python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  ```

- [x] Confirm RTX 3060 Ti, CUDA 12.8, cuDNN, and a successful backward pass:

  ```powershell
  python scripts/check_cuda.py --require-cuda --output outputs/cuda_environment.json
  ```

- [ ] Save the resolved environment and machine information with the experiment artifacts:

  ```powershell
  python -m pip freeze | Out-File -Encoding utf8 outputs\pip_freeze_3060ti.txt
  nvidia-smi | Out-File -Encoding utf8 outputs\nvidia_smi_3060ti.txt
  ```

Do not run `python -m pip install -e ".[dev]"` again after pinning CUDA Torch unless the
resolved Torch build is checked afterward. The unconstrained `torch>=2.3` dependency may select a
newer CPU-only release.

## 2. Code and CUDA smoke checks

- [ ] Run the CUDA-marked tests:

  ```powershell
  pytest -m cuda
  ```

- [ ] Run the regular quality checks before a long or final run:

  ```powershell
  ruff check .
  ruff format --check .
  mypy src
  pytest
  ```

- [ ] Confirm checkpoint save/load on CUDA and CPU, then test resume after an intentional short
  interruption.

## 3. Dataset prerequisites

- [ ] Copy the gitignored corpus beside the repository README at
  `cvrp_s7799_n20-50-100_x66667/`.
- [ ] Confirm the required split directories exist:

  ```powershell
  Test-Path .\cvrp_s7799_n20-50-100_x66667\splits\train
  Test-Path .\cvrp_s7799_n20-50-100_x66667\splits\val
  Test-Path .\cvrp_s7799_n20-50-100_x66667\splits\test
  ```

- [ ] Verify the transferred corpus hashes against the recorded source hashes.
- [ ] Preserve train/validation/test membership and keep the test split untouched during tuning.

The pilot launcher audits the corpus and deterministically creates:

- 1,000 training examples for each of CVRP20, CVRP50, and CVRP100;
- 250 validation examples for each size;
- hard-linked subsets under `data/processed/s7799_1k_per_size/`.

## 4. Bounded 1k-per-size GPU pilot

- [ ] Start the complete pilot from PowerShell through Git Bash:

  ```powershell
  & "C:\Program Files\Git\bin\bash.exe" scripts/run_3060ti_training.sh
  ```

The launcher performs CUDA preflight, corpus audit, deterministic subset selection, GAT
pretraining, and frozen-GAT diffusion training. The committed limits are approximately 30 minutes
for GAT plus 2.5 hours for diffusion; epoch-level limit checks can make the total about 3–4 hours.

- [ ] Keep the terminal open and monitor the live logs under `outputs/`.
- [ ] Confirm timestamped configs, metrics, environment metadata, and checkpoints appear under
  `outputs/train/`.
- [ ] Record peak allocated/reserved VRAM, examples/second, optimizer steps/second, and epoch time
  separately for CVRP20, CVRP50, and CVRP100 where available.
- [ ] Confirm losses remain finite and validation AUC/F1 improve beyond trivial baselines.
- [ ] Confirm CVRP100 does not exhaust the 8 GB VRAM budget.
- [ ] If CUDA OOM occurs, first reduce per-step batch size while preserving effective batch size
  with gradient accumulation. Keep AMP enabled unless it is implicated in instability.

Resume the newest diffusion run after `last.pt` has been written:

```powershell
& "C:\Program Files\Git\bin\bash.exe" scripts/run_3060ti_training.sh --resume-latest
```

- [ ] Test this resume command before treating the setup as ready for a long run.
- [ ] Classify all results from this stage as pipeline/throughput evidence, not final
  solution-quality evidence.

## 5. Strong-label audit before full training

The current corpus contains feasible one-second PyVRP/HGS labels, not verified near-optimal
labels. CMD learns route membership, so similar route cost alone does not establish target quality.

- [ ] Run the 1,500-instance strong-label audit separately from GPU training:

  ```powershell
  python scripts/run_strong_label_audit.py
  ```

- [ ] Do not run this CPU-heavy audit concurrently with other CPU-heavy labeling jobs or the GPU
  pilot if host contention would distort throughput measurements.
- [ ] Preserve its config, source hashes, candidate cache, summaries, metrics, accepted references,
  and rejected/ambiguous cases.
- [ ] Inspect every flagged case.
- [ ] Report results separately for CVRP20, CVRP50, and CVRP100:
  - feasibility and vehicle counts;
  - cost deltas and convergence across solver budgets/seeds;
  - same-route matrix disagreement;
  - per-customer route-membership stability;
  - challenger wins and runtime.
- [ ] Decide separately for each problem size whether to retain one-second labels, use
  best-of-multiple labels, construct soft/robust targets, or relabel that training size.
- [ ] Do not start final/full training until ambiguous label policy is resolved and accepted
  evaluation references are frozen.

## 6. Focused hyperparameter search

Run controlled comparisons on the same deterministic training and validation subsets. Change one
factor at a time initially, use fixed seeds, retain every resolved config, and never select settings
using the test split.

- [ ] Establish the committed 1k recipe as the baseline.
- [ ] Compare diffusion timestep sampling: `t_sample: high` versus `uniform`.
- [ ] Compare a small range of diffusion learning rates around `3e-4`.
- [ ] Measure safe per-size batch sizes and choose gradient accumulation from actual VRAM data.
- [ ] Compare weighted BCE settings, including `pos_weight_power`, against the same validation
  metrics and class prevalence.
- [ ] Measure diffusion augmentation on/off and the cost of the x9 augmentation recipe.
- [ ] Tune GAT learning rate/batch settings only if encoder validation learning is unstable or
  clearly underfit.
- [ ] Change diffusion depth/width only after the training recipe and data targets are stable.
- [ ] Run multiple seeds for finalists so a single favorable seed does not choose the recipe.
- [ ] Select using validation BCE, AUC, precision, recall, F1, calibration, full-chain sample
  quality, runtime, and memory—not training loss alone.
- [ ] Commit the chosen portable configuration without machine-specific absolute paths.

A broad automated sweep is not required initially. Prefer a small, evidence-driven search that
fits the 8 GB card and eliminates weak choices quickly.

## 7. Scaling and stopping gates

- [ ] Run deterministic learning curves at 1k, 5k, 10k, and 30k training examples per size.
- [ ] Use at least 1,000 validation and 1,000 untouched test examples per size for substantive
  comparisons.
- [ ] Keep dataset membership, hashes, seeds, evaluation code, and metric thresholds fixed across
  learning-curve runs.
- [ ] Stop increasing the subset when held-out gains flatten; do not assume all 180,000 training
  examples are necessary.
- [ ] Estimate duration and disk/checkpoint requirements from measured pilot throughput before
  every scale increase.
- [ ] Start a final/full run only when all of these are true:
  - label policy has passed the strong-label audit;
  - CUDA memory and numerical stability are demonstrated;
  - interruption/resume is verified;
  - the selected recipe wins controlled validation comparisons;
  - expected runtime and storage are acceptable;
  - evaluation datasets and hashes are frozen.

## 8. Evaluation and reporting

- [ ] Run a small post-training full-chain inference check, substituting the selected checkpoint:

  ```powershell
  python -m vrp_diffusion_quantum.inference.predict_matrix --checkpoint outputs/train/<run>/checkpoints/best.pt --val-dir cvrp_s7799_n20-50-100_x66667/splits/val --per-size 16 --device cuda
  ```

- [ ] Compare the supervised matrix predictor and diffusion model on the same untouched examples.
- [ ] Report metrics per size and per distribution, including BCE, AUC, precision, recall, F1,
  calibration, feasibility, route cost, runtime, and memory.
- [ ] Evaluate independent IID and R/C/RC spatial stress sets only after their stronger labels and
  immutable hashes are ready.
- [ ] Keep validation threshold tuning separate from the one-time final test report.
- [ ] Archive the exact config, code revision, environment, dataset hashes, checkpoints, logs,
  metrics, and evaluation reports for every result used in a claim.

## Recommended order

```text
CUDA and test smoke checks
-> bounded 1k-per-size GPU pilot
-> checkpoint/resume verification
-> 1,500-instance strong-label audit
-> resolve label policy by CVRP size
-> focused hyperparameter comparisons
-> 1k/5k/10k/30k learning curves
-> freeze recipe and evaluation data
-> final training
-> untouched test and stress-set evaluation
```
