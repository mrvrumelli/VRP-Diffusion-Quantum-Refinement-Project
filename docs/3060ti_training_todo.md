# RTX 3060 Ti training checklist

> **Completed 2026-08-15.** The original 1k/5k/10k/30k roadmap below was written before the
> strong-label audit established that only 500 audited training sources per size are available.
> The executed evidence-based curve was 100/250/500 per size. See
> `docs/3060ti_training_report.md` for the authoritative configs, hashes, metrics, final checkpoint,
> untouched-test result, and limitations. Remaining unchecked historical items are not prerequisites
> for the completed bounded policy-v2 run unless a future larger corpus is generated.

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
