# Data readiness TODO

This list turns the August 2026 repository data audit into ordered implementation work. Items are
roughly sorted from low-risk fixes to work that depends on trained models or long experiments.

## Current inventory

- [x] Record the main corpus: 200,001 labeled instances, with 66,667 examples for each of
  CVRP20, CVRP50, and CVRP100.
- [x] Verify the 90/5/5 size-stratified split: 180,000 train, 10,002 validation, and 9,999 test.
- [x] Verify that all recorded PyVRP solutions are feasible.
- [x] Identify the coverage limitation: the corpus contains only uniform customer/depot locations,
  uniform integer demands, and one fixed capacity per problem size.
- [x] Generate 9,000 independent plain-CVRP spatial stress instances (1,000 per size per regime):
  R seed 8801 at `data/raw/cvrp/spatial_stress_r_s8801` (SHA-256
  `d01f088205cbba5491ccd9ca66fbfaeaee8e6c37f1e28cf9fd0ca7dcf1af2f96`), C seed 8802
  (`cedfc0a8cfeeda99f4bd7183b68ef9edc58062b3e6ae8e6ce4ca2667e8649901`), and RC seed 8803
  (`84ebcf29dfed244b940b5ca3bf15a8cf117d5aa9b352e44cacead65b223fec5c`).
- [x] Validate all generated spatial sets reload with the expected 1,000 instances per size,
  capacities `{20: 30, 50: 40, 100: 50}`, depot demand zero, and customer demands in `1..9`.
- [x] Run a preliminary spatial separation check: over 200 CVRP100 instances, mean nearest-neighbor
  distance was R `0.0520`, RC `0.0469`, and C `0.0282`, in the expected order. Durable plots and
  fuller cluster statistics remain required below.

## Immediate fixes

- [x] Point the corpus-specific GAT and diffusion configs at the available `s7799` split.
- [x] Document the corpus-specific training command and keep generic configs independent of local
  dataset drops.
- [x] Avoid making a second full copy of example JSONs when creating splits; record the selected
  materialization method in the split manifest.
- [x] Add tests for storage-efficient split materialization and its error handling.
- [x] Add a preflight command that reports dataset existence, counts by size, split overlap, hashes,
  and disk footprint before training.
- [x] Load JSON constraint matrices as `uint8` instead of platform integers to reduce matrix memory
  by approximately eight times.
- [ ] Replace eager loading of every JSON example with a lazy/indexed dataset interface.
- [ ] Store constraint matrices compactly (`uint8` or packed binary) rather than as pretty-printed
  JSON integer arrays.

## RTX 3060 Ti / CUDA migration

### System and environment setup

- [ ] Record the desktop OS, GPU model/VRAM, CPU, RAM, storage, NVIDIA driver version, and CUDA
  runtime reported by `nvidia-smi` in the experiment environment notes.
- [ ] Install a current NVIDIA driver supported by the selected PyTorch build; do not depend on a
  separately installed CUDA toolkit unless compiling custom CUDA extensions.
- [ ] Create a clean Python 3.12 virtual environment on the desktop rather than copying `.venv`
  from the current machine.
- [ ] Install the CUDA-enabled PyTorch wheel using the command from the official PyTorch selector,
  then install this project with `python -m pip install -e ".[dev]"`.
- [ ] Save the resolved environment (`python -m pip freeze`) and record Python, PyTorch, CUDA,
  cuDNN, NumPy, and GPU versions without committing machine-specific environments.
- [ ] Copy or mount the gitignored corpus separately and verify its recorded SHA-256 hashes after
  transfer.
- [ ] Update only local/corpus-specific config paths if the dataset location differs on the
  desktop; keep portable template configs free of absolute machine paths.

### CUDA smoke checks

- [x] Add a diagnostic command that reports `torch.cuda.is_available()`, device name, VRAM,
  PyTorch CUDA version, and cuDNN version, and exits non-zero when CUDA was requested but is
  unavailable.
- [ ] Run a small CUDA tensor operation and backward pass before starting project training.
- [ ] Run the complete CPU test suite on the desktop.
- [x] Add CUDA-marked smoke tests for GAT forward/backward, diffusion noising/denoising,
  checkpoint save/load, and one tiny training batch; skip them cleanly when CUDA is unavailable.
- [ ] Run the CUDA-marked smoke tests successfully on the desktop GPU.
- [ ] Verify that `training.device: auto` selects `cuda` on the desktop and log the resolved device
  in every experiment artifact.
- [ ] Verify that checkpoints saved on CUDA load on CPU through `map_location` and can resume on
  CUDA.

### Memory and throughput work

- [ ] Measure free VRAM before training and establish safe batch sizes separately for CVRP20,
  CVRP50, and CVRP100; CVRP100's pairwise tensors are the limiting case.
- [x] Add optional automatic mixed precision (`torch.autocast` plus `GradScaler`) behind a config
  flag and test both enabled and disabled modes.
- [x] Add gradient accumulation so effective batch size can increase without exceeding GPU memory.
- [ ] Add optional gradient checkpointing only if profiling shows model activations, rather than
  data loading, dominate memory.
- [ ] Use pinned host memory and non-blocking device transfers after the lazy/indexed dataset
  loader exists; benchmark before keeping additional workers or prefetching enabled.
- [ ] Add configurable data-loader worker count, persistent workers, and prefetch factor with safe
  single-worker defaults for debugging.
- [x] Record peak allocated/reserved CUDA memory, examples per second, optimizer steps per second,
  and epoch runtime in training summaries.
- [x] Add a clear CUDA out-of-memory message that recommends reducing per-size batch size, enabling
  mixed precision/gradient accumulation, or disabling x9 expansion; do not silently drop batches.
- [ ] Benchmark the 1k-per-size subset first and estimate full-run duration before launching the
  10k-per-size or 180k-example configurations.

### Correctness and reproducibility

- [x] Seed Python, NumPy, CPU Torch, and all CUDA devices; record whether deterministic algorithms
  are enabled and document operations that cannot be deterministic.
- [ ] Compare one fixed tiny CPU and CUDA run for shapes, feasibility, finite losses, and reasonable
  metric agreement rather than requiring bitwise equality.
- [ ] Confirm that padding masks, capacity masks, matrix symmetry, and zero diagonals remain valid
  on CUDA for mixed CVRP sizes.
- [x] Add finite-loss and finite-gradient checks so NaN/Inf failures from mixed precision stop the
  run with useful diagnostics.
- [x] Preserve optimizer, scaler, epoch, RNG, and configuration state in resumable checkpoints
  (there is currently no learning-rate scheduler state to preserve).
- [ ] Test interruption and resume on the desktop before starting a long experiment.
- [ ] Keep validation/test membership identical between CPU and CUDA comparisons and log the exact
  split/source hashes.

### Desktop run sequence

- [x] Add a single bounded launcher for CUDA preflight, corpus audit, deterministic subset setup,
  GAT pretraining, and diffusion training, plus a latest-checkpoint resume command.

- [ ] Run `ruff check .`, `ruff format --check .`, `mypy src`, and the full CPU test suite.
- [ ] Run CUDA smoke tests on tiny fixtures.
- [ ] Run one GAT epoch on a small CVRP20 subset and verify artifacts/checkpoint resume.
- [ ] Run one diffusion epoch on the same subset without mixed precision, then with mixed
  precision, and compare metrics and memory.
- [ ] Run the stratified 1k-per-size GAT-to-diffusion pipeline and collect throughput/VRAM results.
- [ ] Select final per-size batch, accumulation, precision, worker, and augmentation settings from
  measured results and commit the reproducible config—not machine-specific paths.
- [ ] Start longer training only after the 1k-per-size run passes feasibility, logging, checkpoint,
  and resume checks.

## Phase 1 — label quality and coverage

- [x] Select 500 existing instances per size for a 1,500-instance label-quality audit.
- [x] Make newly solved labels preserve the configured hard solver limit as `time_budget` (or null
  only when a no-improvement criterion is used without a hard limit).
- [ ] Backfill or explicitly document the historical label metadata inconsistency: the corpus-level
  solve configuration records a one-second PyVRP limit, while its existing label/example records
  contain `time_budget: null`; do not rewrite the large corpus without recording new hashes.
- [ ] Treat the current 200,001 labels as feasible one-second PyVRP/HGS solutions, not verified
  near-optimal solutions, until the stronger-label audit is complete.
- [ ] Re-solve the 1,500-instance audit subset with longer PyVRP budgets (start with 10 and 30
  seconds) and multiple deterministic solver seeds; preserve every candidate rather than only the
  final winner.
- [x] Add a unified resumable parallel label-audit command with bounded workers, four
  per-instance deterministic PyVRP seeds, size-specific 10/20/40-second budgets, atomic candidate
  caches, progress reporting, and automatic retry of interrupted/failed runs.
- [x] Add automatic OR-Tools challenges for every cost-unstable or matrix-ambiguous audit instance
  plus a deterministic stable control sample of 50 instances per size.
- [x] Add explicit acceptance reports for feasibility, cost convergence, challenger wins,
  same-route matrix disagreement, per-customer membership instability, changes from the original
  one-second labels, vehicle counts, runtime, and cases requiring review.
- [x] Export all selected best references separately from matrix targets that pass every acceptance
  check; retain all candidates so ambiguous near-equal solutions are not discarded.
- [ ] Run `python scripts/run_strong_label_audit.py`, inspect every flagged case, and freeze the
  resulting config, source hash, candidate cache, summary, metrics, and accepted reference hashes.
- [ ] Report cost deltas, same-route matrix disagreement, per-customer route-assignment stability,
  feasibility, vehicles, and runtime between the one-second and stronger labels. Similar route
  costs alone are insufficient because CMD learns route membership.
- [ ] Break the label audit down by CVRP20/50/100 and decide separately whether each size can retain
  one-second labels, needs best-of-multiple labels, or needs broader re-solving.
- [ ] If stronger labels materially change route membership, re-label the affected training sizes
  or measure robust/soft targets from multiple near-equal solutions before final model training.
- [ ] Keep tonight's 1k-per-size CUDA run classified as a pipeline/throughput experiment until
  label quality is measured; do not use it to support final solution-quality claims.
- [x] Generate the independent-seed IID/R test instances with 1,000 examples per size.
- [ ] Label the independent-seed IID/R instances under the same stronger evaluation-label policy
  as C/RC, then freeze their membership and hashes.
- [x] Generate independent-seed plain-CVRP R/C/RC spatial stress instances with 1,000 examples per
  size and committed configs. These are spatially inspired categories, not Solomon VRPTW data.
- [ ] Label the R/C/RC spatial sets with a documented stronger multi-seed PyVRP policy and export
  immutable `CVRPExample` evaluation files with hashes.
- [ ] Complete R/C/RC spatial validation with dispersion and cluster metrics and plot representative
  instances before using the category names in reports (the preliminary nearest-neighbor check is
  recorded in the inventory).
- [ ] Generate and label a shifted-demand test set with 1,000 examples per size.
- [ ] Add capacity, route-size, and depot-placement shifts after the first three stress sets work.
- [ ] Keep the independent R set separate from the existing random training corpus and verify zero
  instance/hash overlap before evaluation.
- [ ] Decide explicitly whether genuine Solomon data is in scope. Solomon C/R/RC instances are
  CVRPTW and require time windows, service times, scheduling horizons, a CVRPTW feasibility checker,
  data-model changes, and a time-window-aware decoder; do not present the plain-CVRP spatial sets as
  genuine Solomon benchmarks.
- [ ] If CVRPTW becomes in scope, add a Solomon parser, preserve C1/C2/R1/R2/RC1/RC2 identities and
  published reference values, add time-window/service-time tests, and evaluate it as a separate
  task rather than mixing it into the current CVRP baseline.

## Phase 2 — supervised matrix predictor

- [ ] Run deterministic learning curves with 1k, 5k, 10k, and 30k training examples per size.
- [ ] Use at least 1,000 validation and 1,000 untouched test examples per size.
- [ ] Report per-size BCE, AUC, precision, recall, F1, calibration, class prevalence, and runtime.
- [ ] Stop scaling the training subset when held-out gains flatten; do not assume all 180k examples
  are necessary.

## Phase 3 — discrete matrix diffusion

- [x] Prepare deterministic hard-linked training and validation subsets for the 1k-per-size CUDA
  migration run.
- [ ] Run the GAT pretraining and denoiser on a small stratified subset before the full corpus.
- [ ] Measure runtime and memory for the x9 augmentation recipe at each learning-curve size.
- [ ] Compare the supervised predictor and diffusion model on the same untouched test examples.
- [ ] Increase generalization evaluation from smoke-test counts to at least 500, preferably 1,000,
  examples per distribution-size case.

## Phase 4 — encoder-decoder policy

- [ ] Implement deterministic on-the-fly instance generation for policy rollouts.
- [ ] Begin with approximately 100,000 CVRP20 rollout episodes for development.
- [ ] Scale toward 1–5 million total generated episodes only after rollout correctness and learning
  curves justify it; generated training instances do not need to be stored.
- [ ] Freeze separate validation sets and log their hashes.
- [ ] Store predicted matrices or model/checkpoint identifiers when policy experiments depend on
  `M_hat`.

## Phase 5 — baseline evaluation

- [ ] Evaluate at least 3,000 IID examples per size; retain the current test split untouched.
- [ ] Evaluate at least 1,000 examples per out-of-distribution distribution-size cell.
- [ ] Add a supported CVRPLIB subset and document exclusions.
- [ ] Use the stronger-label audit subset for fair cost-gap reporting.
- [ ] Freeze and tag `baseline-v1.0` only after all feasibility, ablation, and reproducibility checks
  pass.

## Phases 6–7 — refinement data

- [ ] After `baseline-v1.0`, save CMD solutions and uncertainty signals for 500–1,000 base instances
  per size.
- [ ] Create 20–50 hand-checkable toy neighborhoods for each QUBO formulation.
- [ ] Extract 500–1,000 development/validation neighborhoods per formulation without test leakage.
- [ ] Evaluate classical and quantum refiners on exactly the same initial solution, neighborhood,
  information, and budget.
- [ ] Use 500–1,000 paired simulator neighborhoods per size and reserve smaller hardware subsets
  only after simulator screening.
- [ ] Report every required group: CMD only, CMD + classical local search, CMD + quantum local
  search, and CMD + quantum local search + classical polish.

## Completion rule

Check an item only when its output is reproducible from committed code/configuration and its tests
or experiment artifacts include seeds, hashes, runtime, feasibility, costs, and configuration.
