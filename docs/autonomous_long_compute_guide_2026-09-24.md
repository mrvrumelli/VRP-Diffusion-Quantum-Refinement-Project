# Autonomous long-compute execution guide — 2026-09-24

Companion to [`project_findings_2026-09-24.md`](project_findings_2026-09-24.md). That doc's
"Ordered action plan" mixes fast code/analysis work with jobs that run unattended for a long time.
This guide pulls out only the long-running jobs, puts them in one execution queue, and records how
this assistant should move from one to the next without waiting for a prompt each time: launch in
the background, wait for the completion notification, record the result below under "Log", decide
go/no-go using the finding doc's own stated gate, then launch the next queued job.

Fast, code-only items (all of Section A except A.2's verification, plus D.1-D.3) are **not** in this
queue — do those inline, synchronously, the normal way.

## Classification: long-compute vs. fast

| Item | Compute? | Why |
|---|---|---|
| A.1 Ruff/format/mypy/pytest | Fast | Already verified clean this session (see below) |
| A.2 Reconcile `rcfull` artifacts | Fast (audit) / **feeds Task 1** | Reading manifests is fast; the missing piece is finishing the paused audit itself |
| A.3 Freeze baseline manifest | Fast | Writing down existing hashes/paths |
| A.4 Consolidate reverse-sampling | Fast | Already done, uncommitted (`predict_matrix.py` now delegates to `policy_support.sample_constraint_matrix_batch`) |
| A.5 Remove redundant local-attention-prior compute | Fast | Already done, uncommitted (`decoder.py` computes `local_prior` once, threads it into `LocalMaskedEncoder.forward`) |
| A.6 Batch inference / remove GPU syncs | Fast | Code change, not a long run |
| A.7 Route-structure diagnostics | Fast | Code change |
| A.8 Eval tooling (CI, matched comparisons) | Fast | Tooling only; the doc explicitly says running large panels is a separate step |
| **B.1** Confirm N20/N50 gains on larger frozen panel | **Long** | Full 700-step diffusion chain + policy rollout over a large panel, GPU |
| **B.2** Matched policy/diffusion/PyVRP/OR-Tools comparison | **Long** | Runs all four methods under declared budgets |
| B.3 Expand CVRPLIB eval | **Long** | Solver runs at larger scale than the one-instance smoke test |
| **B.4** Evaluate `rcfull` checkpoints on IID + all R/C/RC cells | **Long**, blocked on Task 1 | 9,000+ instance panel, GPU inference |
| **C.1** +500-1,500 audited IID sources/size, then learning curve | **Long — largest item** | CPU-bound solver audit (hours-to-a-day scale at this corpus size) + multiple GPU training runs |
| C.2 Stop-on-plateau decision | Fast | A judgment call using C.1's results |
| **C.3** Bounded N100 policy hyperparameter study | **Long** | Multiple ~1h GPU training runs |
| C.4 N100 gate check | Fast | Reading C.3's results against the 30.10% target |
| D.1-D.3 Classical refiner, QUBO formulation, leakage-free neighborhoods | Fast (design/implementation) | Not compute-heavy; gated behind the classical baseline being frozen (i.e., behind everything above) |
| **D.4** Simulate quantum/quantum-inspired approaches | **Long** (uncertain magnitude) | Depends on the chosen QUBO simulator; scope to the 20-50 hand-checkable cases from D.2 first |
| **D.5** Final CMD-only vs. +classical vs. +quantum matched comparison | **Long** | Same shape as B.2, run last |

## Verified state as of this session (2026-09-24)

- `ruff check .`: clean. `mypy src`: clean (50 files). `pytest -q`: **428/428** — but only after
  `--basetemp=.test-tmp/pytest_clean`; the default Windows temp dir
  (`C:\Users\Administrator\AppData\Local\Temp\pytest-of-Mustafa Mert`) is permission-denied
  (`PermissionError: [WinError 5]`) and fails 76 tests that use `tmp_path`. This is an environment
  issue, not a code regression — record it as the actual finding for A.1's "restore" item, and use
  `--basetemp` (or fix the Temp dir permissions) going forward.
- A.4 and A.5 are already implemented on disk, **uncommitted**, on `feat/dual-pointer-cvrp-policy`
  (`git diff --stat` shows `predict_matrix.py`, `decoder.py`, `local_masked_encoder.py` and their
  tests changed). Do not redo this work — just account for it when reconciling A.
- The fresh OOD eval-batch audit (`rc_full_eval`, seeds 9901-9903, the one
  [`project_status_2026-08.md`](project_status_2026-08.md) left "in progress") is **paused at
  18,219 / 36,000 candidates (~51%)**, not currently running (no worker processes present), and its
  last log (`outputs/logs/rc_full_eval_audit_resume_20260826T065405Z.log`) ends mid-progress-bar
  with no error/traceback — a clean external stop, not a crash. Safe to resume.
- **The config it needs, `configs/data/label_audit_rc_full_eval.yaml`, only exists on branch
  `local-global-encoder-merve` (commit `b91b95179`), not on the current `feat/dual-pointer-cvrp-policy`
  branch.** The pooled input data it reads (`outputs/label_audit_full_eval/pooled`) is present on
  disk already (filesystem outputs aren't branch-scoped). This needs a decision before Task 1 can
  start — see "Before starting" below.
- GPU confirmed available (RTX 3060 Ti, CUDA 12.8, torch 2.11.0+cu128), 68 GB disk free.
- The original `outputs/label_audit/rc_full` bundle (the training-side R/C/RC audit) is genuinely
  gone from disk, matching the uncommitted note already added to
  [`3060ti_training_todo.md`](3060ti_training_todo.md).

## Before starting: config brought over (done 2026-09-24)

`configs/data/label_audit_rc_full_eval.yaml` has been copied onto this branch from
`local-global-encoder-merve` (`git show local-global-encoder-merve:configs/data/label_audit_rc_full_eval.yaml > configs/data/label_audit_rc_full_eval.yaml`, no branch switch, currently untracked/uncommitted). Verified byte-for-byte equivalent to the settings baked into the cached
`outputs/label_audit/rc_full_eval/config.yaml` (`workers: 11`, same seeds/budgets/tolerances) — the
resume check compares against that cached config, so **do not pass `--workers` on resume**; it
defaults to the file's own `workers: 11` and will match. Passing a different worker count (e.g. 12)
would raise `ValueError: existing audit config differs` instead of resuming.

## Execution queue

Work one item at a time **within the same resource pool**. Never run a CPU-bound label audit
concurrently with another CPU-heavy multiprocessing job (another audit, or a GPU run's own CPU-bound
data-prep step) — that was the source of the 95°C thermal spike on 2026-08-16 (later shown safe at
full parallelism once nothing else was contending for the same cores).

**Exception: Task 1 and Task 2 may run concurrently.** Task 1 is pure CPU (11 PyVRP/OR-Tools worker
processes, never touches the GPU); Task 2 is GPU-bound (diffusion + policy rollout on CUDA, with only
a light single-process CPU footprint). They bottleneck on different resources, so pairing them is not
the same class of contention as two CPU-heavy jobs. Effects to expect, not to worry about: Task 1's
11 workers plus Task 2's host process slightly oversubscribes the Ryzen 5600's 12 logical threads, so
each may run a touch slower than running alone (low single digits of % for Task 1's audit rate; Task
2 is GPU-bound so it will barely notice) — this is a mild slowdown, not a thermal or correctness
concern, since full-core-count load was already the *cooler* regime in the 2026-08-16 test (<70°C
vs. ~95°C at partial-core boost). Every other same-resource pairing in this queue (two CPU audits, or
two GPU training/eval runs) still follows the one-at-a-time rule.

### Task 1 — Finish the paused OOD eval-batch audit (long, CPU-bound)

```bash
./.venv/Scripts/python.exe scripts/run_strong_label_audit.py --config configs/data/label_audit_rc_full_eval.yaml
```

Resumes from the 18,219 cached candidates automatically (verified resumable behavior). Do not add
`--workers` — the config's own `workers: 11` must match the cached run's `workers_resolved: 11` or
the resume check raises `ValueError: existing audit config differs`. 11 workers is close to the
12-worker level already confirmed thermally safe (<70°C) on this machine on 2026-08-16. Remaining
work is roughly 17,781 candidates; expect several hours to about a day depending on which sizes are
left (CVRP100 at a 40s budget is the slow end).

**Completion signal**: process exits 0, `summary.csv`/`metrics.json` written under
`outputs/label_audit/rc_full_eval/`, candidate count reaches 36,000.

**On completion**: materialize the accepted examples into a held-out OOD panel (do **not** fold into
training — that is the entire point of the fresh seeds). This unblocks Task 3 (B.4).

### Task 2 — Confirm N20/N50 policy gains + get a real N100 number on a larger frozen panel (long, GPU)

Uses the existing `s7799_val100_policy_v1` panel (already frozen, no new audit needed) at full size
rather than the smaller per-size slices used for the original K=1 comparison.

```bash
./.venv/Scripts/python.exe scripts/eval_policy_k1.py \
  --checkpoint outputs/policy/policy_reinforce_s7799_n20_heldout_cuda_20260828T081136097007Z/checkpoints/best.pt \
  --val data/processed/s7799_val100_policy_v1_n20 --num-starts 1 --device cuda
# repeat for n50 (…heldout_cuda_20260828T082044076031Z…) and n100 (…heldout_cuda_20260828T090345301408Z…)
```

Pair with the matching frozen diffusion-only decode on the same panel
(`python -m vrp_diffusion_quantum.inference.predict_matrix --checkpoint <persize champion> --val-dir data/processed/s7799_val100_policy_v1_n{size} --device cuda`) so the delta is apples-to-apples on
one run, not compared against the old smaller-panel numbers.

**Completion signal**: both K=1 gap numbers and the diffusion-only gap are recorded for all three
sizes.

**On completion**: this directly answers B.1. Report the real outcome, including if N20/N50 stop
looking like wins on the bigger panel — do not just confirm the earlier smaller-panel numbers by
assumption.

### Task 3 — Evaluate the `rcfull`-merged checkpoints properly (long, GPU; needs Task 1 done)

Score `diffusion_denoiser_s7799_rcfull_persize_n{20,50,100}_cuda` on:
- IID: the same `s7799_val100_policy_v1` panel used above.
- OOD spatial: the freshly completed `rc_full_eval` panel from Task 1 (never seen in training,
  unlike the original `rc_full` pool that went into the merged training set).

This is the real arbiter [[project_status_2026-08]] flagged as unresolved: whether the rcfull merge
is a genuine win on R/C/RC data despite regressing on the s7799 IID panel, or just a strictly worse
model.

**Completion signal**: per-size, per-regime gap table exists for both champions and rcfull-merged
checkpoints on both panels.

**On completion**: update the freeze decision in `stochastic_reference_probe.md` / the
`3060ti_training_todo.md` correction note with a real verdict instead of "unresolved."

### Task 4 — Matched-budget comparison + CVRPLIB expansion (long, mixed CPU/GPU)

```bash
./.venv/Scripts/python.exe eval/compare_or_baselines.py \
  --data-dir outputs/label_audit/s7799_strong_reference/accepted_matrix_examples \
  --sizes 20 50 100 --instances-per-size 20 \
  --time-budgets 1.0 5.0 --solvers pyvrp ortools --seed 42
```
scaled up from the current 2-per-size smoke run, plus the frozen policy/diffusion numbers from Tasks
2-3 reported alongside under the same declared time budgets (B.2). Expand the CVRPLIB subset beyond
the current one-instance parser smoke test the same way (B.3) — pull in more CVRPLIB instances and
run them through the same matched-budget harness.

**Completion signal**: one report covering PyVRP, OR-Tools, diffusion-only, and policy under
identical declared budgets, plus a CVRPLIB table beyond one instance.

### Task 5 — Expand audited IID sources and run the learning curve (long — the largest item)

```bash
python scripts/run_strong_label_audit.py --config <new config, 500-1500 more sources/size, same policy as label_audit_strong_s7799.yaml>
```
then train the 500/1,000/2,000-per-size curve (9 diffusion runs: 3 sizes x 3 curve points) using the
existing per-size training recipe. This is the single biggest compute item in the whole plan — likely
comparable in scale to the original 9,000-instance `rc_full` audit (multi-hour-to-day CPU) plus 9
GPU training runs at up to a few hours each.

**Completion signal per curve point**: checkpoint + large-panel decoded gap recorded.

**Stop rule (C.2, do not skip)**: stop adding data once the frozen-panel decoded route-gap gain
flattens between consecutive curve points. Do not use matrix F1 or training loss alone to decide.

### Task 6 — Bounded N100 policy hyperparameter study (long, GPU)

Sweep learning rate schedule, entropy regularization, normalization, number of starts, and curriculum
around the existing `policy_reinforce_s7799_n100_heldout_cuda.yaml` base, using the same K=1 eval
methodology as Task 2. Each run is roughly an hour at the existing 3,600s cap; budget for 5-15 runs.

**Gate (C.4)**: report whether any configuration beats the 30.10% diffusion-only large-panel gap
without losing feasibility. A clean "still behind, here is by how much" is an acceptable, valid
outcome — do not keep tuning indefinitely looking for a win that may not be there.

### Task 7 — Quantum/quantum-inspired simulation screening (long, magnitude uncertain)

Only after D.1-D.3 (classical local refiner, QUBO formulation, leakage-free neighborhood sets) are
implemented — those are fast/code items, do them inline before this task. Scope the first simulation
run to the 20-50 hand-checkable cases from D.2 before anything larger.

### Task 8 — Final matched comparison (long)

CMD-only vs. CMD+classical refinement vs. CMD+quantum refinement vs. quantum+classical polish, from
identical initial solutions, matched budgets. This is the `baseline-v1.0` capstone evaluation.

## How this gets executed autonomously

1. Launch the current task's command via a backgrounded shell.
2. Do not poll. Continue other inline (fast) work, or wait, until the completion notification
   arrives.
3. On completion: read the actual output/logs (not just the exit code), append a dated entry to the
   **Log** section below with the real numbers and the gate decision, then immediately launch the
   next queued task.
4. If a gate fails, or a job errors instead of completing, stop the queue and report — do not
   silently retry or skip ahead. Section C/D tasks in particular are explicitly conditioned on
   earlier results (stop-on-plateau, the N100 gate, "only after the classical baseline is frozen"),
   so treat those as real branch points, not formalities.
5. Never start a new long CPU/GPU job while another is still running.

## Log

(Append one entry per completed task: date, task, actual numbers, gate decision, next action.)
