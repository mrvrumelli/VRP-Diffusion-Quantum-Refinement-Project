# Autonomous long-compute execution guide — 2026-09-24

Companion to [`project_findings_2026-09-24.md`](project_findings_2026-09-24.md) (Track A: the
`ours_robust` action plan) and [`cmd_paper_alignment_plan.md`](cmd_paper_alignment_plan.md)
(Track B: the `paper_cmd` faithful-reproduction plan, merged onto this branch from
`feat/paper-cmd-reproduction` on 2026-09-24). Both source docs mix fast code/analysis work with jobs
that run unattended for a long time. This guide pulls out only the long-running jobs from both, puts
them in two execution queues, and records how this assistant should move from one task to the next
without waiting for a prompt each time: launch in the background, wait for the completion
notification, record the result below under "Log", decide go/no-go using the source doc's own stated
gate, then launch the next queued job.

Fast, code-only items (all of Track A's Section A except A.2's verification, D.1-D.3, and Track B's
Phases 0/1/4-the-code-part) are **not** in this queue — do those inline, synchronously, the normal
way.

## Track A (`ours_robust`) classification: long-compute vs. fast

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

## Track B (`paper_cmd`) classification: long-compute vs. fast

Two of Phase 3's requirements (the skipped-transition sampler math, paper augmentation functions)
and part of Phase 1 (clean repo gate) were already implemented by the `feat/paper-cmd-reproduction`
merge — not re-listed as open work below.

| Phase | Item | Compute? | Why |
|---|---|---|---|
| 0 Freeze comparison contract | Named `paper_cmd`/`ours_robust` configs, doc every paper detail as explicit/inferred/ambiguous | Fast | Docs/config only |
| 1 Clean repo gate | ruff/mypy/pytest, paper-mode smoke test | Fast | Already green on this branch; smoke test is a small code addition |
| 2 Paper data — generate 200,000 unlabeled RL instances | **Fast — already satisfied, no compute needed** | The existing `cvrp_s7799_n20-50-100_x66667` corpus's 180,000-example train split already matches the paper's distribution (verified 2026-09-24, see below) |
| 2 Paper data — freeze 1,000 test instances/size | **Fast — already satisfied** | The existing test split has 3,333/size, already exceeds the requirement |
| 2 Paper data — **label 50,000 diffusion-training instances** | **Long** | The one real gap. ~29-30h at 11 workers if single-seed HGS-only, per the paper's own `label_policy: single_hgs_route_partition` (see Task 9) |
| 3 Faithful diffusion — implement skipped sampler, config templates | Fast | Already merged in (`q_posterior_between_prob`, `sampler="skipped_posterior"`, `*_paper_cmd.yaml` configs) |
| 3 Faithful diffusion — **train paper-config GAT+diffusion; gate F1≈0.823@50 steps** | **Long** | GPU; needs Task 9's labeled data first |
| 4 Faithful decoder — implement `architecture: paper_cmd` in `CVRPPolicy` (currently `NotImplementedError`) | Fast (code) | **Blocks Task 11 below** — not compute itself |
| 4 Faithful decoder — tiny RL smoke gate | Fast/short | A quick correctness check, not a long run |
| 5 Train paper baseline | **Long — biggest single item** | Plan's own estimate: 1-3 elapsed weeks (pilot → comparison → full diffusion → full 200k/100-epoch policy training → 3-seed confirmation) |
| 6 Exact evaluation suite (synthetic 1,000/size + CVRPLIB-17 + full XML100 10,000 instances + HGS/POMO baselines) | **Long** | XML100-scale baseline runs are audit-scale CPU work again |
| 7 Reproduce ablations (Gaussian-vs-Bernoulli, no-masked-encoder, no-local-pointer, frozen-vs-fine-tuned-GAT × 7 inference-step settings) | **Long** | Plan estimate: 1-2 weeks; each arm is its own training run |
| 8 Robustness study (8 arms × IID/XML100/CVRPLIB/4 corruption levels/cross-size/R-C-RC) | **Long — second-biggest item** | Plan estimate: 4-7 days, but only after Tasks 11 and 13 both exist |
| 9 Freeze baseline | Fast | Bookkeeping/tagging once everything above is real |

### What's already satisfied (verified 2026-09-24, no new generation needed)

Checked `cvrp_s7799_n20-50-100_x66667/README.md` against Phase 2's stated needs:

- **Distribution matches.** Depot/customers uniform random in `[0,1]²`, demand `U{1..9}`, capacity
  30/40/50 for N=20/50/100 — the same convention the paper uses. Confirmed independently: the
  corpus's quick 1-second PyVRP costs (6.11 / 10.36 / 15.79 for N20/50/100) are strikingly close to
  the paper's own reported no-augmentation CMD costs (6.20 / 10.64 / 15.76) — two independently
  generated pools landing that close is strong evidence the generation protocols match.
- **200,000 unlabeled RL instances**: use the existing 180,000-example train split directly as
  `policy_reinforce_paper_cmd.yaml`'s `dataset.path` (its `dataset.name: paper_cmd_uniform_rl_200k`
  placeholder confirms this is exactly the slot). Short of the exact 200,000 figure by ~20,000 (held
  out for val/test) — record this as a documented, deliberate deviation per Phase 0's own
  explicit/inferred/ambiguous tracking, not a silent substitution.
- **1,000 test instances/size**: the existing test split already has 3,333/size; just select a
  1,000/size subset for the frozen paper_cmd test panel.
- **Do not repurpose the paused `rc_full_eval` audit (Track A Task 1) for this.** It's R/C/RC
  spatial-stress data, not the plain IID distribution above, tops out at 9,000 instances even
  finished (18% of 50,000), and is explicitly reserved as a held-out OOD eval set for the unrelated
  rcfull-merge question (Track A Task 3) — using it here would both fail to match the right
  distribution and destroy the only OOD eval set that question has.

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

## Execution queues

Work one item at a time **within the same resource pool**, across *both* tracks — Track B's tasks
are numbered 9+ so the two queues share one global sequence and one Log. Never run a CPU-bound label
audit concurrently with another CPU-heavy multiprocessing job (another audit, or a GPU run's own
CPU-bound data-prep step) — that was the source of the 95°C thermal spike on 2026-08-16 (later shown
safe at full parallelism once nothing else was contending for the same cores).

**Exception: one CPU-only task and one GPU-only task may run concurrently, from either track.**
Established with Track A's Task 1 (pure CPU, 11 PyVRP/OR-Tools workers) paired with Task 2
(GPU-bound). The same reasoning applies across tracks: e.g. Track B's Task 9 (CPU label audit) with
Track A's Task 2 or 3 (GPU), or Track A's Task 1 with Track B's Task 10/11 (GPU training). They
bottleneck on different resources, so pairing them is not the same class of contention as two
CPU-heavy jobs. Effects to expect, not to worry about: an 11-worker CPU task plus a GPU task's host
process slightly oversubscribes the Ryzen 5600's 12 logical threads, so each may run a touch slower
than running alone (low single digits of % for the CPU task's rate; the GPU task will barely notice)
— this is a mild slowdown, not a thermal or correctness concern, since full-core-count load was
already the *cooler* regime in the 2026-08-16 test (<70°C vs. ~95°C at partial-core boost). Every
other same-resource pairing (two CPU audits, or two GPU training/eval runs, in either track or mixed)
still follows the one-at-a-time rule.

### Track A: `ours_robust`

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

### Track B: `paper_cmd` reproduction

Do the Phase 4 code item (implement `architecture: paper_cmd` in `CVRPPolicy`, currently
`NotImplementedError` in `build_policy_from_config`) inline, whenever — it's pure code, not queued
here, but **Task 11 is blocked on it**.

### Task 9 — Label 50,000 paper_cmd diffusion-training instances (long, CPU-bound)

Generation is **not** needed (see the classification section above) — only labeling is. The paper's
own protocol is a single HGS solve per instance treated as ground truth
(`diffusion_denoiser_paper_cmd.yaml`'s `dataset.label_policy: single_hgs_route_partition`), not this
project's stricter 4-seed + OR-Tools-challenger audit — reusing that cheaper, paper-faithful protocol
instead of our own is both more correct here and roughly 4x cheaper.

Stage it rather than committing to 50,000 blind:

1. **Pilot, ~5,000 instances (1,667/size).** Select a deterministic subset from the existing train
   split, then verify (do not assume) zero overlap with the 1,500 instances already in
   `s7799_strong_reference`:
   ```bash
   python scripts/select_dataset_subset.py \
     --source cvrp_s7799_n20-50-100_x66667/splits/train \
     --output data/processed/paper_cmd_hgs_pilot_5k \
     --sizes 20 50 100 --per-size 1667 --seed 50937
   ```
   `scripts/check_generated_dataset_overlap.py` does **not** fit here — it compares raw
   `generate_cvrp`-style CSV directories (`instance_content_hashes`), not the `CVRPExample` JSON
   directories `select_dataset_subset.py` produces. The right tool for JSON-example pools is
   `report_instance_id_overlap` in `src/vrp_diffusion_quantum/eval/matrix_ablation.py` (already
   used for exactly this purpose elsewhere), called on the two loaded pools — this needs a short
   one-off script (fast, write it before running the pilot, not a long-compute item itself).
   Then label with a new single-seed config (copy `label_audit_strong_s7799.yaml`, set
   `base_seeds` to one seed, `stable_sample_per_size: 0` to skip the OR-Tools challenger entirely,
   `expected_counts_by_size` to `{20: 1667, 50: 1667, 100: 1666}`):
   ```bash
   python scripts/run_strong_label_audit.py --config configs/data/label_audit_paper_cmd_pilot.yaml
   ```
2. Train Task 10 on the pilot first and check the F1≈0.823-at-50-steps gate before committing
   further — this is the actual decision point for whether 50,000 is worth it on this hardware, the
   same "don't audit blind" principle Track A's Task 5 already applies to its own label budget.
3. **Full scale, ~16,667/size = 50,000**, only if the pilot's trend justifies it. Same process,
   larger `--per-size`/`expected_counts_by_size`, excluding the pilot's own instances too. Estimated
   **~29-30 hours at 11 workers** (single-seed, 10/20/40s-by-size budgets — the paper doesn't state
   its own HGS budget, so this is a documented assumption, not a known fact).

**Completion signal**: `metrics.json`/`summary.csv` written under the audit's `output_dir`, accepted
count matches the target size.

**On completion**: point `diffusion_denoiser_paper_cmd.yaml`'s `dataset.path` at the result (pilot
first, then the full-scale set if Task 9 is redone), then launch Task 10.

### Task 10 — Train the paper-config diffusion model (long, GPU; needs Task 9)

```bash
python -m vrp_diffusion_quantum.train.train_diffusion \
  --config configs/train/diffusion_denoiser_paper_cmd.yaml
```
(after setting `model.gat_checkpoint` from a paper-config GAT pretrain run, and `dataset.path` /
`validation.path` from Task 9's output, per this config's own `null` placeholders).

**Gate (Phase 3)**: reproduce the paper's qualitative inference-step curve — approximately **F1
0.823 at 50 inference steps** — before connecting this prior to the policy at all.

**On completion**: if the pilot-scale run (Task 9 step 1) already clears the gate, that's evidence
the full 50,000-instance labeling (Task 9 step 3) isn't necessary; if it doesn't, that's the signal
to commit to the full scale and rerun this task.

### Task 11 — Full Phase 5 paper-baseline training (long — the single biggest item; needs Task 10 and the Phase 4 code item)

In increasing cost order, per the plan: CVRP20 convergence pilot → small joint-vs-per-size
comparison → full diffusion training at scale → full 200,000-instance/100-epoch policy training →
three-seed confirmation of the winning configuration. Plan's own estimate: **1-3 elapsed weeks** on
this GPU. Reduce memory pressure with batch size + gradient accumulation, never by reducing
`NStart` (the plan is explicit that this changes the learning objective, not just its cost).

**Gate**: the validation policy clearly improves over no-`M` and random-`M` controls under matched
compute.

### Task 12 — Build and run the exact evaluation suite (long, mixed CPU/GPU; needs Task 11 for the full comparison, Task 10 alone for a partial one)

Synthetic 1,000/size + the exact 17 CVRPLIB instances + all 10,000 XML100 instances (378 groups) +
HGS/POMO-compatible comparison outputs. The XML100 sweep is audit-scale CPU work again — budget
accordingly, and hand-check a subset against the official costs/category assignments before trusting
the full run (the plan's own gate for this phase).

### Task 13 — Reproduce paper ablations (long; needs Task 11)

Train each arm (Gaussian vs. Bernoulli diffusion, no masked encoder, no local pointer, frozen vs.
fine-tuned global GAT) and evaluate at inference-step counts 1/5/10/20/50/200/1000 through Task 12's
suite. Plan's own estimate: **1-2 weeks**. If exact paper test seeds are still unavailable, report
this as a **paper-guided reconstruction**, not an exact reproduction — the plan is explicit about
that distinction.

**Gate**: results are reasonably close to the paper across all three IID sizes and preserve its main
ablation ordering.

### Task 14 — Robustness study: paper baseline vs. `ours_robust` (long — second-biggest item; needs Tasks 11 and 13 both done)

Train the P1-P6 single-change arms (learned fusion gate only, Transformer global encoder only,
LayerNorm only, audited/stochastic labels only, per-size denoisers only, soft adjacency only), then
evaluate all 8 arms (P0 faithful paper, P1-P6, `ours_robust`) on: IID synthetic, full XML100,
CVRPLIB-17, mask-corruption sweeps at 0/5/10/20%, stable-vs-ambiguous label subsets, cross-size
tests, and the R/C/RC spatial stress sets. Plan's own estimate: **4-7 days**, but only once both
prerequisite tasks exist.

**Gate**: an extension is only accepted if it improves OOD mean or tail performance under matched
compute, keeps 100% feasibility, and causes no material IID regression.

Only after Task 14 (and Track A's Tasks 1-6) does Phase 9's freeze — and, per both plans, Track A's
quantum-refinement Tasks 7-8 — become appropriate to start.

## How this gets executed autonomously

1. Launch the current task's command via a backgrounded shell.
2. Do not poll. Continue other inline (fast) work, or wait, until the completion notification
   arrives.
3. On completion: read the actual output/logs (not just the exit code), append a dated entry to the
   **Log** section below with the real numbers and the gate decision, then immediately launch the
   next queued task.
4. If a gate fails, or a job errors instead of completing, stop the queue and report — do not
   silently retry or skip ahead. Both tracks have tasks explicitly conditioned on earlier results
   (Track A: stop-on-plateau, the N100 gate, "only after the classical baseline is frozen"; Track B:
   the F1≈0.823 gate deciding whether Task 9's full 50,000-instance labeling is even needed, Tasks
   11/13/14's hard sequential dependencies) — treat those as real branch points, not formalities.
5. Never start a new long CPU/GPU job while another same-resource job is still running (see the
   CPU/GPU concurrency exception above for what's allowed).
6. Track A and Track B are independent research programs sharing one machine and one queue
   discipline — nothing requires finishing Track A before starting Track B, or vice versa. Pick
   whichever track's next task is actually unblocked (data/checkpoint dependencies met, any
   inline code prerequisite already done) rather than assuming strict track order.

## Log

(Append one entry per completed task: date, task, actual numbers, gate decision, next action.)
