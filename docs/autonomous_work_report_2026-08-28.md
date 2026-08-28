# Autonomous work report — 2026-08-28

This work block picked up `feat/dual-pointer-cvrp-policy`, a branch a collaborator (Hasv07, with a
Cursor co-author) had pushed with ~3,400 new lines implementing CMD Algorithm 1: a dual-pointer
(local + global attention) REINFORCE policy that decodes routes using the frozen diffusion denoiser
as a prior (`M_hat`), trained with a POMO multi-start baseline. It was heavily unit-tested but had
never been run against real data or real hardware. All changes below (configs and one new script)
are uncommitted for review.

## Environment fix

`pyproject.toml` had already been pinned to `pyvrp>=0.14` (commit `f6201e919`, "Port PyVRP labelling
to the 0.14 Model API") and `solve_cvrp.py` rewritten for the 0.14 `Model.add_location` API, but the
checked-out `.venv` still had `pyvrp==0.13.4` installed — 8 of 394 tests failed with
`AttributeError: 'Model' object has no attribute 'add_location'`. Fixed with
`pip install --upgrade "pyvrp>=0.14"` (resolved to 0.14.0). Full suite (394 tests) and the 4
CUDA-marked tests now pass; `torch==2.11.0+cu128` was unaffected.

## Delivered

- Six new smoke-scale policy configs
  (`configs/policy/policy_reinforce_s7799_n{20,50,100}_{smoke,heldout}_cuda.yaml`) confirming the
  REINFORCE + diffusion-prior pipeline runs end to end on this machine at all three problem sizes —
  100% feasibility on every epoch, every size, every run.
- Two new held-out validation subsets (`data/processed/s7799_val100_policy_v1_n{20,50,100}`),
  filtered from the existing frozen large-panel eval set `s7799_val100_policy_v1` (the same panel
  that scored the diffusion champions), with confirmed zero instance-ID overlap against each size's
  500-example training pool.
- A new reusable evaluation script (`scripts/eval_policy_k1.py`) that reloads a saved policy
  checkpoint plus its recorded prior denoiser straight from the checkpoint's `extra` payload (no
  source config file needed) and re-scores it at arbitrary `num_starts` — used to get a true
  single-decode (K=1) comparison against the diffusion pipeline's single-sample gap metric.

## Result: first real comparison against the frozen diffusion champions

The diffusion champions' published large-panel gaps (22.00% / 28.90% / 30.10% for N20/N50/N100,
see [`stochastic_reference_probe.md`](stochastic_reference_probe.md)) score **one decoded sample
per instance** (`eval/routing.py evaluate_decoded_matrix`), not a best-of-K oracle. An early smoke
comparison using the policy's own best-of-num_starts metric looked like a large win, but that was
comparing best-of-8 against single-decode — not apples to apples. Recomputing at true `num_starts=1`
via `eval_policy_k1.py`:

| Size | Policy K=1 gap | Diffusion champion (K=1) | Delta | Training budget |
|---|---:|---:|---:|---|
| N20 | **18.05%** | 22.00% | **−3.95 pp** | 15 epochs, num_starts=8 |
| N50 | **28.01%** | 28.90% | **−0.89 pp** | 8 epochs, num_starts=8 |
| N100 | **31.54%** | 30.10% | +1.44 pp | 6 epochs, num_starts=8 (fair-budget rerun) |

N20 and N50 are real, methodologically-clean wins for the dual-pointer policy over the frozen
diffusion-only decode path. N100's first attempt (3 epochs, num_starts=4) was clearly undertrained
(K=1 gap 31.89%); a matched-budget rerun (6 epochs, num_starts=8, ~1h wall-clock) narrowed the gap
only slightly to 31.54% — still behind, and the per-epoch validation curve was non-monotonic
(28.10% → 31.34% → 32.16% → 29.88% → 29.18% → 27.71% on the best-of-8 metric), suggesting either
more epochs are needed or the fixed hyperparameters (`learning_rate: 1e-4`, `entropy_weight: 0.0`,
no LR schedule) are not yet well-tuned for N100's larger action space. Not resolved further this
session — pushing past this point is a real compute/tuning investment, not a quick check.

## Caveats (flagged deliberately, not smoothed over)

- All comparisons use the frozen persize diffusion denoiser checkpoints as the policy's prior; the
  policy was trained on the same 500-example-per-size pool the diffusion champions were trained on,
  but validated on a held-out subset of the panel used to *validate* (not train) those champions —
  this is the right comparison for "does adding a REINFORCE policy on top help", not for absolute
  generalization claims beyond that panel.
- N100's result should be treated as provisional/negative-so-far, not "the policy loses at N100" —
  the training curve had not converged and no hyperparameter tuning was attempted.
- The two model families decode differently (autoregressive greedy pointer network vs. stochastic
  denoising diffusion chain), so even K=1 metrics are not measuring identical processes, only the
  same downstream cost-gap outcome.

## Quality gates

Ran the checklist from [`3060ti_training_todo.md`](3060ti_training_todo.md) §2, not yet done on
this branch: `ruff check .` was clean except one import-order nit in this session's own new script
(auto-fixed); `ruff format --check .` flags 84 files repo-wide as unformatted, but that predates
this session and is unrelated to the dual-pointer work, so left alone. `mypy src` found 2 real
errors in `models/decoder.py` (`CVRPPolicy.rollout`, lines 852-871): `start_table` is assigned
`Tensor` in the `num_starts is None` branch and `Tensor | None` in the `else` branch with no
explicit annotation, so mypy inferred a non-optional type from the first branch and then flagged
the `else` assignment as incompatible and the following `if start_table is None:` check as
unreachable. Not a runtime bug (extensively covered by passing tests) but it silently defeated type
checking at this spot. Fixed with one explicit `start_table: Tensor | None` annotation; `mypy src`
is now clean (50 files), full test suite still passes (394/394).

## Code review: one confirmed correctness bug, independently verified

An 8-agent `code-review high` pass on `main..feat/dual-pointer-cvrp-policy` surfaced a real bug
that undercuts every benchmark number above. `models/local_masked_encoder.py:184-186`'s
`customer_to_depot = full_customer_nodes[:, :, None] & depot_nodes[:, None, :]` only encodes
customer→depot, never depot→customer, so the depot's row in the local-attention-prior adjacency
matrix has no customer columns set. `models/decoder.py`'s `_RolloutState.local_available()`
(lines 632-639) reads exactly that row whenever `current_node == depot_index` — i.e. at the start
of every route (potentially dozens of times per CVRP100 instance) — finds the restricted mask
all-False, and falls back to the full unrestricted `action_mask`. **The local, diffusion-prior-
guided pointer therefore contributes zero signal at every route-opening decision, silently.**
I verified this by reading both files directly (not just trusting the review agent).

**Correction after tracing this further, same session: this does not undermine the N20/N50/N100
numbers, and is not a fixable bug.** `build_decoder_local_adjacency` computes `allowed_pairs &
(weights >= 0.5)`; at the depot row `weights[depot, customer]` is 0 for every customer (only
`customer→depot` gets a weight contribution). Adding a symmetric `depot_to_customer` term to
`allowed_pairs` alone would still fail the threshold check, since weights would still be 0 — a real
fix would also need to inject a weight for `depot→customer`, and because `M_hat` is a customer-only
route-membership matrix with zero depot-related entries by construction, the only weight available
to inject is a uniform constant, which produces "allow every customer equally" — **the same outcome
as the current fallback**. There is no recoverable signal: `M_hat` structurally cannot inform
route-opening decisions, so no masking fix changes anything. No code change made; this is a
documentation gap (the fallback's intentionality isn't explained), not a defect affecting the
reported results.

The review also flagged, and this session went on to fix: a default lazy-dataset training path
that skipped epoch shuffling when `same_size_batches`/`expand_any`/`stochastic_references` are all
unset (confirmed reachable, verified **not** triggered by any currently-committed config including
the frozen champions, so no past training run was affected). Traced the root cause fully:
`train_diffusion.py`'s lazy `IndexedJSONDataset` fast path (line 593, now `lazy_unshuffled`) skips
`_shuffle_and_maybe_augment_online` entirely — unlike the plain-list path, it has no earlier
shuffle stage to fall back on, so it must always tell `_batches` to shuffle, independent of
`same_size_batches`/`expand_any`. Fixed with `shuffle=True if lazy_unshuffled else
(same_size_batches or expand_any)` at the `_batches(...)` call site (line ~632). Added a monkeypatch
regression test (`test_indexed_json_dataset_default_path_still_shuffles_each_epoch`,
`tests/test_train_diffusion.py`) that spies on the `shuffle` kwarg `_batches` receives — verified it
fails on the pre-fix code (`[False, False]`) and passes after (`[True, True]`) by stashing/unstashing
the fix. Full suite: 395/395 (was 394 + 1 new test), `ruff check`/`ruff format --check` (this
file only)/`mypy src` all clean.

Several other efficiency issues (per-step GPU syncs in the decode loop,
`build_local_attention_prior` computed twice per `encode()` call, unbatched CLI inference); and
several code-duplication issues (the reverse-diffusion sampling loop is reimplemented in
`policy_support.py` and has already drifted from `predict_matrix.py`'s original). One flagged
concern — PyVRP `activity.idx` route-indexing correctness — was independently tested this session
(solved a fresh instance, checked customer identities and capacity against ground truth) and
disconfirmed.

Separately verified (script-only, no production code touched) that `build_local_attention_prior`'s
two call sites in `CVRPPolicy.encode()` (via `build_decoder_local_adjacency`) and
`LocalMaskedEncoder.forward()` do produce bit-identical output for identical inputs, and that
`m_hat`'s dtype always matches the embeddings' dtype in this codebase (no AMP/autocast anywhere in
the policy or decoder path) — confirming the redundant-computation efficiency finding is real and
safely fixable (thread one precomputed prior through both call sites). **Deliberately not
implemented this session**: it touches the same forward-pass code whose behavior the N20/N50/N100
benchmark numbers above depend on, and fully re-validating "identical or better result after the
refactor" would need re-running training for at least one size (~30+ min) — left as a well-scoped,
low-risk follow-up rather than a change made without budget to re-verify it.

## Verification

- `pytest -q`: 394 passed (before and after the mypy fix). `pytest -q -m cuda`: 4 passed.
- All 9 training runs (3 smoke + 3 held-out + 1 N100 rerun, plus the 2 K=1 eval passes) completed
  with exit code 0 and 100% capacity-feasible decoded/rolled-out routes; logs retained under
  `outputs/logs/policy_reinforce_*` and metrics/checkpoints under `outputs/policy/`.
