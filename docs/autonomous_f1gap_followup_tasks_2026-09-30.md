# F1-gap follow-up tasks — 2026-09-30

Evidence and rationale: [root-cause report](f1_gap_root_cause_2026-09-30.md). The report showed that
the customer-only input caps depot-blind F1 and that adding the depot is necessary but not
sufficient. These tasks test the remaining candidate bottlenecks, record the corrected
reconstruction assumptions, and prepare the classical baseline freeze that gates quantum work.

**Operating rules**

- One training job on the GPU at a time. Coding, tests and documentation proceed on the CPU
  while it runs.
- Every model probe uses the full filtered split, seed 42, 12 epochs, and otherwise the
  `paper_cmd` diffusion settings. It changes only the named factor relative to its parent.
- Every probe is scored on the frozen 72-graph development panel with the corrected sampler at
  1, 10 and 50 steps and sampling seeds 0 and 1. That panel has been used for selection, so all
  results are development evidence, not test claims.
- A change is adopted when 50-step pooled F1 improves by at least 0.01 on both sampling seeds.
  Otherwise the parent recipe is kept.
- Probes run on the project track because the current `paper_cmd` contract rejects the filtered
  split. Their claims say so explicitly.
- Package defaults stay unchanged. New behavior is behind explicit config flags with tests.

| Order | Task | Depends on | Main resource |
|---|---|---|---|
| F1 | Finish the trainable-GAT depot-aware probe | Running | GPU |
| F2 | Distance edge features on the denoiser | F1 selects frozen or trainable GAT | GPU, config only |
| F3 | DIFUSCO-faithful denoiser in one run | F2 selects the parent | Code and tests, then GPU |
| F4 | Depot as a real GAT graph node | F3 selects the parent | Code and tests, then GPU |
| F5 | Full-length training of the best recipe | F2–F4 | GPU, about 2 hours |
| F6 | Finish the root-cause report | F5 | Documentation |
| F7 | Record corrected assumptions in the evidence ledger | F1–F4 findings | Documentation |
| F8 | Repair the paper-track contract | F4, F7 | Code and tests |
| F9 | Classical baseline freeze proposal | F5, F8 | Analysis and manifests |

**F1 — Trainable-GAT depot-aware probe**

- [x] Finish the 12-epoch run with the depot-relative GAT fine-tuned during diffusion.
- [x] Score it on the panel against the frozen depot-aware probe and select the parent. 50-step
  F1 0.513 to 0.522 (+0.007 / +0.012 by seed) misses the rule, so the frozen GAT stays parent.

**F2 — Distance edge features**

- [x] Train the F1 parent with `edge_input_features: noisy_matrix_distance`.
- [x] Score on the panel and apply the adoption rule. 50-step F1 0.513 to 0.535 (+0.025 / +0.019
  by seed): adopted as the parent for F3. Not paper-compatible (the paper's edge input is `x_t`).

**F3 — DIFUSCO-faithful denoiser**

- [x] Add flags for the DIFUSCO bit-flip convention (probability β/2) and residual edge updates.
  Default behavior must remain bit-identical. Add unit tests for both. (`schedule.flip_convention`,
  `model.edge_residual`, shared `schedule_from_config`.)
- [x] Train the F2 parent with unweighted BCE, LayerNorm, β/2 noise and residual edges together.
- [x] Score on the panel, including whether F1 now rises with more steps. Rejected: 50-step F1
  0.535 to 0.448 and N20 collapses, although teacher-forced F1 is the best of any probe. A
  threshold sweep shows the chain output is a committed sample, so the bottleneck is the model's
  broad belief, not the sampler.
- [x] F3b (added): the same bundle with weighted BCE kept, to test the other three changes alone.
  50-step F1 0.535 to 0.536 (+0.001 / +0.002): not adopted. Its step curve is flat instead of
  falling, but N20 drops while N100 rises.

**F4 — Depot as a graph node**

- [x] Add an opt-in depot node to the GAT: the GAT runs on depot plus customers with a depot
  indicator feature, and the denoiser receives only customer embeddings. (`model.depot_node` for
  GAT pretraining, `model.gat_depot_node` for diffusion; requires the depot-relative frame.)
- [x] Support it in GAT pretraining, diffusion training, standalone inference, the policy prior
  and checkpoint metadata. Add tests, including loading older checkpoints unchanged.
- [x] Pretrain the GAT and train the F3 parent with it. Score on the panel. The depot-node GAT
  matches the depot-relative GAT in pretraining (validation F1 0.583 versus 0.582), but the
  diffusion probe scores 0.517 / 0.515 at 50 steps versus the parent's 0.538 / 0.531: not adopted.
  The depot node remains the paper-track form because it matches the paper's graph definition.

**F5 — Full-length training**

- [x] Train the selected recipe for up to 50 epochs with the existing early-stopping rule.
  Scoped as the paper-track recipe under the amended contract (depot-node GAT pretrained and
  frozen, BatchNorm, noisy-matrix edges, weighted BCE, corrected sampler) on the manifest-backed
  unfiltered single-HGS 50k split. The adopted distance edges are not paper-compatible, and the
  question this run answers is whether the corrected reconstruction approaches 0.823 at scale.
  Started 2026-09-30 21:03 (`outputs/f1gap_diagnostics_20260930/run_f5.sh`).

  **PAUSED by request on 2026-09-30 22:54.** The GAT stage is complete
  (`outputs/paper_cmd/gat_pretrain_paper_cmd_depot_single_hgs_20260930T180424537775Z`, 50 epochs,
  validation AUC 0.927, F1 0.586). Diffusion training was stopped during epoch 15; epochs 0–14 are
  saved in `outputs/paper_cmd/diffusion_denoiser_paper_cmd_depot_single_hgs_20260930T190005759219Z`.
  Best so far: sample F1 0.5069 at epoch 13 (`checkpoints/best.pt`); `checkpoints/last.pt` is
  epoch 14 with optimizer, scaler and RNG state. To continue:

  1. Resume training (writes a **new** timestamped run directory):
     `.venv/Scripts/python.exe -m vrp_diffusion_quantum.train.train_diffusion --config
     outputs/f1gap_diagnostics_20260930/f5_diffusion_paper_cmd.yaml --resume
     outputs/paper_cmd/diffusion_denoiser_paper_cmd_depot_single_hgs_20260930T190005759219Z/checkpoints/last.pt`
  2. **Merge the two run folders** for reporting: concatenate `summary.csv` (epochs 0–14 from the
     first folder, 15+ from the second). The early-stopping counter restarts at zero on resume, so
     the resumed run may train a few epochs longer than an uninterrupted one would have.
  3. **Pick the best checkpoint across both folders.** The resumed folder only writes `best.pt` if
     sample F1 beats 0.5069; otherwise the first folder's `best.pt` (epoch 13) remains the best.
  4. **Score by hand** — the pipeline's automatic scoring did not run:
     `PANEL_OUT=outputs/f1gap_diagnostics_20260930/panel_f5.json .venv/Scripts/python.exe
     outputs/f1gap_diagnostics_20260930/panel_steps.py <best.pt>` and
     `.venv/Scripts/python.exe outputs/f1gap_diagnostics_20260930/panel_routes.py <best.pt> f5_paper_cmd`.
  5. Then finish F6 (report verdict) and the F5 section of the F9 freeze proposal.

  **RESUMED on 2026-10-01 01:02** into
  `outputs/paper_cmd/diffusion_denoiser_paper_cmd_depot_single_hgs_20260930T220205942876Z` (first
  epoch 15). Steps 2–4 are automated by `outputs/f1gap_diagnostics_20260930/run_f5_resume.sh`,
  which writes the merged history to `outputs/f1gap_diagnostics_20260930/f5_merged_summary.csv`.
- [x] Score F1 and decoded route gap. Done at 1, 10 and 50 steps with two sampling seeds (the
  protocol used for every probe); 5, 20 and 200 steps were not run. Completed 2026-10-01 03:10:
  pooled F1 0.546 / 0.560 / 0.531, best checkpoint epoch 43, route gap 57.8% pooled.

**F6 — Root-cause report**

- [x] Add F1–F5 results and final conclusions to the report.

**F7 — Evidence ledger**

- [x] Record that the GAT must see the depot, because the paper applies it to the instance graph
  and reuses it for the policy.
- [x] Record freezing the GAT during diffusion training as an ambiguity; the paper states freezing
  only for the policy.
- [x] Record the noise convention and the Figure 8 F1 definition and validation split as open
  author questions.

**F8 — Paper-track contract**

- [x] Allow a depot-aware GAT in the `paper_cmd` policy and diffusion path. New `paper_cmd` runs
  now require it; the policy and diffusion model receive identical customer embeddings.
- [x] Keep the single-HGS label rule. Document that `paper_cmd` retrains must use the
  manifest-backed unfiltered dataset, not the filtered split used by these probes.
- [x] Update tests for both rules. Also replaced the policy's locked legacy `skipped_posterior`
  sampler with the corrected `posterior_mixture_v2`. Recorded in the contract amendment.

**F9 — Classical baseline freeze proposal**

- [x] Choose the prior for each size from panel route quality, not F1 alone. Per-size champions
  are kept; no depot-aware joint prior decodes better routes.
- [x] Record the policy-versus-diffusion decision per size from existing matched evidence. Policy
  at every size; at N100 the 16-start policy (primary seed).
- [x] Write an artifact and hash manifest, and list every `baseline-v1.0` criterion that remains
  open. See [the freeze proposal](classical_baseline_freeze_proposal_2026-09-30.md). The freeze
  itself still needs a commit of this work, which is the owner's decision.

**F10 — Depot-aware champion recipe (added 2026-10-01 at the owner's request)**

Goal: test whether the depot improves the *baseline* prior, using the per-size champion recipe
(per-size data, stochastic audited references, LayerNorm, distance edges, route-gap checkpoint
selection) instead of the paper recipe. Declared before any result:

- Two arms differing only in `coordinate_frame` (absolute control versus `depot_relative`), each
  with its own freshly pretrained GAT from the champion GAT recipe; sizes 20 / 50 / 100; training
  seeds 4331 and 4332; everything else identical to the 2026-08-16 champion configs.
- Score each selected checkpoint on its size's 24 panel graphs: 50 steps, `posterior_mixture_v2`,
  graph-ID-derived sampling seeds, decoded mean route gap, plus F1 and recall.
- Adopt the depot arm as the new prior for a size only if its paired route-gap improvement over the
  control has a 95% bootstrap interval excluding zero for both training seeds, and its mean gap
  also beats the existing champion's. A new prior would still require retraining that size's policy.

- [x] Run the 12 trainings and score them (`outputs/f1gap_diagnostics_20260930/champ/run_champ.sh`).
  The seed-4331 controls reproduce the existing champions exactly (F1 and gap at every size).
- [x] Apply the rule per size and record the result here and in the freeze proposal.

| Size | Seed | Control gap % | Depot gap % | Paired change (pp), 95% CI | Rule |
|---|---:|---:|---:|---:|---|
| 20 | 4331 | 21.3 | 18.9 | −2.4 [−8.3, +3.1] | no |
| 20 | 4332 | 23.2 | 14.4 | −8.8 [−12.3, −5.4] | pass |
| 50 | 4331 | 29.6 | 21.9 | −7.7 [−11.1, −4.2] | pass |
| 50 | 4332 | 26.3 | 25.9 | −0.5 [−4.3, +3.6] | no |
| 100 | 4331 | 28.5 | 25.0 | −3.6 [−6.7, −0.3] | pass |
| 100 | 4332 | 29.1 | 28.7 | −0.4 [−4.4, +3.9] | no |

**Decision: no size meets the declared rule, so the existing champions remain the prior.** Every
size passes for one seed only. The depot arm nonetheless lowers the gap in all six size–seed
pairs and raises F1 in all six. A post-hoc, seed-averaged paired comparison (exploratory, not
the adoption rule) gives −5.6 pp [−9.5, −2.1] at N20, −4.1 pp [−7.0, −1.1] at N50 and −2.0 pp
[−5.1, +1.2] at N100. Seed variance of the champion recipe (checkpoint selection by route gap on
five validation graphs per epoch) is as large as the effect. A confirmation with three more seeds
at N20 and N50, declared in advance, is the cheapest way to settle adoption (about 1.5 hours).

**F11 — Confirmation of the depot-aware champion recipe at N20 and N50 (declared 2026-10-01,
before any run)**

- New training seeds 4333, 4334 and 4335; both arms (absolute control, depot-relative); sizes 20
  and 50; the same GATs and configs as F10 otherwise. Seeds 4331 and 4332 are not reused, because
  they motivated this test.
- Score as in F10 (24 panel graphs per size, 50 steps, graph-ID-derived sampling seeds).
- Adopt the depot-aware recipe for a size only if (a) the per-graph paired route-gap change
  (depot minus control), averaged over the three new seeds, has a 95% bootstrap interval entirely
  below zero; (b) the depot arm has the lower mean gap in at least two of the three new seeds; and
  (c) its three-seed mean gap is below the existing champion's.
- If adopted, the new prior for that size is the primary-seed (4331) depot model from F10, matching
  the freeze proposal's convention of not selecting seeds on the development panel. Its policy must
  then be retrained against the new prior.

- [x] Run the 12 trainings and score them (`outputs/f1gap_diagnostics_20260930/champ/run_confirm.sh`).
- [x] Apply the rule and record the result (`analyze_confirm.py`, 2026-10-01).

| Size | Seed | Control gap % | Depot gap % |
|---|---:|---:|---:|
| 20 | 4333 | 18.9 | 17.7 |
| 20 | 4334 | 16.9 | 24.0 |
| 20 | 4335 | 26.4 | 18.0 |
| 50 | 4333 | 22.6 | 21.5 |
| 50 | 4334 | 27.7 | 23.6 |
| 50 | 4335 | 24.5 | 22.9 |

| Size | Three-seed paired change (pp), 95% CI | Depot wins | Depot mean vs champion | Decision |
|---|---:|---:|---:|---|
| 20 | −0.84 [−4.78, +2.82] | 2 / 3 | 19.9 vs 21.3 | Keep champion (fails a) |
| 50 | −2.28 [−3.99, −0.53] | 3 / 3 | 22.7 vs 29.6 | **Adopt** |

The new N50 prior is the primary-seed depot model from F10:
`outputs/f1gap_diagnostics_20260930/champ/runs/champ_n50_dep_s4331_20261001T082938994407Z/checkpoints/best.pt`
(sha256 `5dc6a986…ec9c8d`; 21.9% panel gap, F1 0.591), with its depot-relative GAT
`champ_gat_dep_20261001T074144597419Z`. The N50 policy has not yet been retrained against it.

**F12 — N50 policy against the new depot-aware prior (declared 2026-10-01, before any run)**

- Two arms trained with the existing N50 policy recipe (`policy_reinforce_s7799_n50_heldout_cuda`)
  but with the corrected prior sampler (`posterior_mixture_v2`, 50 steps) during training: the old
  champion N50 prior (control) and the adopted depot-aware N50 prior. Seeds 42, 43 and 44.
- Score every run, and the existing frozen N50 policy, identically: the 24 N50 panel graphs, 16
  greedy starts, prior regenerated per graph with `posterior_mixture_v2` (50 steps, graph-ID seed).
- Adopt the new-prior policy if (a) the per-graph paired gap change (new minus control, averaged
  over the three seeds) has a 95% bootstrap interval entirely below zero, (b) the new arm has the
  lower mean gap in at least two of three seeds, and (c) its three-seed mean gap is below the
  existing frozen N50 policy's gap under the same protocol. If adopted, the frozen N50 policy is the
  seed-42 new-prior run; otherwise the existing N50 policy and prior stay frozen.

F12 result (2026-10-01): **keep the existing N50 policy and prior.**

| Seed | Control (old prior) gap % | New-prior gap % |
|---:|---:|---:|
| 42 | 20.84 | 21.56 |
| 43 | 21.74 | 20.70 |
| 44 | 21.85 | 21.02 |

Three-seed paired change −0.39 pp [−1.48, +0.76] fails (a); wins 2 of 3 (b holds); mean 21.09%
versus the existing policy's 21.51% (c holds). The depot-aware prior improves diffusion-only
decoding (F11) but not the policy pipeline, which compensates for prior quality. The frozen
baseline therefore keeps the existing per-size policies with their original champion priors.
