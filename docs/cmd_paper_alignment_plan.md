# CMD paper-alignment and robustness plan

## Purpose

This plan aligns the project with *Constraints Matrix Diffusion based Generative Neural Solver for
Vehicle Routing Problems* by Zhenwei Wang, Tiehua Zhang, Ning Xue, Ender Ozcan, Ling Wang, and
Ruibin Bai.

The safest approach is to preserve two explicit tracks:

- `paper_cmd`: a faithful, frozen reconstruction of the paper baseline;
- `ours_robust`: the current architecture and audited-label extensions.

Do not gradually modify the current model until it resembles the paper. That would make it
impossible to attribute improvements to individual changes. Paper results are external reference
targets until the `paper_cmd` implementation is reproduced locally on fixed benchmark instances.

Primary references:

- Paper: <https://arxiv.org/pdf/2603.07568>
- arXiv record: <https://arxiv.org/abs/2603.07568>
- Official XML100 benchmark: <https://galgos.inf.puc-rio.br/cvrplib/en/xml100>

As of 2026-08-28, the arXiv record contains only version 1 and no author implementation has been
identified. Exact reproduction therefore requires explicitly documenting assumptions wherever the
paper is underspecified.

## Assessment

Alignment difficulty is **moderate at the code level but high at the experimental level**.

- Most data structures, diffusion primitives, feasibility logic, decoder mechanics, and training
  infrastructure already exist.
- The global/local encoder path and accelerated sampler need meaningful implementation changes.
- The paper does not provide exact dataset seeds, the HGS time budget, or a fully specified skipped
  transition.
- The paper trained on an NVIDIA RTX A6000 with 48 GB, while recorded project runs use an RTX 3060
  Ti with 8 GB.
- The target protocol uses 50,000 labelled diffusion examples, 200,000 RL instances, and up to 100
  RL epochs. Compute is therefore a major part of reproduction difficulty.

Estimated effort for one engineer:

| Work | Difficulty | Estimate |
|---|---|---:|
| Paper-compatible configuration and augmentation | Low-medium | 3-5 days |
| Correct skipped diffusion sampler | Medium-high | 4-7 days |
| Frozen GAT, masked GAT, and paper fusion | Medium | 5-8 days |
| Dataset and benchmark preparation | Medium | 5-8 days plus label generation |
| Diffusion and policy training | High compute risk | 1-3 elapsed weeks |
| Reproduction, ablations, and robustness study | High | 1-2 weeks |
| **Likely total** | **Moderate-high** | **5-9 elapsed weeks on current hardware** |

Access to an A6000-class GPU and clarification from the authors could reduce the elapsed time.

## Robustness assessment

The current version may be more robust in particular failure modes, but there is no evidence yet
that it is more robust overall.

### Where the current version may be stronger

- The learned fusion gate preserves a full global path when the predicted matrix is unreliable.
  This is a plausible defense against mask errors.
- Soft adjacency is available as a more forgiving alternative to a hard `M_hat` threshold.
- Multi-seed label auditing and stochastic or consensus targets explicitly handle route-partition
  ambiguity. The paper treats one HGS partition as the binary answer.
- Per-size denoisers substantially improved matrix-decoded routing results over the pooled
  denoiser.
- Feasibility remains protected by explicit decoder or repair masks rather than depending on
  matrix correctness.

### Where the current version may be weaker

- The global encoder is trainable and Transformer-based. The paper reports that fine-tuning its
  global GAT slightly improved IID CVRP100 but worsened CVRPLIB average gap from 5.10% to 5.52%, so
  it freezes the GAT for OOD robustness.
- Per-size denoisers may be strong at CVRP20/50/100 but brittle for unseen sizes. They exchange
  scale generalization for specialization.
- Default hard thresholding can change the local graph abruptly under small probability errors.
- The approximate skipped sampler currently loses quality, making it less robust to inference
  budget reduction.
- The R/C/RC stress datasets have been validated as distinct distributions, but a trained
  end-to-end policy has not yet been evaluated on them.

The paper reports actual OOD evidence on XML100: a times-eight mean gap of 5.77% with standard
deviation 4.02. It also identifies weaknesses for clustered customers, uniform demands, extreme
route lengths, and centered depots. Until the local policy is evaluated on the same 10,000
instances, the defensible working hypothesis is:

> The current design may be more robust to noisy labels and inaccurate matrices, but it is more
> exposed to cross-distribution and cross-size overfitting.

## Phase 0 - Freeze the comparison contract

**Estimate:** 2-3 days.

**Status (2026-09-24): complete.** The versioned operating contract is documented in
[`cmd_paper_comparison_contract.md`](cmd_paper_comparison_contract.md). Named machine-readable
profiles live in `configs/alignment/paper_cmd.yaml` and `configs/alignment/ours_robust.yaml`; the
complete evidence classification and author-question register live in
`configs/alignment/cmd_paper_evidence.yaml`. Config validation freezes paper-compatible choices,
and checkpoint consumers reject missing, legacy, or cross-track provenance in `paper_cmd` runs.

Create two named configurations:

- `paper_cmd`: only paper-compatible choices;
- `ours_robust`: the current architecture and audited-label extensions.

Record every paper detail as one of:

- explicitly stated;
- mathematically inferred;
- ambiguous and awaiting author clarification.

Questions for the authors should cover exact seeds, HGS budget, GAT pretraining objective,
augmentation enumeration, skipped-transition formula, test-set generation seeds, and whether
models are trained separately by size.

**Gate:** No experiment may be labelled `paper_cmd` if it silently uses an `ours_robust`
component.

## Phase 1 - Restore a clean repository gate

**Estimate:** 1-2 days.

**Status (2026-09-24): complete.** A fresh Python 3.12 environment installed the project and
development extras successfully with PyVRP 0.14.0 and CPU-only PyTorch 2.14.0. `pip check`,
`ruff check .`, and `ruff format --check .` pass. The explicit `paper_smoke` test passes on CPU,
and the complete clean-environment suite reports 489 passed with four CUDA-only tests skipped.
Commands, versions, and results are recorded in
[`phase1_clean_repository_gate_2026-09-24.md`](phase1_clean_repository_gate_2026-09-24.md).

- Recreate the environment with PyVRP 0.14 or newer.
- Format the two currently failing files.
- Require `ruff check`, `ruff format --check`, and the complete `pytest` suite to pass.
- Add a paper-mode smoke test.

**Gate:** Clean installation and complete test suite on CPU.

## Phase 2 - Paper-compatible data and augmentation

**Estimate:** 3-5 engineering days plus HGS runtime.

- Implement paper training augmentation separately from the existing times-nine augmentation.
- Add four label-conditioned demand transforms.
- Use geometric-only times-eight augmentation at inference.
- Generate 50,000 HGS-labelled diffusion instances.
- Generate or stream 200,000 unlabeled RL instances.
- Freeze 1,000 IID test instances for each of CVRP20, CVRP50, and CVRP100.
- Record seeds, manifests, solver budgets, and dataset hashes.
- Keep the audited and stochastic-reference datasets as later experimental arms.

**Gate:** Every transform preserves feasibility and its intended constraint matrix; all splits have
hashes and no overlap.

## Phase 3 - Faithful diffusion path

**Estimate:** 5-8 days.

Implement a paper configuration with:

- `T=1000`, `beta_1=1e-4`, and `beta_T=0.02`;
- embedding dimension 128;
- five-layer, eight-head pretrained GAT;
- BatchNorm where specified;
- batch size 32 and 50 diffusion epochs;
- paper-compatible denoiser depth and input features.

Replace the current stride approximation with a transition defined for arbitrary `s < t`. Tests
must show:

- the skipped posterior reduces to the existing posterior when `s = t - 1`;
- probabilities match brute-force enumeration on toy binary chains;
- symmetry and zero diagonal survive every sampled transition;
- results are deterministic under fixed seeds.

**Gate:** Reproduce the paper's qualitative inference-step curve, particularly approximately F1
0.823 at 50 steps, before connecting the prior to the policy.

## Phase 4 - Faithful encoder-decoder path

**Estimate:** 5-8 days.

**Status (2026-09-24): implemented and covered by automated gates.** The `paper_cmd` policy now
loads either a standalone pretrained GAT or the `node_encoder` embedded in a diffusion checkpoint,
stores the exact frozen weights inside policy checkpoints, and restores without depending on the
original GAT file. Its separate local GAT is masked by `M`, and its fusion is sum followed by an
MLP. The existing `ours_robust` Transformer and learned-gate path remain unchanged. Focused tests
cover exact checkpoint equality, zero global-GAT gradients and parameter drift, a cost-lowering
tiny RL run with 100% feasibility, and the no-`M`, no-local-encoder, and no-local-pointer modes.

Add `architecture: paper_cmd` to the policy:

- load the exact pretrained diffusion GAT;
- freeze and verify all its parameters;
- use it as the global encoder;
- add a trainable masked GAT for local representations;
- implement paper fusion as sum followed by MLP;
- retain the dual local/global pointers, savings bias, NStart, and feasibility masks.

Do not delete the current Transformer encoder or learned gate.

**Gate:**

- A checkpoint round trip proves the global policy GAT equals the diffusion GAT.
- Frozen parameters receive no gradients.
- Tiny RL training lowers cost while maintaining 100% feasibility.
- No-`M`, no-local-encoder, and no-local-pointer modes work through configuration.

## Phase 5 - Train the paper baseline

**Estimate:** 1-3 elapsed weeks on the available GPU.

Run in increasing cost order:

1. CVRP20 convergence pilot.
2. Small joint versus per-size training comparison.
3. Full diffusion training.
4. Full 200,000-instance policy training.
5. Three-seed confirmation for the selected configuration.

Address memory pressure by reducing batch size with gradient accumulation, not by reducing NStart,
because reducing NStart changes the learning objective.

**Gate:** The validation policy clearly improves over no-`M` and random-`M` controls under matched
compute.

## Phase 6 - Build the exact evaluation suite

**Estimate:** 5-8 days.

- Synthetic: 1,000 fixed examples each for CVRP20/50/100.
- CVRPLIB: the exact 17 instances used in the paper.
- XML100: all 10,000 instances and 378 groups.
- Respect CVRPLIB distance rounding and provided optimal values.
- Add HGS and POMO-compatible comparison outputs.

The official XML100 benchmark provides instances, optimal solutions, a generator, category
encoding, and guidance to test every method on the same fixed 10,000 instances.

**Gate:** A hand-checked subset matches official costs and category assignments before full
evaluation.

## Phase 7 - Reproduce paper results and ablations

**Estimate:** 1-2 weeks.

Required results:

- paper CMD without augmentation;
- paper CMD with times-eight inference;
- Gaussian versus Bernoulli diffusion;
- no masked encoder;
- no local pointer;
- frozen versus fine-tuned global GAT;
- inference steps 1, 5, 10, 20, 50, 200, and 1,000.

Report cost, gap, runtime, feasibility, vehicle count, seed, configuration, and confidence
intervals.

If exact test seeds remain unavailable, call the result a **paper-guided reconstruction**, not an
exact reproduction.

**Gate:** Results are reasonably close to the paper across all three IID sizes and preserve its
main ablation ordering.

## Phase 8 - Determine whether our version is more robust

**Estimate:** 4-7 days after both policies exist.

Use identical training data, seeds, instances, and inference budgets. Introduce one change at a
time:

| Arm | Change from paper baseline |
|---|---|
| P0 | Faithful paper CMD |
| P1 | Learned gated fusion only |
| P2 | Transformer global encoder only |
| P3 | LayerNorm only |
| P4 | Audited or stochastic labels only |
| P5 | Per-size denoisers only |
| P6 | Soft adjacency only |
| Ours | All selected extensions |

Evaluate on:

- IID synthetic tests;
- the full XML100 benchmark;
- the 17 CVRPLIB instances;
- explicit mask-corruption sweeps at 0%, 5%, 10%, and 20% pair flips;
- stable versus ambiguous label subsets;
- cross-size tests;
- R/C/RC spatial stress sets.

Primary robustness metrics:

- mean, standard deviation, median, p90, and p95 cost gap;
- worst XML100 category gap;
- feasibility and vehicle count;
- sensitivity to mask corruption;
- matrix calibration and F1;
- runtime and memory;
- paired bootstrap confidence intervals.

A robustness extension should only be accepted if it improves OOD mean or tail performance under
matched compute, maintains 100% feasibility, and does not cause a material IID regression.

## Phase 9 - Freeze the baseline

Tag `baseline-v1.0` only when:

- the faithful paper path is reproducible;
- benchmark datasets and outputs are hashed;
- paper ablations have been reproduced;
- the paper-versus-ours robustness study is complete;
- every result identifies whether it is `paper_cmd` or `ours_robust`.

Only after this gate should quantum-refinement work begin.

## Required paper-reference targets

These values are external targets until reproduced locally:

| Size | CMD, no augmentation | CMD, times-eight inference |
|---|---:|---:|
| CVRP20 | 6.20, gap 1.64% | 6.14, gap 0.65% |
| CVRP50 | 10.64, gap 2.70% | 10.41, gap 0.48% |
| CVRP100 | 15.76, gap 1.61% | 15.70, gap 1.23% |

Additional targets:

- CVRPLIB average gap: 5.10%;
- XML100 average gap: 8.73% without augmentation and 5.77% with times-eight inference;
- XML100 gap standard deviation: 7.07 without augmentation and 4.02 with times-eight inference;
- constraint-matrix F1 at 50 inference steps: approximately 0.823.

## Completion rule

Check a phase only when its output is reproducible from committed code and configuration and its
artifacts include seeds, dataset hashes, runtime, feasibility, costs, vehicle counts, metrics, and
the resolved configuration.
