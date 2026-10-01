# Why paper_cmd diffusion F1 is 0.47 instead of 0.823 — 2026-09-30

This investigates the gap between the corrected full `paper_cmd` checkpoint (0.4682 pooled F1 on
the 72-graph expanded development panel) and the paper's Figure 8 value (0.823 at 50 steps).
All numbers are development diagnostics on that frozen panel unless stated otherwise. Scripts and
raw outputs are in `outputs/f1gap_diagnostics_20260930/` (gitignored).

Status: final as of 2026-10-01, after the follow-up tasks in
[autonomous_f1gap_followup_tasks_2026-09-30.md](autonomous_f1gap_followup_tasks_2026-09-30.md).

## Verdict

**The reconstruction had a real error: the model never saw the depot.** The GAT and the denoiser
received only customer `[x, y, demand, demand/capacity]`, although the paper applies the GAT to the
instance graph, which includes the depot. With a uniformly random depot, route membership depends
heavily on where the depot is. A solver-based experiment shows that no depot-blind predictor can
exceed about 0.65 pooled F1, so 0.823 was unreachable. This explains why 11x more data and longer
training did nothing: the depot-blind model had learned everything its input allowed within one
epoch. The error is now fixed in code and in the `paper_cmd` contract.

**Fixing it is necessary but not sufficient.** The corrected paper reconstruction, trained at the
paper's scale (50,000 unfiltered single-HGS labels, 50 epochs, depot-node GAT, corrected sampler),
reaches 0.531 pooled F1 at 50 steps and 0.560 at its best step count of 10, up from 0.475 and 0.506.
Unlike the depot-blind run, it kept improving across all 50 epochs, so data and training now
matter, but slowly. Under the most generous comparable definition (macro F1, 0.737) and on
accuracy (0.895 versus the paper's 0.95) it still falls clearly short.

**The remaining gap is a broad model belief, not the sampler.** After many steps the chain returns
one committed partition, and that sample scores below the best one-step classifier (about 0.60).
Calibration fixes, a deterministic chain, final-threshold changes, GAT fine-tuning and the
DIFUSCO-style objective were all tested; none produced the paper's rising step curve or closed the
gap. The paper's 0.823 implies a much sharper model, and what would make ours that sharp is not
identified.

**The target itself is underspecified.** The same outputs score from 0.47 to 0.90 depending on the
F1 definition, and the paper's validation split construction is unknown. Near-optimal N100
partitions are not unique on unfiltered labels. The remaining difference cannot be attributed
further without the authors' metric definition, validation protocol, or code.

**Route quality tracks recall, not pooled F1.** Pooled F1 is dominated by N100 pairs and hides
per-size behaviour. Per size, the depot-blind per-size champions match or beat every depot-aware
prior on F1 at N20 and N50 and decode better routes at every size. Across all priors, the route gap
falls as recall rises: missing same-route pairs fragments routes into extra depot round trips,
which the decoder cannot undo, while extra pairs are partly repaired by capacity splitting. F5's
N20 prior has recall 0.34 and an 80% route gap. In matched comparisons on the same recipe and data,
adding the depot improved routes as well as F1, so the depot is not the cause of F5's worse routes.
The classical baseline proposal keeps the champions
(see [the freeze proposal](classical_baseline_freeze_proposal_2026-09-30.md)).

## Evidence 1 — the depot-blind ceiling

For each of the 72 panel instances, customers, demands and capacity were held fixed and the depot
was moved to 32 uniform random positions. Each variant was solved with PyVRP (1 s / 4 s / 20 s for
N20 / N50 / N100). Averaging the 32 route-membership matrices gives the pair probability that an
ideal depot-blind model could know. Thresholding it gives the best depot-blind classifier; a single
random-depot solution gives the expected score of a perfect depot-blind sampler.

| Size | Best depot-blind classifier F1 | Perfect depot-blind sampler F1 | Full checkpoint, 50 steps | Same-depot re-solve F1 |
|---|---:|---:|---:|---:|
| 20 | 0.710 | 0.624 | 0.542 | 1.000 |
| 50 | 0.677 | 0.548 | 0.529 | 0.916 |
| 100 | 0.633 | 0.489 | 0.442 | 0.795 |
| Pooled | 0.651 | 0.514 | 0.468 | — |

The last column re-solves the *original* depot with the same budget. It is below 1.0 at N50/N100,
so solver noise slightly depresses those two rows; N20 has no such noise and shows the same ceiling.

The classifier column is a hard ceiling for any depot-blind model under this data distribution.
The sampler column is the expected score of an *exact* sampler from the depot-averaged partition
distribution, not a hard cap: a model that samples more conservatively can score higher. The
repository's depot-blind per-size champions reach 0.623 / 0.584 / 0.518, above the sampler value at
N50 but below the classifier ceiling everywhere. The full checkpoint sits below both. The ceiling
explains why no amount of data, epochs or steps could reach 0.823 without the depot.

Code locations: `build_customer_node_features` in `models/gat_encoder.py` (customer features only)
and `customer_tensors_from_batch` in `train/train_diffusion.py` (depot dropped, absolute frame).
The paper applies the GAT to the instance graph `G` and reuses the same frozen GAT as the policy's
global encoder, which must embed the depot. The project's `paper_cmd` policy does feed the depot
into that GAT, but the GAT was pretrained without ever seeing one.

## Evidence 2 — the model learned proximity and nothing more

| Predictor on the same panel | N20 | N50 | N100 |
|---|---:|---:|---:|
| Symmetric k-nearest-neighbour graph, best k (5 / 8 / 10) | 0.602 | 0.585 | 0.564 |
| Depot-aware sweep heuristic | 0.601 | 0.527 | 0.473 |
| Full checkpoint, one-shot from pure noise, threshold 0.5 | 0.611 | 0.558 | 0.487 |
| Full checkpoint, 50-step chain | 0.542 | 0.529 | 0.442 |

A one-parameter geometric heuristic matches or beats the trained diffusion model. The depot-blind
GAT pretraining head plateaus at validation F1 0.56 and AUC 0.90 within one epoch and stays there
for 27 epochs.

## Evidence 3 — the denoiser itself works; the prior is the bottleneck

Teacher-forced x0 prediction from true forward-noised inputs (eval mode, threshold 0.5, pooled):

| Noise step t | 0 | 50 | 100 | 150 | 200 | 300 | 500 | 999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model F1 | 0.991 | 0.954 | 0.837 | 0.682 | 0.591 | 0.517 | 0.510 | 0.510 |
| Copying the noisy input | 0.999 | 0.874 | 0.657 | 0.472 | 0.348 | 0.227 | 0.179 | 0.177 |

The network denoises well whenever the input still carries the answer. Once the input is mostly
noise it falls back to the feature-only prior (about 0.51), which is the depot-blind proximity prior.
For this depot-blind checkpoint the chain never receives information beyond that prior, so
iterative refinement has nothing to converge to.

## Evidence 4 — why more steps lower F1

The paper's curve rises from 0.616 at one step to 0.823 at 50. Ours falls:

| Steps | 1 | 10 | 50 | 200 |
|---|---:|---:|---:|---:|
| Production chain | 0.509 | 0.506 | 0.475 | 0.468 |
| Weighted-BCE bias removed inside the chain | — | 0.467 | 0.444 | 0.431 |
| Bias removed inside the chain and at the final threshold | — | 0.441 | 0.423 | 0.408 |

Training uses `pos_weight = neg/pos` (power 1.0), and the chain does drift. The F1-optimal threshold
on teacher-forced predictions is 0.8–0.9 for N100, and the final N100 matrices mark 15.3% of pairs
as same-route against a true 9.1%. However, subtracting the exact prior-ratio logit
(`log(neg/pos)` per size) makes F1 worse at every step count. This agrees with the earlier
held-out intercept calibration, which also lowered F1. With an uninformative prior, inflated
probabilities act like a lower decision threshold, which helps F1.

The depot-blind ceilings in Evidence 1 explain the direction of the curve. A one-shot thresholded
marginal is a classifier (ceiling 0.651 pooled). A long chain returns one consistent sample from the
model's joint belief (ceiling 0.514 pooled). When that belief cannot localize routes, more steps
push the output from the first regime toward the second. Evidence 6 and 7 test whether depot
information sharpens the posterior enough to reverse this. It flattens the curve and moves the peak
to about 10 steps, but does not make it rise to 50 steps as in the paper.

## Evidence 5 — the metric definition is underspecified

The same 72-graph outputs under different definitions:

| Definition | Score |
|---|---:|
| Positive-class F1, pooled off-diagonal (project metric) | 0.468 |
| Mean of per-graph F1 | 0.505 |
| Positive-class F1 counting the trivial diagonal | 0.517 |
| Macro F1 over both classes | 0.694 |
| Class-weighted F1 | 0.872 |
| Accuracy | 0.861 |

The paper's Figure 6 reports F1 0.82 next to accuracy 0.95, so its F1 is not class-weighted. Our
N100 accuracy (0.867) is below the all-zeros predictor (0.909), so a real gap exists whatever the
definition.

## Evidence 6 — matched depot-aware runs

GAT pretraining with the `paper_cmd` settings on the filtered full split, seed 42, 8 epochs. Runs
differ only in the depot input. They were run on the project track because the current
`paper_cmd` contract rejects the filtered split. The customers-only arm reproduces the original
2026-09-28 GAT run to four decimals at every epoch.

| GAT input | Validation AUC | Validation F1 (adaptive threshold) | Final train loss |
|---|---:|---:|---:|
| Customers only (current reconstruction) | 0.900 | 0.561 | 0.681 |
| Depot-relative coordinates (`coordinate_frame: depot_relative`) | 0.914 | 0.582 | 0.636 |
| Depot-relative plus distance and angle to depot | 0.915 | 0.584 | 0.635 |

The customers-only run never exceeded F1 0.565 in 27 epochs. Both depot arms pass it by epoch 3
and are still rising at epoch 8. Explicit polar features learn faster but converge to the same
level as translated coordinates.

A 12-epoch diffusion run then used the depot-relative GAT with every other `paper_cmd` diffusion
setting unchanged, including the frozen GAT. Both checkpoints were scored on the 72-graph panel
with the corrected sampler and identical seeds. Values are pooled F1, averaged over two sampling
seeds.

| Diffusion checkpoint | 1 step | 10 steps | 50 steps |
|---|---:|---:|---:|
| Customers only (full checkpoint, 2026-09-28) | 0.509 | 0.506 | 0.475 |
| Depot-relative, selected epoch 1 | 0.532 | 0.533 | 0.513 |
| Depot-relative, final epoch 11 | 0.534 | 0.531 | 0.509 |

Per size at 50 steps, the gain is concentrated at N50 (0.53 to 0.57) and N100 (0.45 to 0.49).
Teacher-forced validation F1 rose from 0.63 to 0.66 and AUC from 0.93 to 0.95 over training.

Supplying the depot is necessary but not sufficient. It moves generated F1 by about 0.035, which
only reaches the exact depot-blind sampler value, and more steps still lower F1.

Teacher-forced F1 on the panel shows where the depot helps (threshold 0.5, pooled):

| Noise step t | 100 | 150 | 200 | 300 | 999 |
|---|---:|---:|---:|---:|---:|
| Customers only | 0.837 | 0.682 | 0.591 | 0.517 | 0.510 |
| Depot-relative, epoch 11 | 0.915 | 0.802 | 0.676 | 0.552 | 0.534 |

The depot-aware model denoises much better once the input carries signal, but its prediction from
near-pure noise improves only slightly. A reverse chain commits to that weak prediction in its
early high-noise steps; the later steps sharpen it rather than correct it. The remaining
bottleneck is therefore the quality of the pure-noise prediction. The paper's own one-step F1 of
0.616 is close to this level (our best-threshold value is 0.606), so the unexplained part is how
the paper's chain climbs from there to 0.823.

Fine-tuning the depot-relative GAT during diffusion training (the paper states freezing only for
the policy) was tested with every other setting unchanged. Selected checkpoints, same panel and
seeds:

| Diffusion checkpoint | 1 step | 10 steps | 50 steps |
|---|---:|---:|---:|
| Depot-relative, frozen GAT (epoch 1) | 0.532 | 0.533 | 0.513 |
| Depot-relative, fine-tuned GAT (epoch 11) | 0.540 | 0.550 | 0.522 |

Teacher-forced validation F1 rose to 0.68 and AUC to 0.95. The 50-step gain is +0.007 and +0.012
on the two sampling seeds, below the pre-declared adoption rule of +0.01 on both, so the follow-up
probes keep the frozen GAT. It is recorded as a borderline improvement.

Adding pairwise distance to the denoiser's edge input (the robust-track option; the paper's
edge input is the noisy matrix only) was the next probe, on the frozen depot-relative parent:

| Diffusion checkpoint | 1 step | 10 steps | 50 steps |
|---|---:|---:|---:|
| Depot-relative, frozen GAT | 0.532 | 0.533 | 0.513 |
| Plus distance edge features | 0.553 | 0.557 | 0.535 |

The 50-step gain is +0.025 and +0.019 by seed, so distance edges were adopted as the parent for
the next probe. Together with the depot, the project-track recipe now gains 0.06 over the original
checkpoint at 50 steps, but the step curve still does not rise.

A DIFUSCO-faithful bundle was then applied on top of that parent: unweighted BCE, LayerNorm,
bit-flip probability β/2, and residual edge updates.

| Diffusion checkpoint | 1 step | 10 steps | 50 steps |
|---|---:|---:|---:|
| Distance-edge parent | 0.553 | 0.557 | 0.535 |
| Plus DIFUSCO bundle | 0.477 | 0.452 | 0.448 |

The bundle gave the best teacher-forced numbers of any probe (validation AUC 0.952, F1 0.685),
but generation fell sharply, most at N20 (0.31), and decoded route gaps exceeded 100%. It was
rejected. A threshold sweep on the final clean-probability map explains why:

| Pooled F1, sampling seed 0 | Threshold 0.5 | Best threshold |
|---|---:|---:|
| Distance-edge parent, 1 step | 0.554 | 0.603 (0.8) |
| Distance-edge parent, 50 steps | 0.538 | 0.541 (flat) |
| DIFUSCO bundle, 1 step | 0.478 | 0.593 (0.2) |
| DIFUSCO bundle, 50 steps | 0.459 | 0.462 (flat) |

After 50 steps the final prediction is near-binary, so the threshold hardly matters: the chain
returns one committed sample. For both models that sample scores below the best one-step
classifier (about 0.60). The sampler is therefore not the bottleneck. The model's belief about the
partition is still broad, and a sample from a broad belief matches one reference poorly. Weighted
BCE merely biases samples toward larger route clusters, which happens to score better. The
paper's 0.823 implies a far sharper model than any 12-epoch probe here.

Keeping weighted BCE and applying only LayerNorm, β/2 noise and residual edges gave 0.535, 0.535
and 0.536 at 1, 10 and 50 steps: a flat step curve instead of a falling one, but no gain at 50
steps (+0.001 / +0.002 by seed). N20 fell from 0.61 to 0.51 with more steps while N100 rose from
0.51 to 0.53. Not adopted.

Finally, the depot was added as an explicit GAT graph node, the form closest to the paper's
`GAT(G)`. Its GAT pretraining matched the depot-relative GAT (validation F1 0.583 versus 0.582),
and the three depot encodings tried (translated coordinates, explicit polar features, depot node)
all converge to the same pretraining level. On top of the distance-edge parent the depot-node
diffusion probe scored 0.544 / 0.545 / 0.516 at 1 / 10 / 50 steps, below the parent at 50 steps
(−0.021 / −0.017 by seed), so it was not adopted for the project track. It remains the required
form for the paper track because it matches the paper's graph definition.

| Probe (12 epochs, filtered split) | 1 step | 10 steps | 50 steps | Adopted |
|---|---:|---:|---:|---|
| Customers only (full checkpoint) | 0.509 | 0.506 | 0.475 | — |
| Depot-relative, frozen GAT | 0.532 | 0.533 | 0.513 | Yes |
| Plus GAT fine-tuning | 0.540 | 0.550 | 0.522 | No (borderline) |
| Plus distance edges | 0.553 | 0.557 | 0.535 | Yes |
| Plus DIFUSCO bundle | 0.477 | 0.452 | 0.448 | No |
| Plus DIFUSCO bundle, weighted BCE kept | 0.535 | 0.535 | 0.536 | No |
| Plus depot as graph node | 0.543 | 0.545 | 0.516 | No |

## Evidence 7 — full-length corrected paper reconstruction (F5)

The amended `paper_cmd` recipe was trained at the paper's stated scale: a depot-node GAT pretrained
for 50 epochs on the manifest-backed, unfiltered single-HGS split (50,000 labels, balanced across
sizes, disjoint from the panel), then the frozen-GAT BatchNorm denoiser with noisy-matrix edges,
weighted BCE and the corrected sampler for 50 epochs. Training was paused after epoch 14 and resumed
from the saved optimizer state; the merged history is in
`outputs/f1gap_diagnostics_20260930/f5_merged_summary.csv`. Early stopping never triggered. The
selected checkpoint is epoch 43 (best in-training sample F1, 0.546).

| Pooled panel F1, two sampling seeds | 1 step | 10 steps | 50 steps |
|---|---:|---:|---:|
| Original full checkpoint (customers only, 2026-09-28) | 0.509 | 0.506 | 0.475 |
| F5, corrected reconstruction at full scale | 0.546 | 0.560 | 0.531 |

| F5 by size | 1 step | 10 steps | 50 steps |
|---|---:|---:|---:|
| N20 | 0.632 | 0.540 | 0.481 |
| N50 | 0.574 | 0.587 | 0.560 |
| N100 | 0.529 | 0.553 | 0.525 |

Over training, teacher-forced validation F1 rose from 0.64 to about 0.70 and AUC from 0.947 to about
0.963, while in-training sample F1 rose from 0.50 to about 0.54. The depot-blind full run, by
contrast, peaked at epoch 1. With the depot, more data and training do help, but the gain over 50
epochs is about 0.05 F1 at 50 steps. The step curve now rises from 1 to 10 steps before falling, and
it falls hardest at N20.

The same outputs under other definitions (graph-ID seeding, 50 steps):

| Definition | F5 | Paper (Figure 6) |
|---|---:|---:|
| Positive-class F1, pooled | 0.532 | 0.82 |
| Macro F1 over both classes | 0.737 | — |
| Accuracy | 0.895 | 0.95 |
| Precision / recall | 0.503 / 0.565 | 0.91 / about 0.75 (implied by its F1 and precision) |

Decoded routes from the F5 prior are worse than both the original checkpoint and the per-size
champions: mean gaps of 79.9% / 50.6% / 42.8% at N20 / N50 / N100 (pooled 57.8%, versus 43.1% for
the original checkpoint and 21–30% for the champions).

## Ruled out or minor

- **Label noise in training/panel data.** Accepted training labels agree across the two PyVRP
  seeds at pooled F1 0.9997 / 0.9965 / 0.9652 (N20 / N50 / N100). The panel uses accepted canonical
  labels for N50/N100 and near-optimal short solves for N20.
- **Data distribution.** Capacities 30/40/50, integer demands 1–9, uniform coordinates and a random
  depot match the paper.
- **BatchNorm running statistics.** For this checkpoint eval mode is as good as or better than
  batch-statistics mode at every noise level (for example 0.991 versus 0.930 at t=0). This does not
  rule out BatchNorm problems in other runs.
- **Sampler mathematics and step count.** Already corrected; more steps lower F1 (Evidence 4).
- **Post-hoc calibration of weighted BCE.** Exact prior-ratio correction lowers F1 (Evidence 4).
- **DIFUSCO deviations.** β/2 noise, LayerNorm and residual edges, with or without unweighted BCE,
  were tested on a depot-aware parent (Evidence 6). Unweighted BCE badly hurt generation; the rest
  changed 50-step F1 by about 0.001.
- **Deterministic reverse chain.** Thresholding each step instead of sampling peaks at 5 steps
  (0.524) and collapses to 0.356 at 50 steps.
- **Final decision threshold.** After 50 steps the output is near-binary; thresholds from 0.1 to 0.8
  change F1 by under 0.01.
- **Depot encoding.** Translated coordinates, explicit polar features and a depot graph node reach
  the same GAT pretraining level.
- **Data scale once the depot is present.** It helps, slowly: about +0.05 F1 at 50 steps over the
  full 50-epoch, 50,000-label run (Evidence 7).

## Caveat on the target itself

On the *unfiltered* 50k audit, two 40 s PyVRP runs on the same N100 instance agree at only 0.75
pooled F1 (0.92 at N50), with a mean cost difference of 0.24%. Near-optimal CVRP100 partitions are
not unique. If the paper's validation labels were single unfiltered HGS solutions, N100 F1 above
about 0.75–0.8 would be hard for any model, and the paper's pooled 0.875 at 1,000 steps would
imply small-size-weighted aggregation, a different F1 definition, or overlap between augmented
training and validation instances. None of these can be checked without the authors.

## Recommended next steps

1. Ask the authors the open ledger questions, above all the Figure 8 F1 definition, the validation
   split construction, and whether the GAT is fine-tuned during diffusion training. Without these,
   further reconstruction work cannot tell a modelling gap from a measurement difference.
2. Do not spend more GPU time chasing 0.823 on this recipe. The depot fix is done, the full-scale
   corrected run is measured, and every tested inference and objective change is exhausted.
3. Select priors for the classical baseline by decoded route quality, not matrix F1, and proceed
   with the freeze proposal and the gated quantum refinement work.
