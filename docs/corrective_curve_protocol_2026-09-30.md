# Bounded source-count confirmation protocol

Declared before the R10 runs. The small R7–R9 probes did not establish a replacement recipe:
depot conditioning was mixed, unweighted BCE improved F1 but worsened routing, and BatchNorm
was unstable. Retain the existing robust architecture as the scaling control: frozen pretrained
GAT, width 192, eight denoiser layers, LayerNorm, absolute customer coordinates, weighted BCE
power 0.5, stochastic reference selection, T=700. This retains a known missing-depot limitation;
it is a control experiment, not a claim that depot information is unnecessary.

Use corrected `posterior_mixture_v2`, 50 inference steps, threshold 0.5 and one generated sample.
Compare **500 versus 1,000 distinct N100 sources**, retaining every eligible reference in each
source pool. Seeds: 4331 and 4332. Train each for exactly five epochs, batch size 4, gradient
accumulation 4, learning rate 0.0003, mixed precision, gradient clipping 1.0. Five epochs is a
bounded short-training curve, not a repeat of the historical 15-epoch curve. Save/update counts
and runtime; this is fixed epochs, not fixed optimizer steps.

Use the **final checkpoint** for comparison, irrespective of the trainer's saved `best.pt`.
Selection panel: the 24 N100 entries of `outputs/corrective_confirmation_20260930/panel.json`,
fixed before training, outside all compared denoiser/GAT training sources. Validation sampling
seed 9001. Pair graph-level comparisons by ID and report both gap definitions and generated F1.

Practical route improvement threshold: **1 percentage point**. Expand data sizes or epochs only
if the 1,000-source point improves mean instance-relative gap by at least 1 point in **both**
training seeds, and each paired 95% graph-bootstrap interval for the difference excludes zero.
Otherwise stop this scaling branch as inconclusive, reversed, or plateaued as appropriate;
do not assert equality from overlapping intervals. No more than these four runs in this stage.

Only a finalist that passes that rule may proceed to the 96 reserved test sources, and only
after their reference-quality limitation is addressed. If no finalist qualifies, preserve the
untouched test and report that outcome. Original classical and paper freezes remain gated.
