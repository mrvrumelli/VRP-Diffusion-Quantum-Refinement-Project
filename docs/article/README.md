# Solver architecture article

[main.tex](main.tex) is a complete, self-contained LaTeX article about the
implemented CVRP solver, inspired by *Constraints Matrix Diffusion based
Generative Neural Solver for Vehicle Routing Problems*. It centers the
project's diffusion prior, global and local Transformers, learned gated
fusion, capacity-aware dual-pointer decoding, and multi-start reinforcement
learning. Classical and QUBO refinement are extensions of that architecture.

The article includes related work, mathematical definitions, training settings,
constructive and refinement results, paired comparisons, runtime analysis,
limitations, an embedded bibliography with 11 references, and an appendix
mapping claims to repository artifacts. It contains seven main tables, an
artifact table, and two self-contained schematic figures. Existing author
names are retained. No new experiments were run for this writing task.

## Build

Upload `main.tex` to Overleaf as the main document, or run from `docs/article`
with a standard TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The second pass resolves references. Alternatively, run `latexmk -pdf main.tex`.
No BibTeX invocation, external image files, project datasets, or Python
packages are needed for the article build.

## Evidence and verification

Paths below are relative to the repository root. Numerical results are from
project experiments; external papers supply methodological context.

| Content | Evidence |
| --- | --- |
| Actual implemented architecture | `src/vrp_diffusion_quantum/models/{diffusion,constraint_denoiser,global_encoder,local_masked_encoder,fusion_encoder,decoder}.py` |
| Diffusion and policy training objectives | `src/vrp_diffusion_quantum/train/{train_diffusion,train_policy}.py` |
| Frozen hyperparameters and constructive results | `docs/evidence/baseline_freeze_20261001/freeze_manifest.json` and `configs/` |
| Policy versus direct diffusion decoding | `docs/classical_baseline_freeze_proposal_2026-09-30.md` |
| Final panels, configurations, and declared comparisons | `docs/phase6_final_test_protocol_2026-10-02.md` |
| Final reference gaps and intervals | `docs/evidence/phase6_20261001/final/analysis_{reserved,ood}.json` and run summaries |
| Fixed-subproblem comparisons | `docs/evidence/phase6_20261001/experiment_{development,validation}/summary.json` and SQA counterparts |
| QAOA screening and matched random control | `docs/evidence/phase6_20261001/qaoa_random_control_{development,validation}.json` |
| Refinement runtime | `docs/evidence/phase6_20261001/timing/` summaries, using `mean_wall_seconds` |
| Matrix bias, neighborhood selection, and convergence | `docs/evidence/phase6_20261001/{bias_sweep,selection,extension}/` and `docs/phase6_refinement_experiments_2026-10-01.md` |
| QUBO objectives, reconstruction, and acceptance | `src/vrp_diffusion_quantum/quantum/{qubo_reorder,qubo_exchange,qubo_bias,refinement}.py` |

Static checks passed for balanced braces and environments, unique labels,
resolved citation/reference targets, and whitespace. The 162 refinement
numeric entries excluding the five baseline cells were checked against the
archived JSON; construction gaps and panel counts were separately recomputed
from the freeze manifest. Architecture descriptions and training settings
were checked against implementation files and copied configurations. The
materialized training pools were also checked to contain 500 distinct sources
per size, with 500/536/1,268 retained partitions and capacities 30/40/50.
Bibliographic information was verified using primary papers and publisher
records.

No LaTeX compiler is installed in the writing environment, so PDF compilation
and page layout remain unverified.

The article evaluates our implementation. It does not include the separate
paper-alignment test track. It distinguishes heuristic-reference gaps from
certified optimality gaps, graph uncertainty from training-seed uncertainty,
matched neighborhood limits from equal runtime, and simulated quantum methods
from hardware execution. Positive matrix bias was evaluated on development
and validation panels; the final SA and QAOA arms use zero objective bias
while retaining matrix-guided construction and neighborhood selection.
