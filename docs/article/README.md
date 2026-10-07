# Initial article draft

[main.tex](main.tex) is a deliberately simple, self-contained LaTeX article. It
explains the project pipeline and emphasizes the completed quantum refinement
benchmark. Its two principal contributions are the conversion of the diffusion
route-membership matrix into local QUBO coefficients and comparative
benchmarking against classical refinement algorithms. It contains five result
tables, explicit matrix-derived exchange and reorder terms, paired
confidence intervals, limitations, and a short bibliography. Author names and
affiliations are placeholders. No new experiments were run for this draft.

## Build

Upload `main.tex` to Overleaf as the main document, or run these commands from
`docs/article` with a standard TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The second pass resolves references. The bibliography is embedded; BibTeX,
external figures, project data, and Python dependencies are not required.
An alternative is `latexmk -pdf main.tex`.

All 167 numeric table entries were checked against the archived JSON, and the
final aggregate reports were cross-checked against each run summary. Static
checks found balanced LaTeX environments and braces and no missing reference
or citation targets. No LaTeX compiler was available in the drafting
environment, so PDF compilation and page layout have not been verified.

## Evidence used

Paths below are relative to the repository root. Benchmark numbers are project
results, not numbers from the external papers in the bibliography.

| Draft content | Source |
| --- | --- |
| Frozen pipeline, reference construction, baseline runtime | `docs/classical_baseline_freeze_proposal_2026-09-30.md`, freeze record dated 2026-10-01; `docs/evidence/baseline_freeze_20261001/freeze_manifest.json` and `configs/` |
| Final sets, arms T1–T10, comparison rules | `docs/phase6_final_test_protocol_2026-10-02.md` |
| Tables 1–2: final gaps and paired intervals | `docs/evidence/phase6_20261001/final/analysis_reserved.json`, `analysis_ood.json`, and corresponding run `summary.json` files |
| Table 3: fixed-subproblem hit rates | `docs/evidence/phase6_20261001/experiment_{development,validation}/summary.json` and `experiment_{development,validation}_sqa/summary.json` |
| Table 4: QAOA and matched random control | `docs/evidence/phase6_20261001/qaoa_random_control_{development,validation}.json` |
| Table 5: separate timing pass | `docs/evidence/phase6_20261001/timing/{t1_t10,t2_t4_t5,t3,t6_t7,exact_small}/summary.json`, field `mean_wall_seconds` |
| Bias, selection, formulation checks, convergence, warm-start discussion | `docs/phase6_refinement_experiments_2026-10-01.md`, sections 3–4, and supporting `docs/evidence/phase6_20261001/` archives |
| QUBO and repair definitions | `src/vrp_diffusion_quantum/quantum/{qubo_reorder,qubo_exchange,qubo_bias,refinement,qaoa_solver}.py` |
| Boundary between project extension and paper reconstruction | `docs/cmd_paper_comparison_contract.md` |

Numbers use the archived precision (usually two decimals for gaps and one for
hit rates and timing). Table 4 uses the random-control archive's common
best-of-decoded-samples scoring, including retention of the original solution;
the older single-best-energy-sample screening metric is different. Table 3
keeps the SQA rerun separate because time-limited SA can perform different
numbers of reads under different machine loads.

The prose deliberately distinguishes reference gaps from optimality gaps,
graph uncertainty from seed uncertainty, equal neighborhood limits from equal
runtime, and simulation from quantum hardware. T3 is the best tested
configuration but uses larger neighborhoods than QAOA; T6 versus T7/T8/T10
provides the relevant small-neighborhood comparison. An interval containing
zero is described as inconclusive rather than proving a tie.

Evidence for the positive-weight matrix-to-QUBO formulation comes from the
development and validation bias ablations. Final-test SA and QAOA use zero
objective bias while retaining matrix-guided neighborhood selection. The draft
therefore distinguishes evaluation of the proposed formulation from the final
comparison of refinement operators. The contribution is scoped to the
matrix-to-QUBO coupling; broader priority claims would require a related-work
review.

## Before submission

Replace the author placeholders, choose a venue template, and expand related
work as appropriate. A submission should also give the complete training and
dataset specification, software versions and CPU model, and an accessible
artifact release. These are editorial or reporting additions, not unreported
positive results. The present draft does not claim quantum advantage or
state-of-the-art CVRP performance.
