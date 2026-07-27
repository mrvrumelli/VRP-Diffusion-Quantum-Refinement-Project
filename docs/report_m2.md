# Phase 2 — Supervised Constraint-Matrix Predictor

## Objective

De-risk discrete diffusion by first learning the customer-customer route-membership matrix
directly from Phase 1 labels. This phase does not attempt route decoding and makes no claims
about diffusion or quantum improvement.

## Method

`MatrixPredictor` is a pairwise MLP over customer coordinates, pairwise distance, demands, and
capacity. Its output is symmetrized, constrained to `[0, 1]`, and given an exact zero diagonal.
Training minimizes off-diagonal binary cross-entropy against `M_true`.

Examples are shuffled with the configured seed and partitioned into non-overlapping
train/validation/test splits. Model selection metrics are computed only on validation examples.
An optional test partition is evaluated once after training.

## Baselines

Every held-out model evaluation includes the requested baselines on the exact same examples:

- `nearest_neighbor`: deterministic spatial nearest-neighbor chains converted into explicit,
  transitive route-membership clusters.
- `demand_aware`: angular sweep clusters whose accumulated demands respect capacity.
- `random`: seeded balanced random clusters, using the capacity lower bound for the cluster count.
- `all_zero`: an additional negative control that predicts no same-route pairs.

These are structural `M` baselines, not route solvers.

## Configuration and dataset

The smoke configuration is `configs/train/matrix_predictor_sanity.yaml`. It contains only two
hand-authored examples and exists to test the complete code path. Its metrics are not research
results.

For a real run, set `dataset.path` to the `examples/` directory produced by
`scripts/run_data_pipeline.py`, retain an explicit seed, and use enough examples for separate
training, validation, and test partitions.

## Metrics and outputs

Each run records:

- training and validation BCE;
- held-out ROC-AUC, precision, recall, F1, expected calibration error, and capacity consistency;
- the same metrics for all heuristic baselines;
- split sizes, seed, learning rate, runtime, feasibility rate, and dataset hash;
- `model.pt` and held-out predicted-versus-ground-truth plots.

The exact numeric results are stored in the run's `metrics.json`; generated run outputs are not
committed.

## Data quality results

Pending the substantive Phase 1 dataset. The Phase 1 pipeline is already configured to record,
per CVRP size:

- instance count and label feasibility rate;
- customer-demand mean and range;
- mean total demand;
- capacity mean and range;
- mean number of vehicles.

## Label generation speed

Pending the substantive Phase 1 label run. The pipeline records per-size mean, median, P95, and
total solver runtime, in addition to full run wall time and solver configuration.

## Supervised M predictor results

Pending the substantive Phase 2 run. The results table will compare the supervised predictor,
nearest-neighbor clusters, demand-aware clusters, seeded random clusters, and the all-zero control
on the exact same held-out examples.

Once both runs finish, generate the complete result report directly from their logged artifacts:

```bash
python scripts/build_phase2_report.py \
  --phase1-run data/raw/cvrp/<phase1-run> \
  --phase2-run outputs/train/<phase2-run>
```

This writes `docs/report_m2_results.md` without manually copying metrics.

## Failure cases and limitations

- Pairwise predictions need not be transitively consistent.
- Class imbalance can make all-zero accuracy misleading, so accuracy is intentionally not the
  primary metric.
- Capacity consistency is a proxy on thresholded connected components, not final route
  feasibility.
- The pairwise MLP has no global route decoder and is only a supervised pre-diffusion baseline.
- The two-example sanity dataset cannot support meaningful generalization conclusions.

## Next steps

Generate the substantive Phase 1 dataset, run the frozen Phase 2 configuration, build the results
report, inspect at least 20 qualitative plots, and use those results to decide whether Phase 3
diffusion is justified.
