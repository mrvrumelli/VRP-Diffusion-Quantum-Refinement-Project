# Ablation Report: Constraint Matrix M and Diffusion

This smoke ablation uses the committed strong-reference matrix examples and fixed seeds. It evaluates the routing utility of different route-membership matrix sources with the same matrix-to-route decoder, so it isolates the value of `M` itself before full policy training scale.

- Data: `outputs/label_audit/s7799_strong_reference/accepted_matrix_examples`
- Dataset hash: `3f15738a9e47c7f10136cc3e5c77ae4298686b69903a44da0f19f94b1dcab497`
- Seed: `42`
- Train/val/test counts: {'train': 12, 'val': 6, 'test': 12, 'train_per_size': 4, 'val_per_size': 2, 'test_per_size': 4}
- Git commit: `ea054468968d7a3351c25708c613ec07a2189f9a`
- CSV: `docs/assets/m_source_ablation/ablation_table.csv`
- Metrics JSON: `docs/assets/m_source_ablation/ablation_metrics.json`

## Results

| method | matrix_f1 | matrix_auc | route_gap_percent | delta_gap_vs_no_m | route_feasible_rate | route_num_vehicles | threshold | runtime_seconds | f1_n20 | f1_n50 | f1_n100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_m_mask | 0.1923 | 0.5000 | 79.5775 | 0.0000 | 1.0000 | 7.0833 | 0.5000 | 0.0001 | 0.3898 | 0.2462 | 0.1697 |
| ground_truth_m | 1.0000 | 1.0000 | 0.5016 | 79.0759 | 1.0000 | 7.0833 | 0.5000 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |
| predicted_m | 0.5305 | 0.8890 | 54.4162 | 25.1613 | 1.0000 | 12.5000 | 0.6500 | 0.0033 | 0.4151 | 0.5703 | 0.5253 |
| random_m | 0.0980 | 0.4954 | 213.3129 | -133.7354 | 1.0000 | 20.5833 | 0.5000 | 0.0009 | 0.1783 | 0.0778 | 0.0989 |
| supervised_m | 0.2025 | 0.5656 | 167.8420 | -88.2645 | 1.0000 | 25.7500 | 0.1500 | 1.0223 | 0.3613 | 0.2516 | 0.1822 |
| diffusion_m | 0.0000 | 0.5431 | 379.6752 | -300.0977 | 1.0000 | 56.6667 | 0.5000 | 0.4593 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

- Oracle `M` reduces route gap by 79.08 percentage points versus no `M` mask.
- Supervised `M` increases route gap by 88.26 percentage points versus no `M` mask.
- Diffusion `M` increases route gap by 300.10 percentage points versus no `M` mask.
- Diffusion vs supervised: matrix F1 delta -0.2025, route-gap delta -211.83 percentage points.

The oracle row is the upper bound for this matrix decoder. The no-mask and random rows show what is lost when the route-membership structure is absent or uninformative. The supervised and diffusion rows show whether the learned matrix sources recover that oracle value in this small self-contained run.

## Caveat

This is a CPU-friendly smoke report, not the final paper-scale ablation. The diffusion arm trains a tiny linear-encoder denoiser from scratch on the selected split because no full trained diffusion checkpoint is present in this checkout.
