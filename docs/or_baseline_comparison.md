# OR Baseline Comparison

This report compares classical OR baselines under matched wall-clock budgets. Each selected instance is solved by every baseline at the same per-instance time limit; quality is measured against the committed strong-reference label cost.

- Data: `outputs/label_audit/s7799_strong_reference/accepted_matrix_examples`
- Dataset hash: `3f15738a9e47c7f10136cc3e5c77ae4298686b69903a44da0f19f94b1dcab497`
- Selected subset hash: `c364507af1807200fc8c23bfc3a418e967d71e63c33f1ba42f09473037a86876`
- Seed: `42`
- Time budgets: `[0.25, 0.75]` seconds per instance
- Selected counts by size: `{20: 2, 50: 2, 100: 2}`
- Git commit: `ba07ecc8bdd3752692aec040c918d318e1d46407`
- Detailed CSV: `eval/or_baseline_comparison.csv`
- Summary JSON: `eval/or_baseline_comparison_summary.json`

## OR Comparison Table

| baseline | time_budget_seconds | size | instances | success_count | feasible_rate | mean_gap_to_reference_percent | mean_gap_to_best_observed_percent | mean_cost | mean_runtime_seconds | mean_runtime_budget_ratio | mean_number_of_vehicles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OR-Tools GLS | 0.2500 | all | 6 | 6 | 1.0000 | 6.7242 | 5.6512 | 11.5682 | 0.2970 | 1.1882 | 7.1667 |
| OR-Tools GLS | 0.2500 | 20 | 2 | 2 | 1.0000 | 0.9683 | 0.9683 | 5.5401 | 0.3651 | 1.4604 | 3.5000 |
| OR-Tools GLS | 0.2500 | 50 | 2 | 2 | 1.0000 | 8.8132 | 7.8131 | 10.9349 | 0.2573 | 1.0291 | 7.5000 |
| OR-Tools GLS | 0.2500 | 100 | 2 | 2 | 1.0000 | 10.3912 | 8.1722 | 18.2296 | 0.2688 | 1.0751 | 10.5000 |
| PyVRP HGS | 0.2500 | all | 6 | 6 | 1.0000 | 0.9959 | 0.0000 | 10.8285 | 0.2674 | 1.0697 | 7.1667 |
| PyVRP HGS | 0.2500 | 20 | 2 | 2 | 1.0000 | 0.0000 | 0.0000 | 5.4835 | 0.2545 | 1.0182 | 3.5000 |
| PyVRP HGS | 0.2500 | 50 | 2 | 2 | 1.0000 | 0.9202 | 0.0000 | 10.1278 | 0.2613 | 1.0451 | 7.5000 |
| PyVRP HGS | 0.2500 | 100 | 2 | 2 | 1.0000 | 2.0676 | 0.0000 | 16.8741 | 0.2864 | 1.1458 | 10.5000 |
| OR-Tools GLS | 0.7500 | all | 6 | 6 | 1.0000 | 5.6186 | 5.0450 | 11.4109 | 0.7593 | 1.0125 | 7.1667 |
| OR-Tools GLS | 0.7500 | 20 | 2 | 2 | 1.0000 | 0.9683 | 0.9683 | 5.5401 | 0.7527 | 1.0036 | 3.5000 |
| OR-Tools GLS | 0.7500 | 50 | 2 | 2 | 1.0000 | 7.5491 | 7.0996 | 10.8075 | 0.7563 | 1.0085 | 7.5000 |
| OR-Tools GLS | 0.7500 | 100 | 2 | 2 | 1.0000 | 8.3386 | 7.0671 | 17.8849 | 0.7690 | 1.0253 | 10.5000 |
| PyVRP HGS | 0.7500 | all | 6 | 6 | 1.0000 | 0.5433 | 0.0000 | 10.7597 | 0.7666 | 1.0221 | 7.1667 |
| PyVRP HGS | 0.7500 | 20 | 2 | 2 | 1.0000 | 0.0000 | 0.0000 | 5.4835 | 0.7529 | 1.0038 | 3.5000 |
| PyVRP HGS | 0.7500 | 50 | 2 | 2 | 1.0000 | 0.4364 | 0.0000 | 10.0764 | 0.7613 | 1.0150 | 7.5000 |
| PyVRP HGS | 0.7500 | 100 | 2 | 2 | 1.0000 | 1.1935 | 0.0000 | 16.7192 | 0.7855 | 1.0474 | 10.5000 |

## Interpretation

- Best overall quality in this run: PyVRP HGS at 0.75s with mean reference gap 0.54%.
- Runtime fairness is enforced by matching the configured budget per solver and instance.
- Actual runtime includes instance conversion and solver setup overhead, so it can exceed the configured solver time limit on short budgets.

## Caveat

This is a smoke-scale OR comparison over a deterministic subset, not a full benchmark sweep. It is meant to make runtime and solution-quality reporting reproducible inside this checkout.
