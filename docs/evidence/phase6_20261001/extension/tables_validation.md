#### validation: final gap % after convergence (rounds capped at 30)

| Solver | Reorder / exchange size / per round | N20 | N50 | N100 | All | Change vs default limits | Solver s per graph |
|---|---|---:|---:|---:|---:|---:|---:|
| Classical, one pass | 5 / 8 / 10 | 1.97 | 10.99 | 16.38 | 9.34 | — | 2 |
| Classical, one pass | 7 / 12 / 10 | 2.05 | 10.81 | 15.01 | 8.91 | -0.43 [-0.97, +0.12] | 5 |
| Classical, one pass | 10 / 16 / 10 | 2.05 | 10.06 | 14.41 | 8.47 | -0.87 [-1.54, -0.21] | 5 |
| Classical, one pass | 5 / 8 / 30 | 1.97 | 11.14 | 15.29 | 9.07 | -0.27 [-0.57, +0.03] | 3 |
| Classical, one pass | 10 / 16 / 30 | 2.05 | 10.02 | 13.37 | 8.15 | -1.19 [-1.93, -0.45] | 8 |
| Classical, 10 restarts | 5 / 8 / 10 | 1.59 | 6.41 | 11.61 | 6.20 | — | 35 |
| Classical, 10 restarts | 7 / 12 / 10 | 1.45 | 5.52 | 8.92 | 5.05 | -1.14 [-2.05, -0.26] | 88 |
| Classical, 10 restarts | 10 / 16 / 10 | 1.49 | 4.47 | 9.59 | 4.89 | -1.31 [-2.36, -0.32] | 88 |
| Classical, 10 restarts | 5 / 8 / 30 | 1.59 | 6.37 | 9.15 | 5.47 | -0.73 [-1.42, -0.13] | 45 |
| Classical, 10 restarts | 10 / 16 / 30 | 1.49 | 3.78 | 7.70 | 4.10 | -2.10 [-3.30, -0.96] | 113 |
| SA on QUBO | 5 / 8 / 10 | 2.46 | 7.09 | 12.45 | 6.99 | — | 40 |
| SA on QUBO | 7 / 12 / 10 | 1.96 | 7.67 | 12.21 | 6.95 | -0.04 [-0.71, +0.63] | 53 |
| SA on QUBO | 10 / 16 / 10 | 2.18 | 7.69 | 13.49 | 7.41 | +0.41 [-0.39, +1.27] | 55 |
| SA on QUBO | 5 / 8 / 30 | 2.46 | 6.78 | 10.42 | 6.29 | -0.70 [-1.32, -0.17] | 39 |
| SA on QUBO | 10 / 16 / 30 | 2.18 | 7.72 | 11.24 | 6.77 | -0.23 [-1.00, +0.61] | 82 |

#### validation: gap % after each round, default limits

| Solver | Before | Round 1 | Round 2 | Round 3 | Round 5 | Round 10 | Converged | Mean rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical, one pass | 18.33 | 11.64 | 10.06 | 9.64 | 9.43 | 9.34 | 9.34 | 3.6 |
| Classical, 10 restarts | 18.33 | 10.52 | 8.54 | 7.38 | 6.57 | 6.21 | 6.20 | 4.8 |
| SA on QUBO | 18.33 | 11.10 | 9.49 | 8.38 | 7.42 | 7.01 | 6.99 | 5.0 |
