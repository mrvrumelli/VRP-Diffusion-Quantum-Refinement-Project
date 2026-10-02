#### development: final gap % after convergence (rounds capped at 30)

| Solver | Reorder / exchange size / per round | N20 | N50 | N100 | All | Change vs default limits | Solver s per graph |
|---|---|---:|---:|---:|---:|---:|---:|
| Classical, one pass | 5 / 8 / 10 | 2.14 | 10.21 | 13.74 | 8.70 | — | 2 |
| Classical, one pass | 7 / 12 / 10 | 1.85 | 9.73 | 13.57 | 8.38 | -0.32 [-0.67, +0.04] | 4 |
| Classical, one pass | 10 / 16 / 10 | 1.85 | 10.05 | 12.58 | 8.16 | -0.54 [-1.09, +0.07] | 6 |
| Classical, one pass | 5 / 8 / 30 | 2.14 | 10.10 | 12.84 | 8.36 | -0.33 [-0.68, -0.03] | 5 |
| Classical, one pass | 10 / 16 / 30 | 1.85 | 9.72 | 12.09 | 7.89 | -0.81 [-1.31, -0.24] | 9 |
| Classical, 10 restarts | 5 / 8 / 10 | 1.08 | 6.17 | 10.18 | 5.81 | — | 34 |
| Classical, 10 restarts | 7 / 12 / 10 | 0.81 | 4.35 | 9.46 | 4.87 | -0.94 [-1.49, -0.39] | 56 |
| Classical, 10 restarts | 10 / 16 / 10 | 0.77 | 4.63 | 8.78 | 4.72 | -1.09 [-1.70, -0.48] | 94 |
| Classical, 10 restarts | 5 / 8 / 30 | 1.08 | 6.05 | 9.05 | 5.39 | -0.42 [-0.85, -0.06] | 61 |
| Classical, 10 restarts | 10 / 16 / 30 | 0.77 | 4.37 | 7.54 | 4.23 | -1.59 [-2.24, -0.94] | 143 |
| SA on QUBO | 5 / 8 / 10 | 2.31 | 6.45 | 10.62 | 6.46 | — | 34 |
| SA on QUBO | 7 / 12 / 10 | 1.63 | 6.53 | 10.14 | 6.10 | -0.36 [-1.08, +0.33] | 50 |
| SA on QUBO | 10 / 16 / 10 | 1.70 | 7.77 | 11.98 | 7.15 | +0.69 [-0.02, +1.41] | 65 |
| SA on QUBO | 5 / 8 / 30 | 2.31 | 6.34 | 9.52 | 6.05 | -0.40 [-0.70, -0.15] | 56 |
| SA on QUBO | 10 / 16 / 30 | 1.70 | 7.65 | 11.17 | 6.84 | +0.38 [-0.37, +1.13] | 71 |

#### development: gap % after each round, default limits

| Solver | Before | Round 1 | Round 2 | Round 3 | Round 5 | Round 10 | Converged | Mean rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical, one pass | 18.48 | 11.20 | 9.96 | 9.26 | 8.92 | 8.70 | 8.70 | 3.8 |
| Classical, 10 restarts | 18.48 | 10.11 | 8.07 | 7.07 | 6.19 | 5.83 | 5.81 | 5.1 |
| SA on QUBO | 18.48 | 10.54 | 8.57 | 7.51 | 6.81 | 6.47 | 6.46 | 4.8 |
