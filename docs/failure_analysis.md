# Failure Analysis

This report documents representative failure cases for the current checkout.
The analysis uses committed strong-reference examples from
`outputs/label_audit/s7799_strong_reference/accepted_matrix_examples` and the
same deterministic split used by `scripts/run_m_source_ablation.py`:

- seed: `42`
- sizes: `CVRP20`, `CVRP50`, `CVRP100`
- split: `4 train / 2 validation / 4 test` per size
- reference labels: strong multi-seed PyVRP labels from the audit

No trained policy checkpoint is present in this checkout, so the concrete cases
below focus on decoded constraint-matrix and label-audit failures. That is still
the right surface for this task: these are the errors that feed the policy
through `M_hat`, the local pointer prior, route clustering, and the reference
targets used for training.

## Executive Summary

- The strongest observed failure mode is feasible but very poor routing, not
  invalid route output.
- Infeasible policy rollouts were not observed in the available artifacts. The
  decoder masks already prevent revisiting served customers and selecting
  customers whose demand exceeds remaining vehicle capacity.
- Poor clusters show up as many false-negative route-pair decisions: customers
  that should share a route are split apart.
- Bad capacity decisions usually appear as vehicle inflation. Several decoded
  cases use 20, 50, or 100 routes where the reference uses 4 to 11 vehicles.
- Long-horizon errors are concentrated on CVRP100. The same matrix mistakes are
  much more expensive when 100 sequential customer decisions have to stay
  coherent.

## Representative Failure Cases

The table uses actual decoded cases from the deterministic ablation panel plus
two label-audit target-instability cases. `FP/FN` are false-positive and
false-negative pair decisions against the reference route-membership matrix.
`repairs` counts capacity-gate repair events in the matrix-to-route decoder.

| case | failure type | source | instance | n | gap % | vehicles ref | repairs | pair acc | FP/FN | diagnosis |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| F01 | long-horizon over-split | diffusion M | `cvrp_s7799_n20-50-100_x66667_cvrp100_1279` | 100 | 630.47 | 100/10 | 0 | 0.899 | 0/498 | Diffusion sampled an all-singleton partition; every customer became its own route. |
| F02 | long-horizon over-split | diffusion M | `cvrp_s7799_n20-50-100_x66667_cvrp100_7088` | 100 | 612.67 | 100/10 | 0 | 0.904 | 0/476 | Same singleton collapse, with nearly all true same-route edges missing. |
| F03 | long-horizon over-split | diffusion M | `cvrp_s7799_n20-50-100_x66667_cvrp100_6168` | 100 | 545.99 | 100/11 | 0 | 0.911 | 0/439 | Matrix pair accuracy looks high because negatives dominate, but route structure is unusable. |
| F04 | bad capacity decision | diffusion M | `cvrp_s7799_n20-50-100_x66667_cvrp50_37090` | 50 | 372.13 | 50/7 | 0 | 0.861 | 0/170 | The decoded matrix keeps routes feasible by splitting every customer separately. |
| F05 | poor random clusters | random M | `cvrp_s7799_n20-50-100_x66667_cvrp100_7088` | 100 | 310.35 | 34/10 | 0 | 0.886 | 102/463 | Random edges create small disconnected clusters and miss most useful route memberships. |
| F06 | poor random clusters | random M | `cvrp_s7799_n20-50-100_x66667_cvrp100_1279` | 100 | 298.77 | 30/10 | 0 | 0.879 | 119/479 | Random false positives add noisy merges, but false negatives still dominate. |
| F07 | capacity-gated poor merge | supervised M | `cvrp_s7799_n20-50-100_x66667_cvrp100_6168` | 100 | 256.39 | 47/11 | 1 | 0.835 | 440/378 | The learned matrix proposes bad merges, then capacity repair blocks them and leaves many fragments. |
| F08 | capacity-gated poor merge | supervised M | `cvrp_s7799_n20-50-100_x66667_cvrp100_42699` | 100 | 253.59 | 49/11 | 1 | 0.844 | 410/361 | Capacity gate saves feasibility, but the resulting partition has too many tiny routes. |
| F09 | capacity-gated over-split | supervised M | `cvrp_s7799_n20-50-100_x66667_cvrp50_43831` | 50 | 203.86 | 26/7 | 1 | 0.774 | 141/136 | The decoder hits a capacity conflict and stops with a fragmented route set. |
| F10 | dense-mask poor clustering | no M mask | `cvrp_s7799_n20-50-100_x66667_cvrp100_1279` | 100 | 102.61 | 10/10 | 1 | 0.821 | 444/442 | Removing `M` preserves vehicle count but mixes customers into geographically poor clusters. |
| F11 | heuristic under-cluster | predicted M | `cvrp_s7799_n20-50-100_x66667_cvrp20_59781` | 20 | 107.37 | 10/4 | 1 | 0.732 | 9/42 | Capacity-distance heuristic is too conservative, producing many extra routes on a small instance. |
| F12 | heuristic under-cluster | predicted M | `cvrp_s7799_n20-50-100_x66667_cvrp50_66019` | 50 | 73.22 | 14/6 | 1 | 0.885 | 30/111 | Good pair accuracy still hides too many missed positive route pairs. |
| F13 | ambiguous target cluster | label audit | `cvrp_s7799_n20-50-100_x66667_cvrp20_11228` | 20 | n/a | n/a | n/a | n/a | n/a | Audit rejected matrix stability: 24.2% disagreement from original route membership. |
| F14 | capacity/fleet ambiguity | label audit | `cvrp_s7799_n20-50-100_x66667_cvrp50_17560` | 50 | n/a | +1 vehicle delta | n/a | n/a | n/a | Audit marked cost not converged and route membership ambiguous; vehicle count changed by 1. |

## Failure Mode Notes

### Infeasible Rollouts

No concrete infeasible policy rollouts were found in the checked artifacts. This
is an important negative result, not a missing category. The policy decoder is
designed so infeasibility is guarded at decode time:

- visited customers are masked out;
- customers whose demand exceeds remaining capacity are masked out;
- the depot remains selectable to close a route and reload capacity.

The cases above therefore show the more common current failure: the model or
matrix prior can remain feasible while making routes much worse. The practical
risk is that aggregate feasibility can look perfect while solution quality
collapses.

### Poor Clusters

Poor clustering appears in two opposite forms:

- Dense or noisy matrices merge unrelated customers. F10 keeps the reference
  vehicle count but produces a 102.61% gap because the route composition is bad.
- Sparse matrices miss true same-route pairs. F01 to F04 are extreme examples:
  the singleton partitions have zero false positives but hundreds of false
  negatives.

Pair accuracy is not enough for this diagnosis. CVRP100 has many true negatives,
so a matrix can score around 0.90 pair accuracy while producing unusable routes.
F01 to F03 are the clearest examples.

### Bad Capacity Decisions

Capacity handling is currently effective as a safety mechanism but weak as an
optimization signal. The decoder repairs over-capacity merge attempts and keeps
outputs feasible, but the repair often leaves too many routes:

- F07: 47 routes instead of 11.
- F08: 49 routes instead of 11.
- F09: 26 routes instead of 7.
- F11: 10 routes instead of 4.

This suggests that route-count inflation should be reported beside cost gap and
feasibility in every policy evaluation.

### Long-Horizon Errors

The worst failures are CVRP100 cases. A small bias toward splitting compounds
over 100 customers, and a local mistake early in the partition limits all later
merge options. In F01 to F03, the matrix-level error is simple, but the routing
effect is catastrophic: 545.99% to 630.47% gap.

The label audit also shows that CVRP100 is the least stable target source:
`outputs/label_audit/s7799_strong_reference/metrics.json` reports 239 accepted
matrix targets out of 500 CVRP100 instances, compared with 493/500 for CVRP20
and 472/500 for CVRP50.

## Recommended Follow-ups

1. Save per-instance decoded route diagnostics for every ablation run:
   vehicle delta, route-size histogram, false positives, false negatives,
   repair count, and gap.
2. Add a route-count penalty or calibration target for learned `M` so sparse
   matrices do not collapse into singleton routes.
3. Report positive-edge recall separately from pair accuracy, especially on
   CVRP100 where true negatives dominate.
4. Track capacity-gate repair cases as warnings, even when final routes are
   feasible.
5. Keep CVRP100 as a separate long-horizon slice in checkpoint selection; the
   aggregate score can hide CVRP100 collapse.
6. Add an inference-time guardrail that flags extreme vehicle inflation, for
   example decoded vehicles greater than `1.5x` the reference or an OR lower
   bound estimate.

