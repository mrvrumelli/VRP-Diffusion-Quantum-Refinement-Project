# Compute-matched stochastic-reference probe

## Design

The probe preloads all 2,304 competitive candidate examples from
`data/processed/s7799_audit_policy_v1` (dataset SHA-256
`74473db53dd5072b8a0fd6f57a1982d5d2ac2059c20ec3483c56a9343256a4b1`). For every epoch it
selects exactly one binary candidate for each of 1,500 sources using SHA-256 of
`(seed, epoch, source_id)`. Input order cannot affect the selection. Both forward noising and the
loss consume the same selected hard `CVRPExample`, and each epoch performs 94 optimizer steps,
matching the policy-v2 source weight and update budget.

The bounded five-epoch run used
`configs/train/diffusion_denoiser_s7799_stochastic_probe_cuda.yaml`. Its fixed panel contains five
frozen validation examples per CVRP size, exact 700-step stochastic sampling, threshold 0.5, and
sampling seed 24331. The selected epoch-0 checkpoint SHA-256 is
`d579ea40abdbd6d5bcc5520bed3881554d39344675d39eebdd1700eb1784f38d`.

## Checkpoint-selection evidence

| Epoch | Noisy-time AUC | Exact F1 | Decoded cost gap |
|---:|---:|---:|---:|
| 0 | 0.8972 | 0.5498 | **43.37%** |
| 1 | 0.9128 | 0.5306 | 62.46% |
| 2 | 0.9188 | 0.4383 | 88.03% |
| 3 | 0.9236 | 0.4576 | 58.00% |
| 4 | **0.9278** | 0.4308 | 56.55% |

Noisy-time AUC would select epoch 4; decoded route cost correctly selects epoch 0. This is direct
evidence that noisy-time metrics cannot serve as the sole checkpoint rule.

## Matched policy-v2 comparison

The prior policy-v2 checkpoint (SHA-256
`8622a564c45088612a1501c0e2b688e740f69e867796aa65406d13d816fdc87a`) was evaluated on the
same 15 instance IDs with the same exact sampler and seed.

| Metric | Policy-v2 | Stochastic epoch 0 | Change |
|---|---:|---:|---:|
| Overall exact F1 | 0.4254 | **0.5498** | +0.1245 |
| Overall decoded gap | 76.59% | **43.37%** | -33.21 pp |
| CVRP20 exact F1 | 0.4133 | **0.4778** | +0.0645 |
| CVRP50 exact F1 | 0.4240 | **0.5683** | +0.1443 |
| CVRP100 exact F1 | 0.4268 | **0.5490** | +0.1222 |
| CVRP20 decoded gap | 70.13% | **62.63%** | -7.50 pp |
| CVRP50 decoded gap | 82.69% | **39.72%** | -42.97 pp |
| CVRP100 decoded gap | 76.93% | **27.77%** | -49.16 pp |

All decoded solutions in both arms were capacity-feasible. The probe passes the stated gate of a
CVRP100-specific gain without CVRP20/CVRP50 regression on this fixed panel. It remains a small
validation probe, not permission to evaluate the untouched test set or adopt stochastic targets
without the planned consensus/mask controls and replication.

## Consensus controls

The same five-epoch protocol was then run with consensus probabilities as the clean loss target.
Forward noising continued to use the deterministic per-epoch hard reference. The masked arm used
`abs(2p - 1)` pair weights, giving unanimous pairs weight 1 and evenly disputed pairs weight 0.

| Validation-selected arm | Exact F1 | Decoded gap | N20 gap | N50 gap | N100 gap |
|---|---:|---:|---:|---:|---:|
| Stochastic hard reference | 0.5498 | **43.37%** | 62.63% | **39.72%** | **27.77%** |
| Consensus | 0.4753 | 46.29% | **55.06%** | 45.27% | 38.54% |
| Consensus + confidence mask | **0.5501** | 52.04% | 78.04% | 46.61% | 31.48% |

Checkpoint SHA-256 values are, respectively,
`d579ea40abdbd6d5bcc5520bed3881554d39344675d39eebdd1700eb1784f38d`,
`c9314a0caa4ecc961524113e56d4b7d86c2931864c419e024ecdf64aeaa1f7f6`, and
`5ffaf5d65d3054042f25bb85d95167cb84c7bf527da83d9a41601faeb077c20f`.

The masked arm recovers matrix F1 but not routing utility. Plain consensus improves CVRP20 decoded
cost relative to stochastic but regresses CVRP50/100. Stochastic hard references remain the best
overall arm and the only ambiguity-aware arm that improves all three sizes over policy-v2 on this
panel. The exclusion control and multi-seed replication remain required before adopting it.

## Stable-only exclusion control

The declared matrix-stability gate retains 1,211 sources: 500 CVRP20, 472 CVRP50, and 239
CVRP100. Repeating retained sources would silently increase their weight, so this control preserves
one contribution per retained source and reports its lower compute: 76 optimizer steps per epoch
versus 94 for the 1,500-source arms. It is source-weight matched, not optimizer-step matched.

The epoch-0 checkpoint (SHA-256
`8865692fbdf1fa8d229936e53dd74ad336b50cbcb28bd1d48adb2669e3fbeec7`) achieved exact F1
0.5468 and decoded gap 38.72%, with per-size gaps of 56.45%, 32.08%, and 27.62%. This is the best
small-panel routing result, but it uses 19% fewer sources and updates. It therefore motivates a
proper matched-budget replication; it does not prove exclusion is superior to stochastic targets.
