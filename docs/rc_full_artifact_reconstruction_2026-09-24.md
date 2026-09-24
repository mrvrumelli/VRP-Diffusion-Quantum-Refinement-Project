# `rc_full` artifact reconstruction — 2026-09-24

## Scope and conclusion

The original training-side audit directory, `outputs/label_audit/rc_full`, is absent. Its raw
candidate records, aggregate `summary.csv`/`metrics.json`, solver runtimes, failures, and exact
stable-matrix acceptance counts therefore cannot be truthfully recovered from this checkout.

The surviving materialized-label manifests and completed training artifacts do establish a
substantial, hash-addressed provenance chain. This document archives that recoverable evidence so
the project no longer relies on an absent gitignored directory for claims about what entered the
later `rcfull` training sets. It is a reconstruction, not a replacement for the missing audit
summary.

## Recoverable source identity

- Audited pooled-source SHA-256:
  `626006f44e752c2a7ffe497e34b52f2f8a33a794a2ec9bd8426bfc140664c4fb`.
- Materialization commit recorded by both manifests:
  `69508ab4bd96395abd1b68e213f84af94c7b5a29`.
- Policy-v1 manifest SHA-256:
  `b18b816dfacbfb561a015c64cf1ee119f886cda09ce9bfe7a1014dd586a4c4ff`.
- Policy-v2 manifest SHA-256:
  `72284db0354819e650ce094740f08d97284f65b099aadc2bf45ed48c9eced2bd`.
- Every policy manifest contains 9,000 distinct sources: 1,000 each in every
  `{R,C,RC} x {N20,N50,N100}` cell.

The authoritative surviving files are:

- `data/processed/rc_full_audit_policy_v1/training_label_manifest.json`
- `data/processed/rc_full_audit_policy_v2/training_label_manifest.json`

Each retained label entry records its source file, size, selection mode, route hash, and cost.

## Materialized label policies

Policy v1 used a 0.5% competitive-cost tolerance. It retained the original N20 label,
canonical-else-multiple labels at N50, and multiple competitive references at N100.

| Size | Sources | Labels | Sources with >1 label | Maximum labels/source |
|---:|---:|---:|---:|---:|
| 20 | 3,000 | 3,000 | 0 | 1 |
| 50 | 3,000 | 3,215 | 153 | 4 |
| 100 | 3,000 | 7,759 | 2,461 | 4 |
| **Total** | **9,000** | **13,974** | **2,614** | **4** |

Policy-v1 labels by spatial regime:

| Size | C labels | R labels | RC labels |
|---:|---:|---:|---:|
| 20 | 1,000 | 1,000 | 1,000 |
| 50 | 1,088 | 1,077 | 1,050 |
| 100 | 2,819 | 2,447 | 2,493 |

Policy v2 used zero competitive-cost tolerance while retaining the same size-dependent selection
modes. It materialized 9,007 labels: 3,000 at N20, 3,000 at N50, and 3,007 at N100. Only seven
N100 sources had two exactly cost-competitive route references (three C and four RC); all other
sources contributed one label.

These counts prove materialization and training-label multiplicity. They do **not** prove how many
sources met any original audit-level stability criterion because the materialization policies can
retain an original or canonical candidate even when alternative candidates disagree.

## Merged training sets and completed models

The later per-size merged sets used policy v1, combining the original strongly audited IID labels
with the reconstructed spatial labels:

| Size | IID labels | `rc_full` labels | Total train labels | Dataset SHA-256 |
|---:|---:|---:|---:|---|
| 20 | 500 | 3,000 | 3,500 | `f3a11cf4e8b9204e17dc479c4edc46a4c9c502f1eb5964bf30ebd56bcdb96720` |
| 50 | 536 | 3,215 | 3,751 | `370926320ecfcb1cea439ace609255cbc8178a01f2a0e9177520a63ffc0d772e` |
| 100 | 1,268 | 7,759 | 9,027 | `0843434b67374d7da0a2d819b3041fd30d92becb67bd6435ae1bf060e9c94075` |

All three 15-epoch size-specific runs completed:

| Size | Final validation AUC | Final validation F1 | Runtime (s) | Best-checkpoint SHA-256 |
|---:|---:|---:|---:|---|
| 20 | 0.933904 | 0.631546 | 1,383.75 | `f51ffd3062998e0d6c9aa2c81f380fd54bf80dbcf00e71a8aaf9369ca18eae3d` |
| 50 | 0.948867 | 0.681411 | 1,415.80 | `8b0d6f07db9788cb6815505cb1e7e1c8383020e3e0c93b4e7d3bfc762116343d` |
| 100 | 0.953531 | 0.692338 | 1,737.47 | `eaeebe67ded889bf0faa954504ccb2d229e88b1c48dc65c4ecb66b28fd8ffd90` |

The earlier N20 directory ending in `20260824T193852423107Z` has no final `metrics.json`; the
completed N20 directory ending in `20260824T204918783220Z` supersedes it.

## Separate held-out OOD audit

`outputs/label_audit/rc_full_eval` is a different, independent-seed evaluation audit and must not
be confused with the missing training-side bundle. On 2026-09-24 it contains 18,219 of 36,000
candidate files and no final summary. It remains paused and resumable; it was not started or
modified as part of this no-training/no-long-compute work.

## Still unavailable

Recovering the following requires an external backup of the original `outputs/label_audit/rc_full`
bundle or rerunning that audit under a frozen configuration:

- stable-matrix acceptance counts and rates;
- per-candidate solver seeds, statuses, runtimes, and failure records;
- candidate cost-spread and route-agreement distributions;
- the original audit's aggregate summary and exact run completion timestamp.

Until then, reports should cite the materialized-label counts above and explicitly label the raw
audit statistics as unavailable rather than infer them from the derived data.
