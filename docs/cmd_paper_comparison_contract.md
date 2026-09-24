# CMD comparison contract

**Frozen:** 2026-09-24  
**Contract version:** 1  
**Paper:** arXiv:2603.07568v1

This is the Phase 0 operating contract. The machine-readable profiles are
`configs/alignment/paper_cmd.yaml` and `configs/alignment/ours_robust.yaml`; the authoritative
evidence and ambiguity register is `configs/alignment/cmd_paper_evidence.yaml`. Any newly noticed
reproduction-relevant paper detail must be added to that register with exactly one classification
before it changes a `paper_cmd` experiment.

## Track boundary

| Concern | `paper_cmd` | `ours_robust` |
|---|---|---|
| Claim | Paper-guided v1 reconstruction | Project extension, never a paper-baseline claim |
| Diffusion labels | One HGS route partition | Audited, excluded, stochastic, or consensus labels allowed |
| Global policy encoder | Exact frozen diffusion GAT | Trainable Transformer |
| Local encoder | Hard-`M` masked GAT | Project prior-masked Transformer |
| Fusion | Sum then MLP | Learned gate |
| Artifact provenance | Versioned `paper_cmd` checkpoint required | Legacy/project artifacts allowed |
| Ambiguous paper choices | Declared reconstruction assumption | May be tuned as a project choice |

The three component templates named in each profile are the only canonical entry points for that
track. Derived run configs must retain the alignment block and contract version.

## Evidence classifications

- `explicitly_stated`: printed in the paper. The implementation follows it without substitution.
- `mathematically_inferred`: required by printed equations or identities but not written as an
  implementation instruction. The inference is recorded and tested.
- `ambiguous_awaiting_author_clarification`: multiple implementations remain consistent with the
  paper. The current choice is an assumption and cannot be described as author-confirmed.

The evidence ledger records the model dimensions, diffusion process, augmentations, corpus sizes,
training and inference algorithms, benchmark scope, inferred skipped posterior, and all currently
known ambiguities.

## Fail-fast gate

A `paper_cmd` run is rejected when:

- a required paper-compatible config value drifts;
- audited/consensus label policy is selected;
- its GAT, diffusion-prior, or restored policy checkpoint lacks versioned provenance;
- a referenced checkpoint is marked `ours_robust`;
- `architecture: paper_cmd` is used without the explicit alignment contract.

Checkpoints embed `alignment.track`, `alignment.contract_version`, the paper identifier, and the
claim. Policy checkpoints remain self-contained, but their embedded provenance is still checked
when loaded.

## Open author questions

The evidence ledger contains the exact question text and status for:

1. exact seeds;
2. HGS label-generation budget and settings;
3. GAT pretraining objective and checkpoint selection;
4. augmentation enumeration and composition;
5. skipped-transition schedule and formula;
6. synthetic test-set generation seeds/files;
7. joint versus per-size model training.

Until answered, experiments must retain the claim “paper-guided reconstruction”; they must not be
described as an exact author-run reproduction.

