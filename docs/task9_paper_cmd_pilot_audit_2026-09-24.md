# paper_cmd pilot label audit — 2026-09-24

Track B Task 9 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
label a pilot slice of the paper_cmd diffusion-training pool before committing to the full
50,000-instance campaign the plan calls for, staged as its own gate rather than run blind.

## Frozen inputs

- Source pool: 4,949 instances (1,644/1,653/1,652 for N20/N50/N100), selected deterministically
  from `cvrp_s7799_n20-50-100_x66667/splits/train` (`--per-size 1667 --seed 50937`, 5,001 drawn,
  52 pruned for overlapping the already-audited `s7799_strong_reference` pool — verified zero
  overlap after pruning).
- Config: `configs/data/label_audit_paper_cmd_pilot.yaml` — 2 PyVRP seeds (`[50937, 50938]`,
  10/20/40s budgets by size), OR-Tools challenger, 11 workers.
- Tooling note: a real gap was found and fixed to run this at all — `scripts/run_strong_label_audit.py`
  hardcoded "exactly four seeds"; relaxed to `len(base_seeds) >= minimum_near_best_seeds`
  (backward-compatible, verified `pytest -q` still 507/507).
- **Correction on OR-Tools scope**: `stable_sample_per_size: 0` was intended to skip the OR-Tools
  challenger entirely, matching the paper's cheaper single/few-seed protocol. It does not — that
  setting only removes the *additional deterministic stable-control sample*; the challenger still
  runs on every cost-unstable-or-matrix-ambiguous case regardless. It ran on 1,343 instances here
  (all of `needs_review`), not zero. Recorded as a documentation error to fix, not a run defect —
  the audit itself completed correctly under the policy as actually written.

## Results

9,898/9,898 PyVRP runs (231,648 CPU-seconds), 1,343/1,343 OR-Tools challenges (47,162 CPU-seconds),
**0 solver errors**. Wall time: ~5h51m at 11 workers (faster than the ~9-10h early-progress
extrapolation — throughput picked up once past the slower early mix).

| Size | Instances | Cost-stable (`reference_accepted`) | Matrix-stable (`matrix_target_accepted`) | Needs review |
|---|---:|---:|---:|---:|
| N20 | 1,644 | 1,644 (100%) | **1,637 (99.6%)** | 7 (0.4%) |
| N50 | 1,653 | 1,393 (84.3%) | **1,332 (80.6%)** | 321 (19.4%) |
| N100 | 1,652 | 817 (49.5%) | **637 (38.6%)** | 1,015 (61.4%) |
| **Total** | **4,949** | **3,854** | **3,606** | **1,343** |

Artifacts: `outputs/label_audit/paper_cmd_pilot_5k/{metrics.json,summary.csv,accepted_matrix_examples/,reference_examples/,candidates/}`.

## Interpretation

N20 is essentially fully stable at 2 seeds — consistent with the original 4-seed audit's finding
that CVRP20 has zero real candidate ambiguity. N50 and especially N100's lower acceptance rates
(80.6%, 38.6%) versus the original 4-seed protocol's N50/N100 rates (94.4%, 47.8%, from
`docs/archive/evidence/label_audit_s7799_decision.md`) are expected, not a red flag: 2-seed
agreement is a strictly weaker signal than 4-seed agreement (`minimum_near_best_seeds: 2` out of
2 available seeds is a much easier bar to *fail* than 2-out-of-4), so more cases land in
`needs_review` at this cheaper seed count. This is the real cost of approximating the paper's
1-seed protocol at 2 seeds — a smaller, noisier accepted pool per instance audited, documented
rather than hidden.

**Usable pool for Task 10**: 3,606 matrix-stable examples (1,637/1,332/637 by size) — about 7% of
the paper's 50,000-instance target, but real, hashed, and immediately usable. `accepted_matrix_examples/`
is the right folder to point `diffusion_denoiser_paper_cmd.yaml`'s `dataset.path` at, not
`reference_examples/` (cost-stable but not necessarily matrix-stable — the wrong pool for a
binary route-membership training target).

## Next

Launch Task 10 (train the paper-config diffusion model) on this pool and check the F1≈0.823-at-50-steps
gate. If it clears the gate even on this reduced 3,606-example pool, that's real evidence the full
50,000-instance campaign is not necessary on this hardware; if not, the honest read is that this
pilot's yield (esp. N100's 637 accepted examples) may simply be too small, and the decision becomes
whether to run the full-scale audit or accept a lower F1 as the paper_cmd baseline for now.
