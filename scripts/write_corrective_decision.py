"""Write the evidence-based freeze decision without promoting a development winner."""

from __future__ import annotations

import json
from pathlib import Path

from run_corrective_evaluation import sha256


def main() -> None:
    root = Path("docs/evidence/corrective_20260930")
    results = json.loads((root / "summary.json").read_text())
    panel = json.loads(Path("outputs/corrective_confirmation_20260930/panel.json").read_text())
    full = results["expanded_development"]["full_posterior_mixture_v2_steps50_seed0"]
    manifest = {
        "schema_version": 1,
        "date": "2026-09-30",
        "decision": "defer classical and paper freeze; promote no checkpoint",
        "reason": (
            "The repeated source-count curve failed its declared improvement rule. "
            "Diagnostic changes did not establish dependable route improvement. "
            "Development results do not establish independent test improvement."
        ),
        "selected_checkpoint": None,
        "historical_comparators": panel["checkpoints"],
        "data": {
            "development_manifest": "outputs/corrective_confirmation_20260930/panel.json",
            "reserved_test_manifest": "outputs/corrective_20260930/reserved_test.json",
            "reserved_test_scored": False,
            "reference_limitation": "strengthen original reserved labels before final route claims",
            "single_reference_split_counts": {"train": 50000, "validation": 289, "test": 291},
            "single_reference_seed": 50937,
            "stability_filter": False,
            "manifest_sha256": {
                path.as_posix(): sha256(path)
                for path in (
                    Path("outputs/corrective_confirmation_20260930/panel.json"),
                    Path("outputs/corrective_20260930/reserved_test.json"),
                    *Path("data/processed/corrective_20260930/single_hgs").glob(
                        "*/training_label_manifest.json"
                    ),
                )
            },
        },
        "evaluation": {
            "sampler": "posterior_mixture_v2",
            "inference_steps": 50,
            "threshold": 0.5,
            "batch_size": 1,
            "precision": "float32",
            "checkpoint_comparison_seed": 0,
            "primary_route_metric": "mean_instance_gap_percent",
            "input_frame_existing_checkpoints": "absolute",
            "diagnostic_input_frame": "depot_relative; explicit opt-in, not promoted",
            "policy_starts": [1, 8, 16],
            "policy_prior": "champion corrected 50-step, seed 0",
        },
        "paper_target": {
            "source": "https://arxiv.org/pdf/2603.07568v1#page=13",
            "reported_50_step_f1": 0.823,
            "full_checkpoint_development_f1": full["f1"],
            "arithmetic_shortfall": 0.823 - full["f1"]["estimate"],
            "shortfall_is_like_for_like": False,
            "unresolved": [
                "Figure 8 size mixture, F1 averaging and exact scored sampling protocol",
                "author model inputs, especially depot representation",
                "exact training objective and probability parameterization",
                "label solver stopping/seed policy and unfiltered source distribution",
                "normalization/training schedule and checkpoint-selection details",
            ],
        },
        "classification": {
            "reproduced": [
                "legacy soft-posterior mismatch; corrected equation matches enumeration",
                "source-count and label-provenance discrepancies",
                "gap aggregation mismatch and contaminated historical comparison panel",
                "depot information loss in absolute customer-only inputs",
                "CVRPLIB unrestricted-fleet mismatch",
                "tiny-source fit and oracle endpoint sanity",
            ],
            "supported": [
                "F1 gains need not improve decoded routes",
                "reduced mixed-size BatchNorm variants exhibit substantial seed/mode sensitivity",
                "merged OOD checkpoints have worse route point estimates in all nine cells",
            ],
            "rejected_for_tested_protocol": [
                "sampler correction alone recovers paper F1",
                "more inference steps alone recover paper F1",
                "fitted size-specific logit intercept improves full-checkpoint generation",
                "five-epoch 500-to-1000-source N100 expansion passes the route-gain rule",
            ],
            "inconclusive": [
                "depot conditioning improvement at paper scale",
                "which remaining factor explains most of the F1 deficit",
                "independent superiority of Task 5 / Task 6 validation-selected candidates",
                "effects of substantially longer training or a verified variational objective",
            ],
        },
        "curve_decision": results["curve_decision"],
        "task5_development_evidence": results["expanded_paired"],
        "independently_confirmed_new_winners": [],
        "downstream_gates": {
            "paper_tasks_11_to_14": "not launched; prerequisites remain unsatisfied",
            "quantum_tasks_7_to_8": "not launched; classical freeze and D.1-D.3 still required",
            "authorization_scope": "R1-R12 corrective execution; no automatic downstream promotion",
        },
        "bounded_next_hypothesis": (
            "Before more training, resolve paper F1 aggregation and objective/conditioning "
            "ambiguities using author artifacts if available; then preregister one matched "
            "two-seed objective/conditioning comparison on the repaired single-HGS splits. "
            "Do not automatically expand labels or repeat the failed curve."
        ),
    }
    (root / "candidate_baseline.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
