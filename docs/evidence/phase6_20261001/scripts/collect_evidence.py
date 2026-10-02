"""Copy the small Phase 6 result files into docs/evidence/phase6_20261001 with a hash manifest."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/phase6_20261001"
DEST = ROOT / "docs/evidence/phase6_20261001"

PATTERNS = [
    "experiment_*/summary.json",
    "experiment_*/analysis.json",
    "loop_*/summary.json",
    "extension/*/summary.json",
    "extension/*.json",
    "extension/tables_*.md",
    "selection/*/summary.json",
    "selection/analysis_*.json",
    "bias_sweep/*/summary.json",
    "calibration/*.json",
    "qaoa_report_*.json",
    "qaoa_vs_classical_*.json",
    "qaoa_random_control_*.json",
    "qaoa_warm_vs_cold_*.json",
    "exchange_capacity_violations_*.json",
    "final/*/summary.json",
    "final/analysis_*.json",
    "timing/*/summary.json",
    "timing/analysis.json",
]
SETS = ["neighborhoods_development.json", "neighborhoods_validation.json"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "outputs/phase6_20261001", "files": {}, "neighborhood_sets": {}}
    for pattern in PATTERNS:
        for path in sorted(P6.glob(pattern)):
            rel = path.relative_to(P6)
            target = DEST / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest["files"][rel.as_posix()] = sha(path)
    for name in SETS:
        manifest["neighborhood_sets"][name] = sha(P6 / name)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"copied {len(manifest['files'])} files to {DEST}")


if __name__ == "__main__":
    main()
