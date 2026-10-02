"""Build a solutions directory whose routes are another refinement run's final routes (Q5).

Usage: build_polish_start.py <loop rows.jsonl> <solver> <baseline solutions dir> <output dir>

Each output record copies the baseline record (file, reference cost, prior) but replaces the
routes and cost with the given solver's final routes from the loop run, so a following loop run
measures what classical polishing adds on top of that solver.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows_path, solver, baseline_dir, output_dir = sys.argv[1:5]
output = ROOT / output_dir
output.mkdir(parents=True, exist_ok=True)
count = 0
for line in (ROOT / rows_path).read_text().splitlines():
    row = json.loads(line)
    if row["solver"] != solver:
        continue
    source = ROOT / baseline_dir / f"{row['instance_id']}.json"
    record = json.loads(source.read_text())
    record["routes"] = row["routes"]
    record["cost"] = row["final_cost"]
    record["refined_by"] = {"rows": rows_path, "solver": solver}
    (output / source.name).write_text(json.dumps(record) + "\n")
    shutil.copyfile(source.with_name(f"{row['instance_id']}.m_prob.npz"), output / f"{row['instance_id']}.m_prob.npz")
    count += 1
print(f"wrote {count} start records to {output}")
