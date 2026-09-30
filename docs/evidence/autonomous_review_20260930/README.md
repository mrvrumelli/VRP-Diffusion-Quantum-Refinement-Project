# Archived review evidence

These are the exact small evidence files and diagnostic source snapshots used by the
2026-09-30 review. They predate the production sampler correction. The `.py.txt` extension
preserves historical diagnostic code without treating it as maintained package code.
To reproduce, copy the two source snapshots to `outputs/autonomous_review_20260930/`
with their original `.py` names, then run them from the repository root using the project
virtual environment. The checkpoint and dataset paths in the snapshots must still exist.
The 24-example panel is a diagnostic validation panel, not an untouched test or a safe
panel for the unfiltered checkpoint. New comparisons use the corrective evaluation protocol.
