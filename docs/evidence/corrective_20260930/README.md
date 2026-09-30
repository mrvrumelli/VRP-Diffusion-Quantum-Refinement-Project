# Corrective execution evidence

`summary.json` contains readable aggregate tables. `graph_results_and_manifests.json.gz` maps original paths to per-graph counts, costs, timings, source/panel manifests, diagnostic configurations and summaries. The bundle supports recalculating paired graph-bootstrap results without the models.

`artifact_manifest.json` hashes these artifacts and the final code/configuration snapshot. It explicitly records a dirty working-tree snapshot, not an asserted clean start commit. Large probability arrays and checkpoints remain in the corresponding local outputs. The bundle does not include predictions for the reserved untouched test.

`candidate_baseline.json` records the freeze decision and retained candidates.
