import json
from pathlib import Path

import pytest

from test_training_labels import _candidate, _source_example
from vrp_diffusion_quantum.data.dataset import load_example, save_example
from vrp_diffusion_quantum.data.single_reference import (
    materialize_single_hgs_reference,
    validate_single_reference_dataset,
)


def test_single_seed_materialization_and_manifest_enforcement(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    save_example(_source_example(), source)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_candidate([[0], [1, 2]], 11, "selected", 12)))
    output = tmp_path / "labels"
    entry = materialize_single_hgs_reference(
        source, candidate, output / "example.json", base_seed=12
    )
    assert load_example(output / "example.json").solution.cost == 11
    with pytest.raises(ValueError, match="requires a verified"):
        validate_single_reference_dataset(output)
    manifest = {
        "label_policy": "single_hgs_route_partition",
        "stability_filter": False,
        "base_seed": 12,
        "examples": [entry],
    }
    (output / "training_label_manifest.json").write_text(json.dumps(manifest))
    validate_single_reference_dataset(output)
    with pytest.raises(ValueError, match="seed or source"):
        materialize_single_hgs_reference(source, candidate, output / "wrong.json", base_seed=11)
    (output / "example.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_single_reference_dataset(output)
