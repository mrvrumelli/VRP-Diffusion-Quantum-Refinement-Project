from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from test_predict_matrix import _example
from vrp_diffusion_quantum.data.augment import augment_example_d4
from vrp_diffusion_quantum.data.dataset import collate_batch
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    load_denoiser_checkpoint,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.train.train_diffusion import (
    customer_tensors_from_batch,
    save_denoiser_checkpoint,
)


def test_depot_relative_inputs_match_training_and_respond_to_depot() -> None:
    example = _example(5, seed=42)
    moved_coords = example.instance.coords.copy()
    moved_coords[example.instance.depot_index] += [0.1, -0.2]
    moved = replace(example, instance=replace(example.instance, coords=moved_coords))
    assert torch.equal(example_to_model_inputs(example)[0], example_to_model_inputs(moved)[0])
    original = example_to_model_inputs(example, coordinate_frame="depot_relative")
    shifted = example_to_model_inputs(moved, coordinate_frame="depot_relative")
    assert not torch.equal(original[0], shifted[0])
    assert torch.allclose(original[0] - shifted[0], torch.tensor([0.1, -0.2]), atol=1e-7)
    training = customer_tensors_from_batch(
        collate_batch([moved]), coordinate_frame="depot_relative"
    )
    for actual, expected in zip(training, shifted[:3], strict=True):
        assert torch.allclose(actual, expected)


def test_six_customer_depot_counterexample_and_geometric_consistency() -> None:
    example = _example(6, seed=2)
    coords = np.vstack([[0.05, 0.05], np.random.default_rng(2).random((6, 2))])
    example = replace(example, instance=replace(example.instance, coords=coords, capacity=3))
    relative = example_to_model_inputs(example, coordinate_frame="depot_relative")[0]
    moved = coords.copy()
    moved[0] = [0.95, 0.95]
    alternative = replace(example, instance=replace(example.instance, coords=moved))
    assert not torch.equal(
        relative, example_to_model_inputs(alternative, coordinate_frame="depot_relative")[0]
    )
    for k in range(8):
        transformed = augment_example_d4(example, k)
        view = example_to_model_inputs(transformed, coordinate_frame="depot_relative")[0]
        assert torch.allclose(
            torch.linalg.vector_norm(relative, dim=-1),
            torch.linalg.vector_norm(view, dim=-1),
            atol=1e-6,
        )
        assert torch.allclose(torch.cdist(relative, relative), torch.cdist(view, view), atol=1e-6)


def test_coordinate_frame_survives_checkpoint_round_trip(tmp_path: Path) -> None:
    config = {
        "hidden_dim": 16,
        "num_layers": 1,
        "time_embed_dim": 16,
        "coordinate_frame": "depot_relative",
    }
    model = ConstraintDenoiser(**config)
    path = tmp_path / "depot.pt"
    save_denoiser_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        epoch=1,
        row={},
        best_metric_name="loss",
        best_metric_value=0,
        extra={"model": config},
    )
    restored, _ = load_denoiser_checkpoint(path)
    assert restored.coordinate_frame == "depot_relative"
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )
