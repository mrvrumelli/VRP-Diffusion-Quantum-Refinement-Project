import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.matrix_predictor import (
    MatrixPredictor,
    build_pair_features,
    matrix_bce_loss,
    train_matrix_predictor,
)


def _tiny_instance(instance_id: str = "toy_0") -> CVRPInstance:
    return CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        demands=np.array([0.0, 3.0, 4.0, 2.0]),
        capacity=10.0,
        depot_index=0,
        instance_id=instance_id,
        n_customers=3,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )


def _tiny_solution() -> LabeledSolution:
    return LabeledSolution(
        routes=[[0, 1], [2]],
        cost=12.5,
        num_vehicles=2,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )


def _tiny_example(instance_id: str = "toy_0") -> CVRPExample:
    return make_example(_tiny_instance(instance_id), _tiny_solution())


def test_build_pair_features_shape_and_symmetry() -> None:
    customer_coords = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    customer_demands = torch.tensor([1.0, 2.0, 3.0])

    features = build_pair_features(customer_coords, customer_demands, capacity=10.0)

    assert features.shape == (3, 3, 8)
    # distance feature (index 4) must be symmetric and zero on the diagonal
    distance = features[:, :, 4]
    assert torch.allclose(distance, distance.T)
    assert torch.allclose(torch.diag(distance), torch.zeros(3))
    assert torch.isclose(distance[0, 1], torch.tensor(3.0))
    assert torch.isclose(distance[0, 2], torch.tensor(4.0))


def test_matrix_predictor_output_shape_range_symmetry_and_diagonal() -> None:
    torch.manual_seed(0)
    model = MatrixPredictor(hidden_dim=8)
    customer_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    customer_demands = torch.tensor([1.0, 2.0, 3.0])

    m_prob = model(customer_coords, customer_demands, capacity=10.0)

    assert m_prob.shape == (3, 3)
    assert torch.all(m_prob >= 0.0)
    assert torch.all(m_prob <= 1.0)
    assert torch.allclose(m_prob, m_prob.T)
    assert torch.allclose(torch.diag(m_prob), torch.zeros(3))


def test_matrix_bce_loss_lower_when_predictions_match_targets() -> None:
    m_true = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    matching = torch.tensor([[0.0, 0.99, 0.01], [0.99, 0.0, 0.01], [0.01, 0.01, 0.0]])
    mismatched = torch.tensor([[0.0, 0.01, 0.99], [0.01, 0.0, 0.99], [0.99, 0.99, 0.0]])

    matching_loss = matrix_bce_loss(matching, m_true)
    mismatched_loss = matrix_bce_loss(mismatched, m_true)

    assert matching_loss.item() < mismatched_loss.item()


def test_train_matrix_predictor_rejects_empty_examples() -> None:
    torch.manual_seed(0)
    model = MatrixPredictor(hidden_dim=8)
    with pytest.raises(ValueError, match="empty"):
        train_matrix_predictor(model, [], num_epochs=1, learning_rate=0.01)


def test_train_matrix_predictor_overfits_tiny_dataset() -> None:
    torch.manual_seed(0)
    model = MatrixPredictor(hidden_dim=16)
    examples = [_tiny_example("toy_0"), _tiny_example("toy_1")]

    epoch_losses = train_matrix_predictor(model, examples, num_epochs=300, learning_rate=0.05)

    assert len(epoch_losses) == 300
    assert epoch_losses[-1] < epoch_losses[0]
    assert epoch_losses[-1] < 0.1
