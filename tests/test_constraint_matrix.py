import numpy as np
import pytest

from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix


def test_shape_matches_n_customers() -> None:
    m_true = build_constraint_matrix(routes=[[0, 1], [2, 3]], n_customers=4)
    assert m_true.shape == (4, 4)


def test_zero_diagonal() -> None:
    m_true = build_constraint_matrix(routes=[[0, 1, 2]], n_customers=3)
    assert np.all(np.diag(m_true) == 0)


def test_symmetric() -> None:
    m_true = build_constraint_matrix(routes=[[0, 1], [2, 3, 4]], n_customers=5)
    assert np.array_equal(m_true, m_true.T)


def test_same_route_customers_get_one() -> None:
    m_true = build_constraint_matrix(routes=[[0, 1, 2]], n_customers=3)
    assert m_true[0, 1] == 1
    assert m_true[1, 0] == 1
    assert m_true[0, 2] == 1
    assert m_true[1, 2] == 1


def test_different_route_customers_get_zero() -> None:
    m_true = build_constraint_matrix(routes=[[0, 1], [2, 3]], n_customers=4)
    assert m_true[0, 2] == 0
    assert m_true[1, 3] == 0
    assert m_true[0, 3] == 0


def test_single_customer_route_has_no_memberships() -> None:
    m_true = build_constraint_matrix(routes=[[0], [1, 2]], n_customers=3)
    assert np.all(m_true[0, :] == 0)
    assert np.all(m_true[:, 0] == 0)


def test_customer_missing_from_all_routes_is_unconstrained() -> None:
    m_true = build_constraint_matrix(routes=[[0, 1]], n_customers=3)
    assert np.all(m_true[2, :] == 0)
    assert np.all(m_true[:, 2] == 0)


def test_empty_routes_gives_all_zero_matrix() -> None:
    m_true = build_constraint_matrix(routes=[], n_customers=3)
    assert np.all(m_true == 0)


def test_zero_customers_gives_empty_matrix() -> None:
    m_true = build_constraint_matrix(routes=[], n_customers=0)
    assert m_true.shape == (0, 0)


def test_out_of_range_customer_index_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_constraint_matrix(routes=[[0, 5]], n_customers=3)


def test_negative_customer_index_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_constraint_matrix(routes=[[-1, 0]], n_customers=3)


def test_customer_in_two_routes_raises() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_constraint_matrix(routes=[[0, 1], [1, 2]], n_customers=3)


def test_negative_n_customers_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_constraint_matrix(routes=[], n_customers=-1)


def test_bool_customer_index_raises() -> None:
    with pytest.raises(ValueError, match="integer"):
        build_constraint_matrix(routes=[[True]], n_customers=3)


def test_non_integer_customer_index_raises() -> None:
    with pytest.raises(ValueError, match="integer"):
        build_constraint_matrix(routes=[["1"]], n_customers=3)  # type: ignore[list-item]


def test_non_integer_n_customers_raises() -> None:
    with pytest.raises(ValueError, match="integer"):
        build_constraint_matrix(routes=[], n_customers=3.5)  # type: ignore[arg-type]
