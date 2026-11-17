"""Regression tests for :mod:`filters.kalman_ckf` (cubature Kalman filter).

The module was unreferenced by the notebooks and tests, which is where its
shape bug survived: ``ckf_transform`` returned a column mean while the state
was 1-D, so ``update`` raised a broadcast error unless ``predict`` had just
turned the state into a column.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman_ckf import CubatureKalmanFilter, ckf_transform


def _identity_model() -> CubatureKalmanFilter:
    """Two-state, one-measurement filter with an identity process model."""
    return CubatureKalmanFilter(
        dim_x=2,
        dim_z=1,
        dt=1.0,
        hx=lambda x, *_args: np.array([x[0]]),
        fx=lambda x, _dt, *_args: np.asarray(x),
    )


def test_state_stays_one_dimensional_after_predict():
    ckf = _identity_model()
    ckf.predict()
    assert ckf.x.shape == (2,)


def test_update_before_predict_does_not_raise():
    """A (dim_x,) state against a (dim_z, 1) residual used to broadcast."""
    ckf = _identity_model()
    ckf.update(np.array([1.0]))

    assert ckf.x.shape == (2,)
    assert ckf.y.shape == (1,)


def test_column_and_flat_measurements_agree():
    flat = _identity_model()
    flat.Q = np.eye(2) * 0.1
    flat.R = np.array([[0.5]])
    flat.predict()
    flat.update(np.array([1.0]))

    column = _identity_model()
    column.Q = np.eye(2) * 0.1
    column.R = np.array([[0.5]])
    column.predict()
    column.update(np.array([[1.0]]))

    assert np.allclose(flat.x, column.x)
    assert np.allclose(flat.P, column.P)


def test_covariance_stays_symmetric_and_positive():
    ckf = _identity_model()
    ckf.Q = np.eye(2) * 0.1
    ckf.R = np.array([[0.5]])

    for i in range(25):
        ckf.predict()
        ckf.update(np.array([float(i)]))

    assert np.allclose(ckf.P, ckf.P.T)
    assert np.all(np.linalg.eigvalsh(ckf.P) > -1e-12)


def test_callback_returning_a_column_is_flattened():
    ckf = CubatureKalmanFilter(
        dim_x=2,
        dim_z=1,
        dt=1.0,
        hx=lambda x, *_args: np.array([[x[0]]]),
        fx=lambda x, _dt, *_args: np.asarray(x).reshape(-1, 1),
    )

    ckf.predict()
    ckf.update(np.array([2.0]))

    assert ckf.x.shape == (2,)
    assert ckf.sigmas_h.shape == (4, 1)


def test_callback_returning_wrong_size_fails_loudly():
    ckf = CubatureKalmanFilter(
        dim_x=2,
        dim_z=1,
        dt=1.0,
        hx=lambda x, *_args: np.array([x[0], x[1]]),
        fx=lambda x, _dt, *_args: np.asarray(x),
    )
    ckf.predict()

    with pytest.raises(ValueError, match="hx"):
        ckf.update(np.array([1.0]))


def test_ckf_transform_recovers_mean_and_covariance():
    Xs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x, P = ckf_transform(Xs, np.zeros((2, 2)))

    assert x.shape == (2,)
    assert np.allclose(x, Xs.mean(axis=0))
    assert np.allclose(P, np.cov(Xs.T, bias=True))


def test_gain_has_matrix_shape_before_first_update():
    ckf = _identity_model()
    assert ckf.K.shape == (2, 1)
