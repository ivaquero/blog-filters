"""Regression tests for :mod:`filters.kalman_hinf` (H-infinity filter).

``update`` computed the H-infinity gain into a local variable and never wrote
it back, so ``self.K`` stayed at its initial ``0`` (an int, not even a matrix),
and ``__repr__`` printed ``dim_u`` under the ``gamma`` label.
"""

from __future__ import annotations

import numpy as np

from filters.kalman_hinf import HInfinityFilter


def _filter(gamma: float = 0.01) -> HInfinityFilter:
    f = HInfinityFilter(dim_x=2, dim_z=1, dim_u=0, gamma=gamma)
    f.H = np.array([[1.0, 0.0]])
    f.P = np.eye(2)
    f.Q = np.eye(2) * 0.01
    f.W = np.eye(2) * 0.1
    f.V = 1.0
    return f


def test_gain_is_recorded_after_update():
    f = _filter()
    f.update(np.array([[1.0]]))

    assert f.K.shape == (2, 1)
    assert np.any(f.K != 0.0)


def test_gain_is_a_matrix_before_the_first_update():
    f = _filter()

    assert f.K.shape == (2, 1)
    assert np.all(f.K == 0.0)


def test_repr_reports_gamma_not_dim_u():
    f = HInfinityFilter(dim_x=2, dim_z=1, dim_u=3, gamma=0.5)

    assert "gamma = 0.5" in repr(f)


def test_update_with_none_leaves_the_state_untouched():
    f = _filter()
    x_before, P_before = f.x.copy(), f.P.copy()

    f.update(None)

    assert np.array_equal(f.x, x_before)
    assert np.array_equal(f.P, P_before)


def test_update_keeps_the_covariance_symmetric():
    f = _filter()
    f.update(np.array([[1.0]]))

    assert np.allclose(f.P, f.P.T)


def test_measurement_pulls_the_estimate_towards_it():
    f = _filter()
    f.update(np.array([[4.0]]))

    assert f.x[0, 0] > 0.0
    assert np.isclose(f.y[0, 0], 4.0)


def test_predict_applies_the_control_input():
    f = _filter()
    f.G = np.array([[0.0], [1.0]])

    f.predict(u=np.array([[2.0]]))

    assert np.isclose(f.x[1, 0], 2.0)


def test_batch_filter_returns_means_and_covariances():
    f = _filter()
    means, covs = f.batch_filter([np.array([[1.0]]), np.array([[2.0]])])

    assert means.shape == (2, 2, 1)
    assert covs.shape == (2, 2, 2)
    assert np.all(np.isfinite(means))


def test_scalar_measurement_noise_expands_to_dim_z():
    f = HInfinityFilter(dim_x=2, dim_z=2, dim_u=0, gamma=0.01)

    f.V = 4.0

    assert f.V.shape == (2, 2)
    assert np.allclose(f.V, np.eye(2) * 4.0)
