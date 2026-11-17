"""Tests for :mod:`filters.helpers` — the shared Mahalanobis distance.

The four covariance-based filters each carried their own copy of the distance
formula. Three of them wrote ``float(self.y.T @ SI @ self.y)``, which raises
``TypeError: only 0-dimensional arrays can be converted to Python scalars``
because a ``(1, 1)`` array is not 0-d — so ``repr(kf)`` blew up before the
first ``update``. ``helpers.mahalanobis`` is now the single implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.helpers import mahalanobis


def test_distance_matches_the_quadratic_form():
    y = np.array([1.0, -2.0])
    S = np.array([[2.0, 0.0], [0.0, 0.5]])

    expected = np.sqrt(y @ np.linalg.inv(S) @ y)
    assert mahalanobis(y, S) == pytest.approx(expected)


def test_accepts_a_column_vector():
    """``self.y`` is ``(dim_z, 1)`` in some filters and ``(dim_z,)`` in others."""
    assert mahalanobis(np.array([[3.0]]), np.array([[4.0]])) == pytest.approx(1.5)


def test_zero_covariance_returns_zero():
    """Before the first update ``S`` is all zeros — the distance is undefined."""
    assert mahalanobis(np.zeros(2), np.zeros((2, 2))) == 0.0


def test_singular_covariance_falls_back_to_the_pseudoinverse():
    y = np.array([1.0, 1.0])
    S = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1

    assert mahalanobis(y, S) == pytest.approx(np.sqrt(y @ np.linalg.pinv(S) @ y))
    assert np.isfinite(mahalanobis(y, S))


def test_negative_quadratic_form_is_clamped_to_zero():
    """Defensive: a sign-flipped covariance must not reach ``sqrt``."""
    assert mahalanobis(np.array([1.0]), np.array([[-1.0]])) == 0.0


def test_repr_is_safe_before_the_first_update():
    """Regression: ``repr`` of every filter used to crash on ``mahalanobis``."""
    from filters.kalman import KalmanFilter
    from filters.kalman_ckf import CubatureKalmanFilter
    from filters.kalman_ekf import ExtendedKalmanFilter
    from filters.kalman_ukf import UnscentedKalmanFilter
    from filters.sigma_points import MerweScaledSigmas

    fx = np.eye(2)
    hx = np.array([[1.0, 0.0]])

    fresh = [
        KalmanFilter(2, 1),
        ExtendedKalmanFilter(2, 1),
        UnscentedKalmanFilter(2, 1, 1.0, hx, fx, MerweScaledSigmas(2, 0.1, 2, 0)),
        CubatureKalmanFilter(2, 1, 1.0, hx, fx),
    ]

    for kf in fresh:
        assert "mahalanobis" in repr(kf)
        assert kf.mahalanobis == 0.0


def test_kfsaver_flatten_drops_trailing_singleton_axes():
    """``shape[2]`` used to IndexError on 1-D/2-D arrays and get swallowed."""
    from filters.helpers import KFSaver

    class Dummy:
        col: np.ndarray
        cube: np.ndarray
        flat: np.ndarray

    kf = Dummy()
    kf.col = np.array([[1.0], [2.0], [3.0]])
    kf.cube = np.zeros((3, 2, 1))
    kf.flat = np.arange(3.0)

    saver = KFSaver(kf)
    saver.save()
    saver.save()
    saver.to_array()
    saver.flatten()

    assert saver.col.shape == (2, 3)
    assert saver.cube.shape == (2, 3, 2)
    assert saver.flat.shape == (2, 3)


def test_kfsaver_flatten_keeps_non_singleton_last_axis():
    from filters.helpers import KFSaver

    class Dummy:
        mat: np.ndarray

    kf = Dummy()
    kf.mat = np.arange(6.0).reshape(3, 2)

    saver = KFSaver(kf)
    saver.save()
    saver.to_array()
    saver.flatten()

    assert saver.mat.shape == (1, 3, 2)


def test_likelihood_of_prefers_the_native_property():
    from filters.helpers import likelihood_of

    class Native:
        likelihood = 0.25

    assert likelihood_of(Native()) == 0.25


def test_likelihood_of_reconstructs_a_gaussian_likelihood():
    """H-infinity has no ``likelihood``; y + V are enough for mode weights."""
    from filters.helpers import likelihood_of

    class NoProperty:
        def __init__(self):
            self.y = np.array([[1.0]])
            self.S = np.zeros((1, 1))
            self.V = np.array([[4.0]])

    from scipy import stats

    expected = stats.multivariate_normal.logpdf(np.array([1.0]), cov=np.array([[4.0]]))
    assert likelihood_of(NoProperty()) == pytest.approx(np.exp(expected))


def test_likelihood_of_floors_when_covariance_is_unavailable():
    import sys

    from filters.helpers import likelihood_of

    class Bare:
        y = np.array([1.0])

    assert likelihood_of(Bare()) == sys.float_info.min


def test_as_covariance_expands_scalars_and_preserves_matrices():
    from filters.helpers import as_covariance

    assert np.array_equal(as_covariance(2.0, 2), np.array([[2.0, 0.0], [0.0, 2.0]]))
    matrix = np.array([[1, 2], [3, 4]])
    result = as_covariance(matrix, 2)
    assert np.array_equal(result, matrix)
    assert result.dtype == np.float64
