"""Regression tests for :mod:`filters.sigma_points`.

Guards the specific crashes that were fixed in Phase 1:

- ``SphericalRadialSigmas.sigma_points`` used to fall off the end and return
  ``None``.
- ``SimplexSigmas`` used to skip the ``hstack`` when ``n > 1`` and produce a
  mis-shaped point set.
- ``num_sigmas`` used to take an argument even though it's a no-op method.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.sigma_points import (
    JulierSigmas,
    MerweScaledSigmas,
    SimplexSigmas,
    SphericalRadialSigmas,
)


def test_julier_sigma_count_and_weights():
    sp = JulierSigmas(n=3)
    assert sp.num_sigmas() == 2 * 3 + 1
    assert np.isclose(sp.Wm.sum(), 1.0)
    assert np.isclose(sp.Wc.sum(), 1.0)


def test_merwe_sigma_points_roundtrip_mean_covariance():
    n = 3
    sp = MerweScaledSigmas(n=n, kappa=0.0, alpha=1.0, beta=0.0)
    x = np.array([1.0, -2.0, 0.5])
    P = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 0.8]])
    sigmas = sp.sigma_points(x, P)
    assert sigmas.shape == (2 * n + 1, n)
    assert np.allclose(sp.Wm @ sigmas, x, atol=1e-9)
    dx = sigmas - x
    cov = dx.T @ np.diag(sp.Wc) @ dx
    assert np.allclose(cov, P, atol=1e-9)


def test_simplex_sigma_points_handles_multidim():
    """Phase 1 fix: SimplexSigmas used to skip the hstack when n > 1.

    The generator yields n+1 points and reproduces the input mean/covariance
    under the sigma-weighted transform.
    """
    n = 3
    sp = SimplexSigmas(n=n)
    x = np.array([0.0, 1.0, -2.0])
    P = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 0.8]])
    sigmas = sp.sigma_points(x, P)
    assert sigmas.shape == (n + 1, n)
    # Weighted mean reproduces x.
    assert np.allclose(sp.Wm @ sigmas, x, atol=1e-9)
    # Weighted covariance reproduces P.
    dx = sigmas - x
    cov = dx.T @ np.diag(sp.Wc) @ dx
    assert np.allclose(cov, P, atol=1e-6)


def test_spherical_radial_returns_points_not_none():
    """Phase 1 fix: `sigma_points` used to have no return statement."""
    n = 3
    P = np.eye(n)
    sp = SphericalRadialSigmas(n=n, P=P)
    x = np.zeros(n)
    sigmas = sp.sigma_points(x, P)
    assert sigmas is not None, "SphericalRadialSigmas must return an array"
    assert sigmas.ndim == 2
    assert sigmas.shape[1] == n


def test_num_sigmas_takes_no_argument():
    sp = MerweScaledSigmas(n=2, kappa=0)
    # Phase 1 fix: `num_sigmas(x)` used to raise TypeError.
    assert sp.num_sigmas() == 5
    with pytest.raises(TypeError):
        sp.num_sigmas(np.zeros(2))  # ty: ignore[too-many-positional-arguments]
