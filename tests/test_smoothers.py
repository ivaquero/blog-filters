"""Regression tests for :mod:`filters.smoothers`.

Guards the P0 crash that was fixed in Phase 1 (missing ``.copy()`` parens on
``Ps``/``Fs`` and the mis-bracketed covariance update in the top-level
``rts_smoother`` function).
"""

from __future__ import annotations

import numpy as np

from filters.kalman import KalmanFilter
from filters.smoothers import rts_smoother


def _collect_runs(zs, kf):
    """Return (means, covs) shaped for both rts_smoother variants.

    ``KalmanFilter.rts_smoother`` happily consumes ``(n, dim_x)``, but the
    top-level ``filters.smoothers.rts_smoother`` unpacks a third axis, so we
    hand it ``(n, dim_x, 1)`` column vectors.
    """
    means, covs = [], []
    for z in zs:
        kf.predict()
        kf.update([z])
        means.append(kf.x.reshape(-1, 1).copy())
        covs.append(kf.P.copy())
    return np.array(means), np.array(covs)


def _build_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = np.array([[0.05, 0.05], [0.05, 0.05]])
    kf.R = np.array([[0.25]])
    kf.x = np.array([0.0, 1.0])
    kf.P = np.eye(2)
    return kf


def test_top_level_rts_produces_finite_output():
    """The Phase 1 crash used to fire on ``F = Fs.copy`` (missing parens)."""
    rng = np.random.default_rng(7)
    zs = [float(i + rng.standard_normal() * 0.5) for i in range(15)]
    kf = _build_filter()
    means, covs = _collect_runs(zs, kf)

    x, P, _K, _Pp = rts_smoother(means, covs, Fs=kf.F, Q=kf.Q)

    assert np.all(np.isfinite(x))
    assert x.shape == means.shape
    assert P.shape == covs.shape
    # The last smoothed step equals the last filtered step.
    assert np.allclose(x[-1], means[-1], atol=1e-9)


def test_top_level_rts_returns_copies_not_views():
    """`rts_smoother` must not mutate the caller's Xs/Ps arrays."""
    kf = _build_filter()
    means, covs = _collect_runs([1.0, 2.0, 3.0, 4.0, 5.0], kf)
    means_snapshot = means.copy()
    covs_snapshot = covs.copy()

    rts_smoother(means, covs, Fs=kf.F, Q=kf.Q)

    assert np.array_equal(means, means_snapshot)
    assert np.array_equal(covs, covs_snapshot)


def test_top_level_rts_matches_method_on_constant_fs():
    """When F and Q do not change per step, the standalone function should
    agree with the bound method up to float tolerance."""
    rng = np.random.default_rng(11)
    zs = [float(i + rng.standard_normal() * 0.5) for i in range(12)]

    kf_a = _build_filter()
    means_a, covs_a = _collect_runs(zs, kf_a)
    smoothed_a, _, _, _ = kf_a.rts_smoother(means_a, covs_a)

    kf_b = _build_filter()
    means_b, covs_b = _collect_runs(zs, kf_b)
    smoothed_b, _, _, _ = rts_smoother(means_b, covs_b, Fs=kf_b.F, Q=kf_b.Q)

    assert np.allclose(smoothed_a, smoothed_b, atol=1e-6)
