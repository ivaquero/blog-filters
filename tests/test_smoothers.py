"""Regression tests for :mod:`filters.smoothers`.

Guards the P0 crash that was fixed in Phase 1 (missing ``.copy()`` parens on
``Ps``/``Fs`` and the mis-bracketed covariance update in the top-level
``rts_smoother`` function).
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman import KalmanFilter
from filters.smoothers import FixedLagSmoother, rts_smoother


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


def test_top_level_rts_accepts_flat_means():
    """Only ``(n, dim_x, 1)`` used to be accepted; ``(n, dim_x)`` raised."""
    kf = _build_filter()
    means, covs = _collect_runs([1.0, 2.0, 3.0], kf)
    flat = means[:, :, 0]

    x, P, _K, _Pp = rts_smoother(flat, covs, Fs=kf.F, Q=kf.Q)

    assert x.shape == flat.shape
    assert P.shape == covs.shape
    assert np.allclose(x, kf.rts_smoother(flat, covs)[0])


def test_top_level_rts_accepts_per_step_matrices():
    """``Fs`` used to be treated as one matrix even when a stack was given."""
    kf = _build_filter()
    means, covs = _collect_runs([1.0, 2.0, 3.0, 4.0], kf)
    n = means.shape[0]

    x, P, _K, _Pp = rts_smoother(means, covs, Fs=[kf.F] * n, Qs=[kf.Q] * n)

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(P))


def test_top_level_rts_defaults_to_identity_transition():
    kf = _build_filter()
    means, covs = _collect_runs([1.0, 2.0, 3.0], kf)

    x, _P, _K, _Pp = rts_smoother(means, covs)

    assert np.all(np.isfinite(x))
    assert x.shape == means.shape


def _control_smoother():
    """A smoother whose gain is zero, so x after update equals x_pre exactly."""
    fls = FixedLagSmoother(2, 1, N=2, dim_u=1)
    fls.G = np.array([[1.0], [0.0]])
    fls.Q = np.zeros((2, 2))
    fls.R = np.array([[1e6]])
    fls.P = np.zeros((2, 2))
    fls.H = np.array([[1.0, 0.0]])
    return fls


def test_fixed_lag_control_without_dim_u_raises_clear_error():
    """The old ``G = 0.0`` raised a cryptic matmul error; now it explains dim_u."""
    fls = FixedLagSmoother(2, 1, N=2)
    assert fls.G.shape == (2, 0)

    with pytest.raises(ValueError, match="dim_u"):
        fls.smooth(np.array([[1.0]]), u=np.array([1.0]))


def test_fixed_lag_smoother_applies_control_input():
    fls = _control_smoother()
    fls.smooth(np.array([[1.0]]), u=np.array([2.0]))

    assert np.allclose(fls.x, np.array([[2.0], [0.0]]))


def test_fixed_lag_smooth_batch_applies_control_inputs():
    fls = _control_smoother()
    x_smooth, x_hat = fls.smooth_batch(
        [np.array([[1.0]]), np.array([[2.0]])],
        N=2,
        us=[np.array([2.0]), np.array([3.0])],
    )

    # The state propagates: step 2 starts from step 1's state, then adds u=3.
    assert np.allclose(x_hat[0], np.array([[2.0], [0.0]]))
    assert np.allclose(x_hat[1], np.array([[5.0], [0.0]]))
    assert x_smooth.shape == x_hat.shape
    # The batch variant is functional: it must not touch filter state.
    assert np.allclose(fls.x, np.zeros((2, 1)))


def test_fixed_lag_smooth_batch_rejects_control_without_dim_u():
    fls = FixedLagSmoother(2, 1, N=2)
    with pytest.raises(ValueError, match="dim_u"):
        fls.smooth_batch([np.array([[1.0]])], N=2, us=[np.array([1.0])])


def test_fixed_lag_repr_labels_the_dimensions():
    fls = FixedLagSmoother(3, 2, N=2)
    text = repr(fls)

    assert "dim_x = 3" in text
    assert "dim_z = 2" in text
