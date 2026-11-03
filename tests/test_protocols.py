"""Guardrails for the Protocol-first typing layer.

These tests lock in the Python 3.14 modernisation contract:

* ``KalmanFilter`` and ``ExtendedKalmanFilter`` satisfy the runtime-checkable
  ``Filter`` protocol, so IMM/MMAE helpers can accept duck-typed doubles.
* Batch and smoother APIs return ``NamedTuple`` instances whose fields are
  addressable by name AND still unpack as positional tuples — the exact
  behaviour our notebooks rely on.
* ``resolve_rng`` honours a caller-supplied ``numpy.random.Generator`` and
  falls back to a fresh one when ``None`` is passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters import resamplers
from filters._typing import (
    Batch2Output,
    BatchOutput,
    Filter,
    IMMOutput,
    RtsOutput,
    SmootherOutput,
    UkfRtsOutput,
    resolve_rng,
)
from filters.kalman import KalmanFilter
from filters.kalman_ekf import ExtendedKalmanFilter
from filters.kalman_hinf import HInfinityFilter
from filters.kalman_ukf import UnscentedKalmanFilter
from filters.sigma_points import JulierSigmas


def _synthetic_zs(n: int = 10, rng: np.random.Generator | None = None) -> np.ndarray:
    generator = resolve_rng(rng)
    return np.array([i + generator.standard_normal() * 0.1 for i in range(n)])


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_kalman_filter_satisfies_protocol():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    assert isinstance(kf, Filter)


def test_extended_kalman_filter_satisfies_protocol():
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    assert isinstance(ekf, Filter)


def test_hinfinity_filter_is_deliberately_off_protocol():
    """H∞ rejects the ``Filter`` protocol on purpose.

    The H-infinity formulation is a worst-case energy bound rather than a
    Bayesian update, so it never carries a measurement covariance ``R``. That
    makes ``isinstance(hf, Filter)`` false and warns downstream code not to
    treat H∞ as a drop-in Kalman substitute inside IMM / MMAE banks.
    """
    hf = HInfinityFilter(dim_x=2, dim_z=1, dim_u=0, gamma=1.0)
    assert not hasattr(hf, "R")
    assert not isinstance(hf, Filter)


def test_ukf_does_not_claim_the_linear_filter_protocol():
    """The UKF is state-driven by ``fx``/``hx`` callables — it intentionally
    omits ``F``/``H`` and therefore does not pretend to satisfy ``Filter``.
    """
    points = JulierSigmas(1)
    ukf = UnscentedKalmanFilter(
        dim_x=1,
        dim_z=1,
        dt=1.0,
        hx=lambda x, *a, **kw: x,
        fx=lambda x, *a, **kw: x,
        points=points,
    )
    assert not hasattr(ukf, "F")
    assert not hasattr(ukf, "H")


# ---------------------------------------------------------------------------
# NamedTuple shapes
# ---------------------------------------------------------------------------


def test_batch_output_field_names_are_stable():
    assert BatchOutput._fields == ("means", "covs", "means_prior", "covs_prior")
    assert Batch2Output._fields == ("means", "covs")
    assert RtsOutput._fields == ("x", "P", "K", "P_prior")
    assert UkfRtsOutput._fields == ("x", "P", "K")
    assert SmootherOutput._fields == ("x_smoothed", "P_smoothed")
    assert IMMOutput._fields == ("x", "P", "mu")


def test_kalman_batch_filter_returns_named_tuple():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = np.eye(2) * 0.01
    kf.R = np.array([[0.1]])
    kf.x = np.array([0.0, 1.0])
    kf.P = np.eye(2)

    out = kf.batch_filter(_synthetic_zs(12))
    assert isinstance(out, BatchOutput)
    assert out.means.shape[0] == 12
    assert out.covs.shape[0] == 12
    # Legacy positional unpacking must keep working for notebooks.
    means, covs, means_prior, covs_prior = out
    np.testing.assert_allclose(means, out.means)
    np.testing.assert_allclose(covs, out.covs)
    np.testing.assert_allclose(means_prior, out.means_prior)
    np.testing.assert_allclose(covs_prior, out.covs_prior)


def test_kalman_rts_smoother_returns_named_tuple():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = np.eye(2) * 0.01
    kf.R = np.array([[0.1]])
    kf.x = np.array([0.0, 1.0])
    kf.P = np.eye(2)

    means, covs, _, _ = kf.batch_filter(_synthetic_zs(12))
    smooth = kf.rts_smoother(means, covs)
    assert isinstance(smooth, RtsOutput)
    x, P, K, P_prior = smooth
    np.testing.assert_allclose(x, smooth.x)
    np.testing.assert_allclose(P, smooth.P)
    np.testing.assert_allclose(K, smooth.K)
    np.testing.assert_allclose(P_prior, smooth.P_prior)


def test_hinfinity_batch_filter_uses_batch2_output():
    hf = HInfinityFilter(dim_x=2, dim_z=1, dim_u=0, gamma=1.0)
    hf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    hf.H = np.array([[1.0, 0.0]])
    hf.Q = np.eye(2) * 0.01
    out = hf.batch_filter(_synthetic_zs(10))
    assert isinstance(out, Batch2Output)
    means, covs = out
    np.testing.assert_allclose(means, out.means)
    np.testing.assert_allclose(covs, out.covs)


# ---------------------------------------------------------------------------
# RNG resolution
# ---------------------------------------------------------------------------


def test_resolve_rng_returns_the_same_generator():
    generator = np.random.default_rng(7)
    assert resolve_rng(generator) is generator


def test_resolve_rng_falls_back_to_a_fresh_generator():
    first = resolve_rng()
    second = resolve_rng()
    assert isinstance(first, np.random.Generator)
    assert first is not second


def test_resolve_rng_accepts_none_explicitly():
    result = resolve_rng(None)
    assert isinstance(result, np.random.Generator)


# ---------------------------------------------------------------------------
# Resampler contract (duck-typed callables)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resampler",
    [
        resamplers.systematic_resample,
        resamplers.multinomial_resample,
        resamplers.stratified_resample,
        resamplers.residual_resample,
    ],
    ids=["systematic", "multinomial", "stratified", "residual"],
)
def test_resampler_accepts_injected_generator(resampler):
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    a = resampler(weights, rng=np.random.default_rng(1234))
    b = resampler(weights, rng=np.random.default_rng(1234))
    np.testing.assert_array_equal(a, b)
