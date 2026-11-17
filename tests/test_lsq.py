"""Regression tests for :mod:`filters.lsq` (recursive least squares filter).

Unreferenced module. Two behaviours are pinned:

* the recursion itself (order 0/1/2 reproduce polynomial inputs exactly),
* ``errors()`` no longer divides by zero at the first steps — ``n(n²-1)`` and
  ``(n²-1)(n²-4)`` vanish at n = 1 and n = 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.lsq import LeastSquaresFilter


def test_order_zero_converges_to_the_mean():
    f = LeastSquaresFilter(dt=1.0, order=0, noise_sigma=1.0)
    for z in (1.0, 2.0, 3.0, 4.0):
        f.update(z)

    assert np.isclose(f.x[0], 2.5)


def test_order_one_recovers_a_ramp():
    f = LeastSquaresFilter(dt=1.0, order=1)
    for i in range(1, 11):
        f.update(float(i))

    assert np.isclose(f.x[0], 10.0, atol=1e-6)
    assert np.isclose(f.x[1], 1.0, atol=1e-6)


def test_order_two_recovers_a_quadratic():
    f = LeastSquaresFilter(dt=1.0, order=2)
    for i in range(1, 21):
        f.update(0.5 * float(i) ** 2)

    assert np.isclose(f.x[0], 0.5 * 20.0**2, atol=1e-4)
    assert np.isclose(f.x[1], 20.0, atol=1e-4)
    assert np.isclose(f.x[2], 1.0, atol=1e-4)


def test_update_returns_the_state_vector():
    f = LeastSquaresFilter(dt=1.0, order=1)
    out = f.update(5.0)
    assert out is f.x
    assert out.shape == (2,)


@pytest.mark.parametrize("order", [1, 2])
def test_errors_at_first_step_are_finite(order):
    """n = 1 (and n = 2 for order 2) made the derivative terms divide by zero."""
    f = LeastSquaresFilter(dt=1.0, order=order, noise_sigma=2.0)

    for step in range(1, 4):
        f.update(float(step))
        error, std = f.errors()

        assert error.shape == (order + 1,)
        assert std.shape == (order + 1,)
        assert np.all(np.isfinite(error))
        assert np.all(np.isfinite(std))


def test_errors_before_any_update_are_zero():
    f = LeastSquaresFilter(dt=1.0, order=2, noise_sigma=1.0)
    error, std = f.errors()

    assert np.all(error == 0.0)
    assert np.all(std == 0.0)


def test_order_zero_error_matches_the_sample_mean_formula():
    f = LeastSquaresFilter(dt=1.0, order=0, noise_sigma=3.0)
    for _ in range(9):
        f.update(1.0)

    error, std = f.errors()

    assert np.isclose(error[0], 3.0 / np.sqrt(9))
    assert np.isclose(std[0], 3.0 / np.sqrt(9))


def test_reset_restores_the_initial_state():
    f = LeastSquaresFilter(dt=1.0, order=1)
    f.update(1.0)
    f.update(2.0)

    f.reset()

    assert f.n == 0
    assert np.all(f.x == 0.0)
    assert np.all(f.K == 0.0)


@pytest.mark.parametrize("order", [-1, 3])
def test_order_out_of_range_is_rejected(order):
    with pytest.raises(ValueError, match="order"):
        LeastSquaresFilter(dt=1.0, order=order)
