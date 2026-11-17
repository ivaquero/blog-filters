"""Regression tests for :mod:`filters.kalman_fm` (fading memory filter).

The module was unreferenced, which hid a ``TypeError``: the update step used
``@`` for scalar multiplication, so a plain float state (the documented usage)
could not be filtered at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.kalman_fm import FadingMemoryFilter


def test_order_zero_converges_to_a_constant():
    f = FadingMemoryFilter(0.0, dt=1.0, order=0, beta=0.6)
    for _ in range(200):
        f.update(10.0)

    assert np.isclose(f.x[0], 10.0, atol=1e-3)


def test_order_one_accepts_python_float_measurements():
    """``dx @ dt`` on a numpy scalar worked; on a float it raised."""
    f = FadingMemoryFilter(0.0, dt=1.0, order=1, beta=0.6)

    f.update(1.0)

    assert f.x.shape == (2,)


def test_order_one_tracks_a_ramp():
    f = FadingMemoryFilter(0.0, dt=1.0, order=1, beta=0.6)
    for i in range(1, 201):
        f.update(float(i))

    assert np.isclose(f.x[1], 1.0, atol=0.05)
    assert np.isclose(f.x[0], 200.0, atol=2.0)


def test_order_two_updates_acceleration():
    f = FadingMemoryFilter([0.0, 0.0, 0.0], dt=1.0, order=2, beta=0.7)
    for i in range(1, 201):
        f.update(0.5 * float(i) ** 2)

    # d²/dt² of 0.5 t² is 1
    assert np.isclose(f.x[2], 1.0, atol=0.1)


def test_order_zero_tracks_with_float_and_array_input():
    scalar = FadingMemoryFilter(0.0, 1.0, 0, 0.5)
    scalar.update(4.0)

    array = FadingMemoryFilter(np.zeros(1), 1.0, 0, 0.5)
    array.update(np.array([4.0]))

    assert np.allclose(scalar.x, array.x)


def test_coefficients_have_one_entry_per_state():
    for order in (0, 1, 2):
        f = FadingMemoryFilter(0.0, 1.0, order, 0.6)
        assert f.P.shape == (order + 1,)
        assert f.e.shape == (order + 1,)


@pytest.mark.parametrize("order", [-1, 3])
def test_order_out_of_range_is_rejected(order):
    with pytest.raises(ValueError, match="order"):
        FadingMemoryFilter(0.0, 1.0, order, 0.6)


@pytest.mark.parametrize("beta", [0.0, 1.0, -0.2, 1.5])
def test_beta_out_of_range_is_rejected(beta):
    """beta=1 divided by zero; beta=0 silently turned fading off."""
    with pytest.raises(ValueError, match="beta"):
        FadingMemoryFilter(0.0, 1.0, 0, beta)
