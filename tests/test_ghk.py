"""Regression tests for :mod:`filters.ghk` (g, g-h and g-h-k filters).

Pinned defects:

* ``GFilter.update`` declared ``h=None`` and multiplied by it, so every call
  that relied on the default raised ``TypeError``.
* ``GHKFilter.batch_filter`` was a copy of the g-h loop: ``k`` never entered
  the recursion and the returned stack had two columns instead of three.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.ghk import GFilter, GHFilter, GHKFilter


def test_gfilter_update_does_not_need_h():
    gf = GFilter(x0=0.0, dx=1.0, dt=1.0, g=0.5)

    x, dx = gf.update(2.0)

    # prediction: 0 + 1*1 = 1; residual 1; x = 1 + 0.5*1
    assert np.isclose(x, 1.5)
    # h = 0 leaves the velocity estimate untouched
    assert np.isclose(dx, 1.0)


def test_gfilter_matches_ghfilter_with_zero_h():
    data = np.arange(1.0, 11.0)
    g_only = GFilter(0.0, 1.0, 1.0, 0.3).batch_filter(data)
    gh_zero = GHFilter(0.0, 1.0, 1.0, 0.3, 0.0).batch_filter(data)

    assert np.allclose(g_only, gh_zero)


def test_gfilter_repr_uses_its_own_name():
    assert repr(GFilter(0.0, 1.0, 1.0, 0.3)).startswith("GFilter object")


def test_ghfilter_repr_uses_its_own_name():
    assert repr(GHFilter(0.0, 1.0, 1.0, 0.3, 0.1)).startswith("GHFilter object")


def test_ghkfilter_repr_uses_its_own_name():
    ghk = GHKFilter(0.0, 1.0, 0.0, 1.0, 0.3, 0.1, 0.01)
    assert repr(ghk).startswith("GHKFilter object")


def test_ghk_batch_filter_returns_three_states():
    data = np.arange(1.0, 21.0)
    estimates = GHKFilter(0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 0.05).batch_filter(data)

    assert estimates.shape == (len(data) + 1, 3)


def test_ghk_batch_filter_applies_k():
    """With k ignored, two different gains produced identical output."""
    data = np.arange(1.0, 21.0)
    slow = GHKFilter(0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 0.05).batch_filter(data)
    fast = GHKFilter(0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 0.5).batch_filter(data)

    assert not np.allclose(slow, fast)
    assert not np.allclose(slow[1:, 2], 0.0), "acceleration state must be populated"


def test_ghk_batch_filter_matches_stepwise_update():
    data = np.linspace(0.0, 9.0, 10)
    gains = {"x0": 0.0, "dx": 0.5, "ddx": 0.0, "dt": 1.0, "g": 0.4, "h": 0.2, "k": 0.1}

    batch = GHKFilter(**gains).batch_filter(data)

    stepwise = GHKFilter(**gains)
    rows = [(stepwise.x, stepwise.dx, stepwise.ddx)]
    for z in data:
        stepwise.update(z)
        rows.append((stepwise.x, stepwise.dx, stepwise.ddx))

    assert np.allclose(batch, np.array(rows))


def test_ghk_batch_filter_returns_predictions_when_asked():
    data = [1.0, 2.0, 3.0]
    estimates, predictions = GHKFilter(0.0, 0.0, 0.0, 1.0, 0.5, 0.2, 0.1).batch_filter(
        data, save_preds=True
    )

    assert predictions.shape == (len(data),)
    assert estimates.shape == (len(data) + 1, 3)


def test_ghk_update_returns_position_and_velocity():
    ghk = GHKFilter(0.0, 1.0, 0.0, 1.0, 0.5, 0.2, 0.1)
    x, dx = ghk.update(3.0)
    assert np.isclose(x, ghk.x)
    assert np.isclose(dx, ghk.dx)


def test_critical_damping_rejects_out_of_range_theta():
    from filters.ghk import critical_damping_params

    with pytest.raises(ValueError, match="theta"):
        critical_damping_params(1.5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GHFilter(0.0, 1.0, 1.0, 0.5, 0.2),
        lambda: GHKFilter(0.0, 1.0, 0.0, 1.0, 0.5, 0.2, 0.1),
    ],
)
def test_update_records_the_measurement(factory):
    """``repr`` and KFSaver used to report z=0 regardless of the input."""
    filt = factory()

    filt.update(2.0)

    assert filt.z == 2.0
    assert filt.y == 2.0 - filt.x_pred


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GHFilter(0.0, 1.0, 1.0, 0.5, 0.2),
        lambda: GHKFilter(0.0, 1.0, 0.0, 1.0, 0.5, 0.2, 0.1),
    ],
)
def test_batch_filter_leaves_state_at_the_last_estimate(factory):
    data = [1.0, 2.0, 3.0]
    filt = factory()

    estimates = filt.batch_filter(data)

    assert np.allclose(filt.x, estimates[-1, 0])
    assert np.allclose(filt.dx, estimates[-1, 1])
    assert filt.z == data[-1]


def test_batch_filter_saver_sees_the_current_state():
    class Saver:
        def __init__(self, filt):
            self.filt = filt
            self.xs = []

        def save(self):
            self.xs.append(self.filt.x)

    filt = GHFilter(0.0, 1.0, 1.0, 0.5, 0.2)
    saver = Saver(filt)
    estimates = filt.batch_filter([1.0, 2.0, 3.0], saver=saver)

    assert np.allclose(saver.xs, estimates[1:, 0])
