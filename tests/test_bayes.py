"""Regression tests for :mod:`filters.bayes` — the scalar Bayesian update.

``prior_estimate`` used to mix units: the beliefs carry a *standard deviation*
(``stats.gaussian.sigma``) while ``R`` and the returned stacks were meant to be
*variances*, so ``plot_kf(..., var=priors[:, 1])`` was fed a σ and the
"Variance" panels plotted a σ. One convention now: beliefs are σ, everything
handed back is a variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters.bayes import prior_estimate
from filters.stats import gaussian


def test_r_is_a_measurement_variance():
    """``R = 4`` must behave like σ = 2, not σ = 4."""
    # σ_x² = 10² = 100, process σ = 1 → prior variance 101.
    priors, _estimates, ps = prior_estimate(
        gaussian(0.0, 10.0), gaussian(1.0, 1.0), [1.0], R=4.0
    )

    assert priors[0, 0] == pytest.approx(1.0)
    assert priors[0, 1] == pytest.approx(101.0)
    # posterior variance = 101 * 4 / (101 + 4)
    assert ps[0, 1] == pytest.approx(101.0 * 4.0 / 105.0)


def test_posterior_mean_matches_the_kalman_gain_form():
    prior_mean, prior_var = 1.0, 101.0
    z, R = 4.0, 4.0

    _, _estimates, ps = prior_estimate(
        gaussian(0.0, 10.0), gaussian(1.0, 1.0), [z], R=R
    )

    gain = prior_var / (prior_var + R)
    assert ps[0, 0] == pytest.approx(prior_mean + gain * (z - prior_mean))


def test_returned_stacks_are_variances_not_deviations():
    """The stacks feed ``plot_kf(var=...)``, which takes a variance."""
    _, _estimates, ps = prior_estimate(
        gaussian(0.0, 10.0), gaussian(1.0, 1.0), [1.0, 1.0, 1.0], R=4.0
    )

    assert np.all(ps[:, 1] >= 0.0)
    # A σ of ~1.96 would be variance ~3.85; the stack must not hold the σ.
    assert ps[0, 1] > 1.0


def test_estimate_stack_carries_the_same_posterior_as_ps():
    """``estimates`` is a legacy duplicate kept for the 3-tuple unpacking."""
    _, estimates, ps = prior_estimate(
        gaussian(0.0, 10.0), gaussian(1.0, 1.0), [1.0, 2.0], R=4.0
    )

    assert np.allclose(estimates, ps)


def test_show_steps_prints_the_first_and_last_step(capsys):
    prior_estimate(
        gaussian(0.0, 10.0), gaussian(1.0, 1.0), [1.0, 2.0, 3.0], R=4.0, show_steps=True
    )

    out = capsys.readouterr().out
    assert "final estimate" in out
    assert "actual final value" in out
