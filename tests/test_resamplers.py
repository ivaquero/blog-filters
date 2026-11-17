"""Regression tests for :mod:`filters.resamplers`.

``residual_resample`` normalised the residual by its own sum without checking
it; whenever every ``N * w`` was already an integer (or the caller's weights
did not sum to one) that denominator was zero, so the cumulative sum became
``NaN`` and ``searchsorted`` returned garbage indices.
"""

from __future__ import annotations

import numpy as np
import pytest

from filters import resamplers

RESAMPLERS = [
    resamplers.residual_resample,
    resamplers.stratified_resample,
    resamplers.systematic_resample,
    resamplers.multinomial_resample,
]


@pytest.fixture
def weights() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize("resample", RESAMPLERS)
def test_returns_one_valid_index_per_weight(resample, weights):
    indexes = resample(weights, rng=np.random.default_rng(1))

    assert indexes.shape == weights.shape
    assert indexes.dtype.kind == "i"
    assert indexes.min() >= 0
    assert indexes.max() < len(weights)


@pytest.mark.parametrize("resample", RESAMPLERS)
def test_is_reproducible_for_a_seeded_generator(resample, weights):
    first = resample(weights, rng=np.random.default_rng(7))
    second = resample(weights, rng=np.random.default_rng(7))

    assert np.array_equal(first, second)


def test_residual_resample_survives_a_zero_residual():
    """Every N*w is an integer here, so the residual sums to zero."""
    indexes = resamplers.residual_resample(
        np.array([0.25, 0.0, 0.0, 0.0]), rng=np.random.default_rng(0)
    )

    assert np.all((indexes >= 0) & (indexes < 4))


def test_residual_resample_survives_unnormalised_weights():
    """Weights summing to less than one also zeroed the denominator."""
    indexes = resamplers.residual_resample(
        np.array([0.1, 0.0, 0.0, 0.0]), rng=np.random.default_rng(0)
    )

    assert np.all((indexes >= 0) & (indexes < 4))


def test_residual_resample_keeps_heavy_particles():
    """The deterministic part of the algorithm copies large weights first."""
    indexes = resamplers.residual_resample(
        np.array([0.7, 0.1, 0.1, 0.1]), rng=np.random.default_rng(3)
    )

    assert np.count_nonzero(indexes == 0) >= 2


def test_systematic_resample_uses_a_single_offset():
    """Adjacent positions differ by exactly 1/N, by construction."""
    indexes = resamplers.systematic_resample(
        np.full(8, 0.125), rng=np.random.default_rng(5)
    )

    assert np.array_equal(np.diff(indexes), np.ones(7))


def test_resample_from_index_copies_particles_and_resets_weights():
    particles = np.arange(12, dtype=float).reshape(4, 3)
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    resamplers.resample_from_index(particles, weights, np.array([3, 3, 1, 0]))

    assert np.array_equal(particles[0], [9.0, 10.0, 11.0])
    assert np.allclose(weights, 0.25)


@pytest.mark.parametrize(
    "bad_weights",
    [
        np.array([0.3, 0.3, 0.3, 0.0]),  # sums to 0.9
        np.array([0.2] * 4),  # sums to 0.8
        np.array([0.6, 0.6, 0.6, 0.6]),  # sums to 2.4
    ],
)
@pytest.mark.parametrize("resample", RESAMPLERS)
def test_unnormalised_weights_stay_in_bounds(resample, bad_weights):
    """A CDF below 1.0 used to let searchsorted/while return index N."""
    indexes = resample(bad_weights, rng=np.random.default_rng(11))

    assert indexes.min() >= 0
    assert indexes.max() < len(bad_weights)


def test_uniform_fallback_for_zero_weights():
    for resample in RESAMPLERS:
        indexes = resample(np.zeros(5), rng=np.random.default_rng(2))
        assert np.all((indexes >= 0) & (indexes < 5))


def test_residual_uses_fractional_part_in_count_units():
    """N*w - floor(N*w) must decide the residual slots, not w - floor(N*w)."""
    # N=10: each of the first four particles gets 2 deterministic copies and
    # 0.5 residual share, so the last two slots must also go to those four.
    # The old (w - copies) formula was negative everywhere and fell back to
    # uniform, which would have selected the zero-weight particles here.
    weights = np.array([0.25] * 4 + [0.0] * 6)
    counts = np.bincount(
        resamplers.residual_resample(weights, rng=np.random.default_rng(4)),
        minlength=10,
    )

    assert counts.sum() == 10
    assert counts[4:].sum() == 0
    assert counts[:4].min() >= 2
