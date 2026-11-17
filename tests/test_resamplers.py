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
