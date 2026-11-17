"""Regression tests for :mod:`filters.particle`."""

from __future__ import annotations

import numpy as np

from filters.particle import ParticleFilter, pf_update


def test_pf_update_treats_R_as_variance():
    """scipy's scale is a standard deviation; the package passes variance."""
    particles = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    weights = np.full(2, 0.5)
    R = 4.0  # sigma = 2

    pf_update(particles, weights, [0.0], R, [(0.0, 0.0)])

    # Gaussian likelihoods for distances 0 and 2 under sigma=2.
    sigma = np.sqrt(R)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    likelihoods = np.array([norm, norm * np.exp(-0.5 * (2.0 / sigma) ** 2)])
    expected = likelihoods / likelihoods.sum()
    assert np.allclose(weights, expected)


def test_pf_update_matches_weight_method_for_one_landmark():
    """``weight(z, var)`` already used sqrt(var); the function must agree."""
    rng = np.random.default_rng(5)
    pf = ParticleFilter(20, 10.0, 10.0, rng=rng)
    # Keep particles on the positive x axis: pf_update uses the range from
    # the landmark (unsigned), weight() uses the signed offset to the obs.
    pf.particles[:, 0] = rng.uniform(0.0, 5.0, 20)
    pf.particles[:, 1] = 0.0

    distances = pf.particles[:, 0]
    z = 2.0
    var = 1.5

    expected = np.full(20, 1.0 / 20)
    pf_update(pf.particles.copy(), expected, [z], var, [(0.0, 0.0)])

    # weight() compares the particle to the 2-D observation point (z, 0).
    other = ParticleFilter(20, 10.0, 10.0, rng=rng)
    other.particles = pf.particles.copy()
    other.weights = np.full(20, 1.0 / 20)
    other.weight((z, 0.0), var)

    # pf_update drops a 1e-300 floor, weight() a 1e-12 floor; with these
    # distances no particle underflows, so the posteriors coincide.
    assert np.allclose(expected, other.weights, atol=1e-9)
    assert distances.shape == (20,)


def test_resample_is_vectorised_and_resets_weights():
    """Same seeded stream as the old loop, but one rng.random() call batch."""
    pf = ParticleFilter(50, 10.0, 10.0, rng=np.random.default_rng(9))
    before = pf.particles.copy()

    pf.resample()

    assert pf.particles.shape == before.shape
    assert np.allclose(pf.weights, 1.0 / pf.N)
    # Every resampled row must be one of the original particles.
    assert np.all(
        np.any(np.all(pf.particles[:, None, :] == before[None, :, :], axis=2), axis=1)
    )
    # The result is a copy, not a view into the pre-resample array.
    assert not np.shares_memory(pf.particles, before)


def test_resample_survives_unnormalised_weights():
    pf = ParticleFilter(20, 10.0, 10.0, rng=np.random.default_rng(3))
    pf.weights[:] = 0.2

    pf.resample()

    assert np.isfinite(pf.particles).all()
