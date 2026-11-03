"""Shared fixtures for the blog-filters regression suite.

Keeps every test deterministic without touching the module-level NumPy RNG:
each fixture that produces randomness constructs an explicit
``numpy.random.Generator`` so CI runs match author runs bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A freshly seeded Generator for tests that need randomness."""
    return np.random.default_rng(20260101)


@pytest.fixture
def linear_cv_truth() -> tuple[np.ndarray, np.ndarray]:
    """Ten-step constant-velocity trajectory plus a scalar observation stream.

    The measurement noise is baked in with a fixed seed so the arrays never
    drift between runs — golden-value tests can rely on them.
    """
    rng = np.random.default_rng(1234)
    dt = 1.0
    xs = np.array([[i * dt, 1.0] for i in range(10)])
    zs = xs[:, 0] + rng.standard_normal(10) * 0.5
    return xs, zs
