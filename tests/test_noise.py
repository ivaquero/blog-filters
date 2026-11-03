"""Regression tests for :mod:`models.noise`.

Covers the discrete-time white-noise Q generator and the missing ``raise``
bug that was patched in Phase 1. Also guards the block-diagonal expansion used
by the CV/CA scenario notebooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.noise import white_noise_discrete


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_white_noise_discrete_is_symmetric_positive_semidefinite(dim):
    Q = white_noise_discrete(dim=dim, dt=0.5, var=1.0)
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-9)


def test_white_noise_discrete_scales_with_variance():
    q1 = white_noise_discrete(dim=2, dt=1.0, var=1.0)
    q4 = white_noise_discrete(dim=2, dt=1.0, var=4.0)
    assert np.allclose(q4, 4 * q1)


def test_white_noise_discrete_block_diagonal_expansion():
    Q_single = white_noise_discrete(dim=2, dt=1.0, var=1.0, block_size=1)
    Q_double = white_noise_discrete(dim=2, dt=1.0, var=1.0, block_size=2)
    assert Q_double.shape == (2 * Q_single.shape[0], 2 * Q_single.shape[1])
    # Off-diagonal block should be zero (independent x/y axes).
    n = Q_single.shape[0]
    assert np.allclose(Q_double[:n, n:], np.zeros((n, n)))


@pytest.mark.parametrize("bad_dim", [0, 1, 5, 6])
def test_white_noise_discrete_rejects_bad_dim(bad_dim):
    with pytest.raises(ValueError, match="dim"):
        white_noise_discrete(dim=bad_dim)
