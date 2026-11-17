from typing import NamedTuple

import numpy as np
from numpy import linalg


class gaussian(NamedTuple):
    """Scalar Gaussian parameterised by mean and **standard deviation**.

    The second field is σ, not σ². The historical name for it was `var`,
    which silently disagreed with ``plot_bayes`` (that code did
    ``g.var**0.5``) and let ``gaussian_product`` apply the variance formula
    to standard deviations. Use ``g.sigma**2`` when a variance is needed.
    """

    mean: float
    sigma: float

    def __repr__(self) -> str:
        return f"N(μ={self.mean:.3f}, σ={self.sigma:.3f})"


def gaussian_product(g1: gaussian, g2: gaussian) -> gaussian:
    """Product of two Gaussians — Bayes' rule for a conjugate prior.

    The parameters are combined through their variances; the return value
    keeps the σ convention. ``N(μ=0, σ=1) ⊗ N(μ=10, σ=2)`` gives
    ``N(μ=2.000, σ=0.894)``, which is the variance-weighted mean, not the
    σ-weighted one the previous implementation produced (μ=3.333).
    """

    var1 = g1.sigma**2
    var2 = g2.sigma**2
    var = var1 + var2

    mean = (var2 * g1.mean + var1 * g2.mean) / var
    return gaussian(mean, np.sqrt(var1 * var2 / var))


def gaussian_sum(g1: gaussian, g2: gaussian) -> gaussian:
    """Sum of two independent Gaussians: means add, variances add."""

    return gaussian(g1.mean + g2.mean, np.sqrt(g1.sigma**2 + g2.sigma**2))


def NEES(xs, est_xs, ps):
    est_err = xs - est_xs
    return [x.T @ linalg.inv(p) @ x for x, p in zip(est_err, ps, strict=True)]


def multi_gaussian_product(mean1, cov1, mean2, cov2):
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)
    mean1 = np.asarray(mean1)
    mean2 = np.asarray(mean2)

    sum_inv = np.linalg.inv(cov1 + cov2)
    mean = cov2 @ sum_inv @ mean1 + cov1 @ sum_inv @ mean2
    cov = cov1 @ sum_inv @ cov2

    return mean, cov
