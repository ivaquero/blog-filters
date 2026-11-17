"""Regression tests for :mod:`filters.stats`.

Two defects are pinned here:

* ``gaussian`` stored σ in a field named ``var``, so ``plot_bayes`` took a
  square root of it again and ``gaussian_product`` fed σ into the variance
  formula.
* ``gaussian_product`` therefore returned μ = 3.333 for ``N(0, σ=1) ⊗
  N(10, σ=2)``, where the conjugate update gives μ = 2 and σ = 0.894.
"""

from __future__ import annotations

import numpy as np

from filters.stats import gaussian, gaussian_product, gaussian_sum


def test_gaussian_field_is_named_sigma():
    g = gaussian(mean=0.0, sigma=2.0)
    assert g.sigma == 2.0
    assert not hasattr(g, "var")


def test_gaussian_repr_reports_sigma():
    # σ = 2 → σ² = 4; the label must say σ, and the value must be 2.
    assert "σ=2.000" in repr(gaussian(0.0, 2.0))


def test_gaussian_product_uses_variances():
    var1, var2 = 1.0**2, 2.0**2
    product = gaussian_product(gaussian(0.0, 1.0), gaussian(10.0, 2.0))

    expected_mean = (var2 * 0.0 + var1 * 10.0) / (var1 + var2)
    expected_sigma = np.sqrt(var1 * var2 / (var1 + var2))

    assert np.isclose(product.mean, expected_mean)
    assert np.isclose(product.mean, 2.0)
    assert np.isclose(product.sigma, expected_sigma)


def test_gaussian_product_is_commutative():
    a, b = gaussian(1.0, 0.5), gaussian(-2.0, 3.0)
    left, right = gaussian_product(a, b), gaussian_product(b, a)
    assert np.isclose(left.mean, right.mean)
    assert np.isclose(left.sigma, right.sigma)


def test_gaussian_product_of_equal_precision_is_midpoint():
    product = gaussian_product(gaussian(0.0, 1.0), gaussian(4.0, 1.0))
    assert np.isclose(product.mean, 2.0)
    assert np.isclose(product.sigma, np.sqrt(0.5))


def test_gaussian_product_shrinks_uncertainty():
    product = gaussian_product(gaussian(0.0, 2.0), gaussian(1.0, 2.0))
    assert product.sigma < 2.0


def test_gaussian_sum_adds_variances():
    # σ = 3 and σ = 4 add in quadrature to 5, not to 7.
    total = gaussian_sum(gaussian(1.0, 3.0), gaussian(2.0, 4.0))
    assert np.isclose(total.mean, 3.0)
    assert np.isclose(total.sigma, 5.0)
