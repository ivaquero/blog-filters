"""Poisson clutter model used by tracking examples."""

from __future__ import annotations

import numpy as np

from filters._typing import resolve_rng


class PoissonClutter2D:
    """Poisson clutter model.

    The number of clutters, ``k``, follows a Poisson distribution with mean
    ``density * range_**2``; the ``k`` clutters are then spread uniformly
    within a square of side ``2 * range_`` centred on the observed position.
    """

    def __init__(self, density: float, range_: float):
        self.density = density
        self.range_ = range_

    def arise(
        self, center: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw one clutter cloud around ``center``."""
        generator = resolve_rng(rng)
        num_clutter = generator.poisson(lam=self.density * (self.range_**2))
        if num_clutter == 0:
            return np.empty((0, 2))

        return center + generator.uniform(
            low=-self.range_, high=self.range_, size=(num_clutter, 2)
        )
