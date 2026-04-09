import numpy as np


class PoissonClutter2D:
    """Poisson clutter model.

    The number of clutters, k, follows a poisson distribution.
    The k clutters are uniformaly spatially distributed.
    """

    def __init__(self, density, range_):
        self.dentity = density
        self.range_ = range_

    def arise(self, center, seed=42):
        rng = np.random.default_rng(seed)
        num_clutter = rng.poisson(lam=self.dentity * (self.range_**2))
        if num_clutter == 0:
            return np.empty((0, 2))

        return center + rng.uniform(
            low=-self.range_, high=self.range_, size=(num_clutter, 2)
        )
