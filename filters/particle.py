"""Particle-filter primitives.

Randomness flows through an injected ``numpy.random.Generator`` so tests and
notebooks own the stream; the legacy ``seed=`` knobs are gone. When ``rng`` is
omitted we build a fresh unseeded generator, which matches the pattern used by
``filters._typing.resolve_rng``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._typing import resolve_rng


class ParticleFilter:
    """Object-centric particle filter with uniform initial weights."""

    def __init__(
        self, N: int, x_dim: float, y_dim: float, rng: np.random.Generator | None = None
    ):
        self.rng = resolve_rng(rng)
        self.particles = np.empty((N, 3))  # x, y, heading
        self.N = N
        self.x_dim = x_dim
        self.y_dim = y_dim

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1.0 / N)

        self.particles[:, 0] = self.rng.uniform(0, x_dim, size=N)
        self.particles[:, 1] = self.rng.uniform(0, y_dim, size=N)
        self.particles[:, 2] = self.rng.uniform(0, 2 * np.pi, size=N)

    def predict(self, u, std, dt: float = 1):
        """Move according to control input ``u`` with per-step noise ``std``."""
        self.particles[:, 2] += u[0] + self.rng.standard_normal(self.N) * std[0]
        self.particles[:, 2] %= 2 * np.pi

        dist = (u[1] * dt) + (self.rng.standard_normal(self.N) * std[1])
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * dist
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * dist

    def weight(self, z, var):
        dist = np.sqrt(
            (self.particles[:, 0] - z[0]) ** 2 + (self.particles[:, 1] - z[1]) ** 2
        )

        # simplification assumes variance is invariant to world projection
        n = stats.norm(0, np.sqrt(var))
        prob = n.pdf(dist)

        # particles far from a measurement will give us 0.0 for a probability due to floating point limits. Once we hit zero we can never recover, so add some small nonzero value to all points.
        prob += 1.0e-12
        self.weights *= prob
        self.weights /= sum(self.weights)  # normalize

    def neff(self) -> float:
        return 1.0 / np.sum(np.square(self.weights))

    def resample(self):
        p = np.zeros((self.N, 3))
        w = np.zeros(self.N)

        cumsum = np.cumsum(self.weights)
        for i in range(self.N):
            index = np.searchsorted(cumsum, self.rng.random())
            p[i] = self.particles[index]
            w[i] = self.weights[index]

        self.particles = p
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """Return mean and variance of the weighted particle positions."""
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu) ** 2, weights=self.weights, axis=0)

        return mu, var


def neff(weights):
    return 1.0 / np.sum(np.square(weights))


def pf_estimate(particles, weights):
    """Return mean and variance of the weighted particles."""
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def pf_update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= stats.norm(distance, R).pdf(z[i])
        weights += 1.0e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize


def pf_predict(
    particles, u, std, dt: float = 1.0, rng: np.random.Generator | None = None
):
    """Move particles according to control ``u`` with per-step noise ``std``.

    Heading is updated first, then each particle advances along its (noisy)
    commanded direction. Pass an explicit ``rng`` to keep figure output
    reproducible across calls.
    """
    generator = resolve_rng(rng)
    N = len(particles)

    # update heading
    particles[:, 2] += u[0] + (generator.standard_normal(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (generator.standard_normal(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
