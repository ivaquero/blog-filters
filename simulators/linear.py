"""One- and two-dimensional constant-velocity simulators.

Every class accepts a `numpy.random.Generator` via `rng=` (no more
`seed=` integers scattered through the constructor). Callers build a
`default_rng(seed)` at the top of a script or notebook, hand it to the
simulator, and get byte-for-byte reproducibility across calls.
"""

import numpy as np

from filters._typing import resolve_rng


class DogSimulation:
    """1-D "dog walking" simulator with per-step velocity noise."""

    def __init__(
        self,
        x0: float = 0,
        model_mean: float = 1,
        Q: float = 0,
        R: float = 0,
        rng: np.random.Generator | None = None,
    ):
        self.x = x0
        self.model_mean = model_mean
        self.p_std = np.sqrt(Q)
        self.z_std = np.sqrt(R)
        self.rng = resolve_rng(rng)

    def move(self, dt: float = 1.0, acc: float = 0) -> float:
        model_mean = self.model_mean + self.rng.standard_normal() * self.p_std
        self.x += model_mean * dt
        self.model_mean += acc
        return self.x

    def sense_position(self) -> float:
        return self.x + self.rng.standard_normal() * self.z_std

    def move_and_sense(self, dt: float = 1.0, acc: float = 0) -> float:
        self.move(dt, acc)
        return self.sense_position()

    def simulate(self, dt: float = 1, num: int = 1, acc: float = 0):
        xs = np.array([self.move(dt, acc) for _ in range(num)])
        zs = np.array([self.move_and_sense(dt) for _ in range(num)])
        return xs, zs


class PosSensor:
    """Two-dimensional position sensor with constant-velocity motion."""

    def __init__(
        self,
        pos: tuple[float, float] = (0, 0),
        vel: tuple[float, float] = (0, 0),
        R: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.vel = vel
        self.pos = [pos[0], pos[1]]
        self.R = R
        self.rng = resolve_rng(rng)

    def read(self) -> list[float]:
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [
            self.pos[0] + self.rng.standard_normal() * self.R,
            self.pos[1] + self.rng.standard_normal() * self.R,
        ]


class CVObject:
    """Constant-velocity object with process noise on velocity."""

    def __init__(
        self,
        x0: float = 0,
        vel: float = 1.0,
        Q: float = 0.06,
        R: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.x = x0
        self.vel = vel
        self.Q = Q
        self.R = R
        self.rng = resolve_rng(rng)

    def update(self) -> tuple[float, float]:
        self.vel += self.rng.standard_normal() * self.Q
        self.x += self.vel
        return (self.x, self.vel)

    def sense(self) -> float:
        return self.x + self.rng.standard_normal() * self.R

    def simulate(self, count: int):
        xs, zs = [], []
        for _ in range(count):
            x = self.update()
            z = self.sense()
            xs.append(x)
            zs.append(z)
        return np.array(xs), np.array(zs)


class CAObject:
    """Constant-acceleration object with noise on the acceleration term."""

    def __init__(
        self,
        x0: float = 0,
        vel: float = 1.0,
        acc: float = 0.1,
        Q: float = 0.1,
        R: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.x = x0
        self.vel = vel
        self.acc = acc
        self.Q = Q
        self.R = R
        self.rng = resolve_rng(rng)

    def update(self) -> tuple[float, float, float]:
        self.acc += self.rng.standard_normal() * self.Q
        self.vel += self.acc
        self.x += self.vel
        return (self.x, self.vel, self.acc)

    def sense(self) -> float:
        return self.x + self.rng.standard_normal() * self.R

    def simulate(self, count: int):
        xs, zs = [], []
        for _ in range(count):
            x = self.update()
            z = self.sense()
            xs.append(x)
            zs.append(z)
        return np.asarray(xs), zs
