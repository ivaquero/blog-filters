import math

import numpy as np

from filters.solvers import rk4


class _VelocityFn:
    """Callable integrator helper that carries a mutable initial velocity.

    Signature matches what :func:`filters.solvers.rk4` expects:
    ``fn(x, t, acc=default_acc) -> velocity_at_time_t``.
    """

    __slots__ = ("default_acc", "vel")

    def __init__(self, vel: float = 0.0, default_acc: float = 0.0):
        self.vel = vel
        self.default_acc = default_acc

    def __call__(self, x, t, acc=None):
        a = self.default_acc if acc is None else acc
        return self.vel + a * t


class BallTrajectory2D:
    def __init__(self, x0, y0, vel, theta_deg=0.0, g=9.8, noise=None, seed=None):
        if noise is None:
            noise = [0.0, 0.0]
        self.x = x0
        self.y = y0
        self.t = 0
        theta = math.radians(theta_deg)
        self._xvel = _VelocityFn(math.cos(theta) * vel, default_acc=0.0)
        self._yvel = _VelocityFn(math.sin(theta) * vel, default_acc=-g)
        self.g = g
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def step(self, dt):
        self.x = rk4(self.x, self.t, dt, self._xvel)
        self.y = rk4(self.y, self.t, dt, self._yvel)
        self.t += dt
        return (
            self.x + self.rng.standard_normal() * self.noise[0],
            self.y + self.rng.standard_normal() * self.noise[1],
        )


class BaseballPath:
    def __init__(self, x0, y0, launch_angle_deg, vel_ms, noise=(1.0, 1.0), seed=None):
        """Create 2D baseball path object"""
        omega = math.radians(launch_angle_deg)
        self.v_x = vel_ms * math.cos(omega)
        self.v_y = vel_ms * math.sin(omega)
        self.x = x0
        self.y = y0
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def drag_force(self, vel):
        """Returns the force on a baseball due to air drag at
        the specified vel. Units are SI
        """
        B_m = 0.0039 + 0.0058 / (1.0 + math.exp((vel - 35.0) / 5.0))
        return B_m * vel

    def update(self, dt, vel_wind=0.0):
        """Compute the ball position based on the specified time
        step and wind vel. Returns (x, y) position tuple.
        """
        # Euler equations for x and y
        self.x += self.v_x * dt
        self.y += self.v_y * dt
        # force due to air drag
        v_x_wind = self.v_x - vel_wind
        v = np.sqrt(v_x_wind**2 + self.v_y**2)
        F = self.drag_force(v)
        # Euler's equations for vel
        self.v_x = self.v_x - F * v_x_wind * dt
        self.v_y = self.v_y - 9.81 * dt - F * self.v_y * dt
        return (
            self.x + self.rng.standard_normal() * self.noise[0],
            self.y + self.rng.standard_normal() * self.noise[1],
        )
