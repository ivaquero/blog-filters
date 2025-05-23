import numpy as np

from .helpers import pretty_str


class FadingMemoryFilter:
    """Creates a fading memory filter of order 0, 1, or 2.

    The KalmanFilter class also implements a more general fading memory
    filter and should be preferred in most cases. This is probably faster for low order systems.

    References
    ----------
    Paul Zarchan and Howard Musoff. "Fundamentals of Kalman Filtering:
    A Practical Approach" American Institute of Aeronautics and Astronautics, Inc. Fourth Edition. p. 521-536. (2015)
    """

    def __init__(self, x0, dt, order, beta):
        if order < 0 or order > 2:
            error_message = "order must be between 0 and 2"
            raise ValueError(error_message)

        if np.isscalar(x0):
            self.x = np.zeros(order + 1)
            self.x[0] = x0
        else:
            self.x = np.copy(x0)

        self.dt = dt
        self.order = order
        self.beta = beta

        if order == 0:
            self.P = np.array([(1 - beta) / (1 + beta)], dtype=float)
            self.e = np.array([dt * beta / (1 - beta)], dtype=float)

        elif order == 1:
            p11 = (1 - beta) * (1 + 4 * beta + 5 * beta**2) / (1 + beta) ** 3
            p22 = 2 * (1 - beta) ** 3 / (1 + beta) ** 3
            self.P = np.array([p11, p22], dtype=float)

            e = 2 * dt * 2 * (beta / (1 - beta)) ** 2
            de = dt * ((1 + 3 * beta) / (1 - beta))
            self.e = np.array([e, de], dtype=float)

        else:
            p11 = (1 - beta) * (
                (1 + 6 * beta + 16 * beta**2 + 24 * beta**3 + 19 * beta**4)
                / (1 + beta) ** 5
            )

            p22 = (1 - beta) ** 3 * (
                (13 + 50 * beta + 49 * beta**2) / (2 * (1 + beta) ** 5 * dt**2)
            )

            p33 = 6 * (1 - beta) ** 5 / ((1 + beta) ** 5 * dt**4)

            self.P = np.array([p11, p22, p33], dtype=float)

            e = 6 * dt**3 * (beta / (1 - beta)) ** 3
            de = dt**2 * (2 + 5 * beta + 11 * beta**2) / (1 - beta) ** 2
            dde = 6 * dt * (1 + 2 * beta) / (1 - beta)

            self.e = np.array([e, de, dde], dtype=float)

    def __repr__(self):
        return "\n".join([
            "FadingMemoryFilter object",
            pretty_str("dt", self.dt),
            pretty_str("order", self.order),
            pretty_str("beta", self.beta),
            pretty_str("x", self.x),
            pretty_str("P", self.P),
            pretty_str("e", self.e),
        ])

    def update(self, z):
        """Update the filter with measurement z. z must be the same type
        as self.x[0].
        """

        if self.order == 0:
            G = 1 - self.beta
            self.x = self.x + G @ (z - self.x)

        elif self.order == 1:
            G = 1 - self.beta**2
            H = (1 - self.beta) ** 2
            x = self.x[0]
            dx = self.x[1]
            dxdt = dx @ self.dt

            residual = z - (x + dxdt)
            self.x[0] = x + dxdt + G * residual
            self.x[1] = dx + (H / self.dt) * residual

        else:  # order == 2
            G = 1 - self.beta**3
            H = 1.5 * (1 + self.beta) * (1 - self.beta) ** 2
            K = 0.5 * (1 - self.beta) ** 3

            x = self.x[0]
            dx = self.x[1]
            ddx = self.x[2]
            dxdt = dx @ self.dt
            T2 = self.dt**2.0

            residual = z - (x + dxdt + 0.5 * ddx * T2)

            self.x[0] = x + dxdt + 0.5 * ddx * T2 + G * residual
            self.x[1] = dx + ddx * self.dt + (H / self.dt) * residual
            self.x[2] = ddx + (2 * K / (self.dt**2)) * residual
