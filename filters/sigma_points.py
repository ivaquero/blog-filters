import numpy as np
from scipy import linalg

from .helpers import pretty_str


def print_sigmas(n=1, mean=5, cov=3, kappa=2, alpha=0.1, beta=2.0):
    """Print the Merwe scaled sigma points, weights and scaling constant.

    Parameters
    ----------
    n : int, optional
        State dimension, by default 1.
    mean : int, optional
        State mean, by default 5.
    cov : int, optional
        State covariance (scalar expands to ``cov * I``), by default 3.
    kappa : int, optional
        Secondary scaling parameter, by default 2.
    alpha : float, optional
        Primary scaling parameter, by default 0.1.
    beta : float, optional
        Prior-knowledge parameter (2 for Gaussian), by default 2.0.
    """

    points = MerweScaledSigmas(n, kappa, alpha, beta)
    print(f"sigmas: {points.sigma_points(mean, cov).T[0]}")
    Wm, Wc = points.Wm, points.Wc
    print(f"mean weights: {Wm}")
    print(f"cov weights: {Wc}")
    print(f"lambda: {alpha**2 * (n + kappa) - n}")
    print(f"sum cov: {sum(Wc)}")


class JulierSigmas:
    """
    Generates sigma points and weights according to Simon J. Julier
    and Jeffery K. Uhlmann's original paper.
    It parametizes the sigma points using kappa.
    """

    def __init__(self, n, kappa=0.0, sqrt_method=None, subtract=None):
        self.n = n
        self.kappa = kappa
        self.sqrt = linalg.cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self.sigma_weights()

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        """Computes the sigma points for an UKF given the mean (x) and covariance (P) of the filter.
        kappa is an arbitrary constant. Returns sigma points.
        """

        if self.n != np.size(x):
            error_message = f"expected size(x) {self.n}, but size is {np.size(x)}"
            raise ValueError(error_message)

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)
        sigmas = np.zeros((2 * n + 1, n))

        # U'*U = (n+kappa)*P.
        # Returns lower triangular matrix.
        U = self.sqrt((n + self.kappa) * P)

        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])
        return sigmas

    def sigma_weights(self):
        """Computes the weights for the UKF. In this formulation the weights for the mean and covariance are the same."""

        n = self.n
        k = self.kappa
        c = n + k

        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wm[0] = k / c
        self.Wc = self.Wm

    def __repr__(self):
        return "\n".join(
            [
                "JulierSigmas object",
                pretty_str("n", self.n),
                pretty_str("kappa", self.kappa),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("subtract", self.subtract),
                pretty_str("sqrt", self.sqrt),
            ]
        )


class MerweScaledSigmas:
    """
    Generates sigma points and weights according to Van der Merwe's 2004 dissertation. It parametizes the sigma points using kappa, alpha, beta terms, and is the version seen in most publications.
    Unless you know better, this should be your default choice.
    """

    def __init__(
        self, n, kappa, alpha=1, beta=0, sqrt_method=None, subtract_method=None
    ):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - n
        self.sqrt = linalg.cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract_method is None else subtract_method
        self.sigma_weights()

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        """Computes the sigma points for an UKF given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        """

        if self.n != np.size(x):
            error_message = f"expected size(x) {self.n}, but size is {np.size(x)}"
            raise ValueError(error_message)

        if np.isscalar(x):
            x = np.asarray([x])

        n = self.n
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        c = self.alpha**2 * (n + self.kappa)  # scaling constant
        U = self.sqrt(c * P)  # upper-triangular matrix
        # lam = c - n

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])

        # equivalent to
        # U = self.sqrt(P)
        # sigmas = np.hstack([np.zeros(M.shape), U, -U])
        # sigmas = np.sqrt(c) * sigmas + np.tile(M, sigmas.shape[1])

        return sigmas

    def sigma_weights(self):
        """Computes the weights for the scaled UKF."""

        n = self.n
        c = self.alpha**2 * (n + self.kappa)  # scaling constant
        lam = c - n

        self.Wc = np.full(2 * n + 1, 0.5 / c)
        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wc[0] = lam / c + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lam / c

    def __repr__(self):
        return "\n".join(
            [
                "MerweScaledSigmas object",
                pretty_str("n", self.n),
                pretty_str("alpha", self.alpha),
                pretty_str("beta", self.beta),
                pretty_str("kappa", self.kappa),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("subtract", self.subtract),
                pretty_str("sqrt", self.sqrt),
            ]
        )


class SimplexSigmas:
    """
    Generates sigma points and weights according to the simplex
    method.
    """

    def __init__(self, n, alpha=1, sqrt_method=None, subtract=None):
        self.n = n
        self.alpha = alpha
        self.sqrt = linalg.cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self.sigma_weights()

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""
        return self.n + 1

    def sigma_points(self, x, P):
        """
        Computes the implex sigma points for an UKF given the mean (x) and covariance (P) of the filter.
        Returns tuple of the sigma points and weights.
        """

        if self.n != np.size(x):
            error_message = f"expected size(x) {self.n}, but size is {np.size(x)}"
            raise ValueError(error_message)

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)

        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)
        U = self.sqrt(P)

        lam = n / (n + 1)
        Istar = np.array([[-1 / np.sqrt(2 * lam), 1 / np.sqrt(2 * lam)]])

        for d in range(2, n + 1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1.0 / np.sqrt(lam * d * (d + 1))
            row[0, -1] = -d / np.sqrt(lam * d * (d + 1))

            Istar = np.hstack((Istar, np.zeros((Istar.shape[0], 1))))
            Istar = np.vstack((Istar, row))

        I_ = np.sqrt(n) * Istar
        scaled_unitary = U.T @ I_

        sigmas = self.subtract(x, -scaled_unitary)
        return sigmas.T

    def sigma_weights(self):
        """Computes the weights for the scaled UKF."""

        n = self.n
        c = 1.0 / (n + 1)
        self.Wm = np.full(n + 1, c)
        self.Wc = self.Wm

    def __repr__(self):
        return "\n".join(
            [
                "SimplexSigmas object",
                pretty_str("n", self.n),
                pretty_str("alpha", self.alpha),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("subtract", self.subtract),
                pretty_str("sqrt", self.sqrt),
            ]
        )


class SphericalRadialSigmas:
    """Generates cubature points and weights by the spherical-radial rule.

    The rule uses ``2n`` points with equal weights ``1 / (2n)``. The previous
    implementation stored a ``P`` argument it never read and returned weights
    of length ``4n`` from ``sigma_weights``.
    """

    def __init__(self, n):
        self.n = n
        self.sigma_weights()

    def num_sigmas(self):
        """Number of cubature points for each variable in the state x"""

        return 2 * self.n

    def sigma_points(self, x, P):
        """Creates cubature points for the the specified state and covariance.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Mean state.
        P : ndarray, shape (n, n)
            State covariance.

        Returns
        -------
        ndarray, shape (2n, n)
            Cubature points.

        References
        ----------
        Arasaratnam, I, Haykin, S. "Cubature Kalman Filters," IEEE Transactions on Automatic Control, 2009, pp 1254-1269, vol 54, No 6
        """
        if self.n != np.size(x):
            error_message = f"expected size(x) {self.n}, but size is {np.size(x)}"
            raise ValueError(error_message)

        n = self.n
        x = np.asarray(x).flatten()

        # evaluation points (nx2n)
        sigmas = np.empty((2 * n, n))
        U = linalg.cholesky(P) * np.sqrt(n)

        for k in range(n):
            sigmas[k] = x + U[k]
            sigmas[n + k] = x - U[k]

        return sigmas

    def sigma_weights(self):
        """Computes the equal weights of the cubature rule."""

        num = self.num_sigmas()
        self.Wm = np.full(num, 1.0 / num)
        self.Wc = self.Wm
