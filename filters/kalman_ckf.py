import math
import sys
from copy import deepcopy

import numpy as np
from numpy import linalg
from scipy import stats

from .helpers import pretty_str
from .sigma_points import SphericalRadialSigmas


def ckf_transform(Xs, Q):
    """_summary_

    Parameters
    ----------
    Xs : _type_
        _description_
    Q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    m, n = Xs.shape

    x = sum(Xs, 0)[:, None] / m
    P = np.zeros((n, n))
    xf = x.flatten()
    for k in range(m):
        P += np.outer(Xs[k], Xs[k]) - np.outer(xf, xf)

    P *= 1 / m
    P += Q

    return x, P


class CubatureKalmanFilter:
    """Implements the Cubuture Kalman filter (CKF).

    References
    ----------
    Arasaratnam, I, Haykin, S. "Cubature Kalman Filters," IEEE Transactions on Automatic Control, 2009, pp 1254-1269, vol 54, No 6
    """

    def __init__(
        self,
        dim_x,
        dim_z,
        dt,
        hx,
        fx,
        x_mean_fn=None,
        z_mean_fn=None,
        residual_x=None,
        residual_z=None,
    ):
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.K = 0
        self.dim_x = dim_x
        self.dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2 * dim_x
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.y = np.zeros((dim_z, 1))
        self.z = np.array([[None] * self.dim_z]).T
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self.residual_x = np.subtract if residual_x is None else residual_x
        self.residual_z = np.subtract if residual_z is None else residual_z
        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = np.zeros((2 * self.dim_x, self.dim_x))
        self.sigmas_h = np.zeros((2 * self.dim_x, self.dim_z))

        # Only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, dt=None, fx_args=()):
        """Performs the predict step of the CKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P).

        Important: this MUST be called before update() is called for the first time.
        """

        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        sigmas = SphericalRadialSigmas(self.dim_x, self.P).sigma_points(self.x, self.P)

        # evaluate cubature points
        for k in range(self._num_sigmas):
            self.sigmas_f[k] = self.fx(sigmas[k], dt, *fx_args)

        self.x, self.P = ckf_transform(self.sigmas_f, self.Q)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, hx_args=()):
        """Update the CKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.
        """

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        for k in range(self._num_sigmas):
            self.sigmas_h[k] = self.hx(self.sigmas_f[k], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = ckf_transform(self.sigmas_h, R)
        self.SI = linalg.inv(self.S)

        # compute cross variance of the state and the measurements
        m = self._num_sigmas  # literaure uses m for scaling factor
        xf = self.x.flatten()
        zpf = zp.flatten()

        Pxz = (
            np.sum(
                np.einsum("ij,ik->ijk", self.sigmas_f - xf, self.sigmas_h - zpf), axis=0
            )
            / m
        )

        self.K = Pxz @ self.SI  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        self.x += self.K @ self.y
        self.P -= self.K @ self.S @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    @property
    def log_likelihood(self):
        """
        Computed the log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                x=self.y, cov=self.S
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small, so by default we always return a number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = math.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """ "
        Mahalanobis distance of innovation.
        e.g. 3 means measurement was 3 standard deviations away from the predicted value.
        """
        if self._mahalanobis is None:
            self._mahalanobis = math.sqrt(float(self.y.T @ self.SI @ self.y))
        return self._mahalanobis

    def __repr__(self):
        return "\n".join(
            [
                "CubatureKalmanFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dt", self._dt),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
            ]
        )
