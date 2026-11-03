import math
import sys
from copy import deepcopy

import numpy as np
from numpy import linalg
from scipy import stats

from ._typing import Batch2Output, UkfRtsOutput
from .helpers import pretty_str


def _as_covariance(value, dim):
    return np.eye(dim) * value if np.isscalar(value) else value


def unscented_transform(sigmas, Wm, Wc, noise_cov=None, mean_fn=None, residual_fn=None):
    """Computes unscented transform of a set of sigma points and weights.

    Parameters
    ----------
    sigmas : ndarray, of size (n, 2n+1)
        2D array of sigma points.
    Wm : ndarray
        Weights for the mean.
    Wc : ndarray
        Weights for the covariance.
    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.
    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this when your state variable contains nonlinear values such as angles which cannot be summed.
    residual_fn : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        Supply this when your state variable cannot support subtraction, such as angles. x and y are state vectors, not scalars.

    Returns
    -------
    x : ndarray
        Mean of the sigma points after passing through the transform.
    P : ndarray
        covariance of the sigma points after passing through the transform.
    """

    kmax, n = sigmas.shape
    x = Wm @ sigmas if mean_fn is None else mean_fn(sigmas, Wm)

    # new covariance is the sum of the outer product of the residuals
    # times the weights
    # the fast way: for linear cases
    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x
        P = (y.T * Wc) @ y
    else:
        # the slow way: for nonlinear cases, like angles
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)


class UnscentedKalmanFilter:
    def __init__(
        self,
        dim_x,
        dim_z,
        dt,
        hx,
        fx,
        points,
        sqrt_fn=None,
        x_mean_fn=None,
        z_mean_fn=None,
        residual_x=None,
        residual_z=None,
        state_add=None,
    ):
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.msqrt = linalg.cholesky if sqrt_fn is None else sqrt_fn
        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        self.residual_x = np.subtract if residual_x is None else residual_x
        self.residual_z = np.subtract if residual_z is None else residual_z
        self.state_add = np.add if state_add is None else state_add
        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros(dim_z)
        self.z = np.array([[None] * dim_z]).T
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)

        # pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(
            self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x
        )

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        if z is None:
            self.z = np.array([[None] * self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        R = self.R if R is None else _as_covariance(R, self._dim_z)

        for i, sigma in enumerate(self.sigmas_f):
            self.sigmas_h[i] = hx(sigma, **hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(
            self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z
        )
        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = linalg.solve(self.S.T, Pxz.T).T  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        # update Gaussian state estimate (x, P)
        self.x = self.state_add(self.x, self.K @ self.y)
        self.P -= self.K @ self.S @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """
        if self.residual_x is np.subtract and self.residual_z is np.subtract:
            dx = sigmas_f - x
            dz = sigmas_h - z
            return (dx.T * self.Wc) @ dz

        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        for i in range(sigmas_f.shape[0]):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """Computes the values of sigmas_f. It is useful when you need to call update more than once between calls to predict, so the sigmas correctly reflect the updated state x, P."""

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, sigma in enumerate(sigmas):
            self.sigmas_f[i] = fx(sigma, dt, **fx_args)

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        try:
            z = zs[0]
        except TypeError as e:
            error_message = "zs must be list-like"
            raise TypeError(error_message) from e

        if self._dim_z == 1:
            if not np.isscalar(z) and (z.ndim != 1 or len(z) != 1):
                error_message = "zs must be a list of scalars or 1D, 1 element arrays"
                raise TypeError(error_message)
        elif len(z) != self._dim_z:
            error_message = (
                f"each element in zs must be a 1D array of length {self._dim_z}"
            )
            raise TypeError(error_message)

        z_n = len(zs)

        Rs = [self.R] * z_n if Rs is None else Rs
        dts = [self._dt] * z_n if dts is None else dts

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = np.zeros((z_n, self._dim_x))
        else:
            means = np.zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = np.zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts, strict=True)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return Batch2Output(means, covariances)

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        """Runs the Rauch-Tung-Striebel Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.
        """

        if len(Xs) != len(Ps):
            error_message = "Xs and Ps must have the same length"
            raise ValueError(error_message)

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif np.isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform

        # smoother gain
        Ks = np.zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = np.zeros((num_sigmas, dim_x))

        for k in reversed(range(n - 1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i, sigma in enumerate(sigmas):
                sigmas_f[i] = self.fx(sigma, dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x
            )

            # compute cross variance
            if self.residual_x is np.subtract:
                y = sigmas_f - xb
                z = sigmas - Xs[k]
                Pxb = (z.T * self.Wc) @ y
            else:
                Pxb = np.zeros((dim_x, dim_x))
                for i in range(num_sigmas):
                    y = self.residual_x(sigmas_f[i], xb)
                    z = self.residual_x(sigmas[i], Xs[k])
                    Pxb += self.Wc[i] * np.outer(z, y)

            # compute gain
            K = linalg.solve(Pb.T, Pxb.T).T

            # update the smoothed estimates
            xs[k] += K @ self.residual_x(xs[k + 1], xb)
            ps[k] += K @ (ps[k + 1] - Pb) @ K.T
            Ks[k] = K

        return UkfRtsOutput(xs, ps, Ks)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                self.y, cov=self.S, allow_singular=True
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """Computed from the log-likelihood. The log-likelihood can be very small, so by default we always return a number >= sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = math.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(self.y.T @ linalg.solve(self.S, self.y)))
        return self._mahalanobis

    def __repr__(self):
        return "\n".join(
            [
                "UnscentedKalmanFilter object",
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("S", self.S),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
                pretty_str("sigmas_f", self.sigmas_f),
                pretty_str("h", self.sigmas_h),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("residual_x", self.residual_x),
                pretty_str("residual_z", self.residual_z),
                pretty_str("msqrt", self.msqrt),
                pretty_str("hx", self.hx),
                pretty_str("fx", self.fx),
                pretty_str("x_mean", self.x_mean),
                pretty_str("z_mean", self.z_mean),
            ]
        )
