"""Discrete-time linear Kalman filter with fade-memory support.

`KalmanFilter` implements the classic predict/update pair, plus batched
processing and a Rauch-Tung-Striebel smoother. The class satisfies the
`filters._typing.Filter` protocol structurally — no inheritance required —
so `IMMEstimator`, `MMAE`, and the plotting helpers accept any conforming
object (including test doubles).

Return values of `batch_filter` and `rts_smoother` are NamedTuples that
unpack exactly like the historical 4-tuples, so
`means, covs, means_p, covs_p = kf.batch_filter(zs)` still works while
`result.means` now also type-checks.
"""

import math
import sys

import numpy as np
import numpy.typing as npt
from numpy import linalg
from scipy import stats

from ._typing import BatchOutput, Matrix, MeasurementLike, RtsOutput, Vector
from .helpers import pretty_str
from .transformers import reshape_z


def _default_sequence(value: object, n: int) -> list:
    return [value] * n


def _as_covariance(value: npt.NDArray[np.float64] | float, dim: int) -> Matrix:
    return np.eye(dim) * value if np.isscalar(value) else value


class KalmanFilter:
    """Linear Kalman filter with optional fading memory.

    Attributes
    ----------
    x : Vector, shape (dim_x, 1)
        Current state estimate.
    P : Matrix, shape (dim_x, dim_x)
        Current state covariance.
    F : Matrix, shape (dim_x, dim_x)
        State transition matrix.
    Q : Matrix, shape (dim_x, dim_x)
        Process noise covariance.
    H : Matrix, shape (dim_z, dim_x)
        Measurement function.
    R : Matrix, shape (dim_z, dim_z)
        Measurement noise covariance.
    G : Matrix | None, shape (dim_x, dim_u)
        Control transition matrix.
    K : Matrix, shape (dim_x, dim_z)
        Kalman gain from the most recent update.
    y : Vector, shape (dim_z, 1)
        Innovation (residual) from the most recent update.
    S : Matrix, shape (dim_z, dim_z)
        Innovation covariance.
    z : Vector
        Last measurement consumed by update().
    x_prior, P_prior, x_post, P_post : np.ndarray
        Snapshots taken after predict() / update() respectively.

    References
    ----------
    Dan Simon, "Optimal State Estimation", Wiley, 2006, ch. 4.
    """

    dim_x: int
    dim_z: int
    dim_u: int

    x: Vector
    P: Matrix
    F: Matrix
    Q: Matrix

    z: Vector
    H: Matrix
    R: Matrix

    G: Matrix | None
    M: Matrix
    _alpha_sq: float

    K: Matrix
    y: Vector
    S: Matrix
    _I: Matrix

    _log_likelihood: float | None
    _likelihood: float | None
    _mahalanobis: float | None

    x_prior: Vector
    P_prior: Matrix
    x_post: Vector
    P_post: Matrix

    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0, _alpha_sq: float = 1.0):
        """Configure a linear Kalman filter of the given dimensions."""

        if dim_x < 1:
            error_message = "dim_x ≥ 1"
            raise ValueError(error_message)
        if dim_z < 1:
            error_message = "dim_z ≥ 1"
            raise ValueError(error_message)
        if dim_u < 0:
            error_message = "dim_u ≥ 0"
            raise ValueError(error_message)

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # state
        self.x: Vector = np.zeros((dim_x, 1))
        self.P: Matrix = np.eye(dim_x)
        self.F: Matrix = np.eye(dim_x)
        self.Q: Matrix = np.eye(dim_x)

        # measurement
        self.z: Vector = np.array([[None] * self.dim_z]).T
        self.H: Matrix = np.zeros((dim_z, dim_x))
        self.R: Matrix = np.eye(dim_z)

        # control
        self.G: Matrix | None = None
        self.M: Matrix = np.zeros((dim_x, dim_z))
        self._alpha_sq = _alpha_sq

        # innovation bookkeeping
        self.K: Matrix = np.zeros((dim_x, dim_z))
        self.y: Vector = np.zeros((dim_z, 1))
        self.S: Matrix = np.zeros((dim_z, dim_z))
        self._I: Matrix = np.eye(dim_x)

        self._log_likelihood: float | None = None
        self._likelihood: float | None = None
        self._mahalanobis: float | None = None

        # prior/posterior snapshots
        self.x_prior: Vector = self.x.copy()
        self.P_prior: Matrix = self.P.copy()
        self.x_post: Vector = self.x.copy()
        self.P_post: Matrix = self.P.copy()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _reset_likelihood(self) -> None:
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def _save_prior(self) -> None:
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def _save_posterior(self) -> None:
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    # ------------------------------------------------------------------
    # core two-step update
    # ------------------------------------------------------------------

    def predict(
        self,
        F: Matrix | None = None,
        G: Matrix | None = None,
        u: Vector | None = None,
        Q: Matrix | float | None = None,
    ) -> None:
        """Time-update: propagate state and covariance one step.

        Parameters
        ----------
        F : Matrix | None
            Override transition matrix; defaults to ``self.F``.
        G : Matrix | None
            Override control matrix; used only when ``u`` is provided.
        u : Vector | None
            Control input vector.
        Q : Matrix | float | None
            Process noise covariance or scalar multiple of identity.
        """
        F = self.F if F is None else F
        if G is None and u is not None:
            G = self.G
        Q = self.Q if Q is None else _as_covariance(Q, self.dim_x)

        # x = Fx + Gu
        self.x = F @ self.x
        if G is not None and u is not None:
            self.x += G @ u

        # P = α F P F' + Q
        self.P = self._alpha_sq * (F @ self.P @ F.T) + Q

        self._save_prior()

    def update(
        self,
        z: MeasurementLike | None,
        H: Matrix | None = None,
        R: Matrix | float | None = None,
    ) -> None:
        """Measurement-update: incorporate a new observation ``z``.

        Passing ``None`` leaves the state untouched but still refreshes the
        posterior snapshot so downstream savers see the missing-measurement
        step.
        """
        self._reset_likelihood()

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.y = np.zeros((self.dim_z, 1))
            self._save_posterior()
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)
        H = self.H if H is None else H
        R = self.R if R is None else _as_covariance(R, self.dim_z)

        # y = z - Hx
        self.y = z - H @ self.x

        Pxz = self.P @ H.T
        # S = H P H' + R
        self.S = H @ Pxz + R
        # K = P H' S^{-1}, solved via transpose to avoid inverting S.
        self.K = linalg.solve(self.S.T, Pxz.T).T

        # x = x + K y
        self.x += self.K @ self.y

        # Joseph form: (I-KH) P (I-KH)' + K R K'
        I_KH = self._I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

        self.z = z.copy()
        self._save_posterior()

    # ------------------------------------------------------------------
    # batch / smoother
    # ------------------------------------------------------------------

    def batch_filter(
        self,
        zs,
        Fs=None,
        Qs=None,
        Hs=None,
        Gs=None,
        us=None,
        Rs=None,
        update_first: bool = False,
        saver=None,
    ) -> BatchOutput:
        """Run the filter over a full sequence of measurements.

        Returns a :class:`BatchOutput` NamedTuple whose four fields unpack
        exactly like the historical raw tuple.
        """
        n = np.size(zs, 0)
        Fs = _default_sequence(self.F, n) if Fs is None else Fs
        Qs = _default_sequence(self.Q, n) if Qs is None else Qs
        Hs = _default_sequence(self.H, n) if Hs is None else Hs
        Gs = _default_sequence(self.G, n) if Gs is None else Gs
        us = _default_sequence(None, n) if us is None else us
        Rs = _default_sequence(self.R, n) if Rs is None else Rs

        mean_shape = (n, self.dim_x) if self.x.ndim == 1 else (n, self.dim_x, 1)
        means = np.zeros(mean_shape)
        means_p = np.zeros(mean_shape)
        cov = np.zeros((n, self.dim_x, self.dim_x))
        cov_p = np.zeros((n, self.dim_x, self.dim_x))

        steps = zip(zs, Fs, Qs, Hs, Rs, Gs, us, strict=True)
        for i, (z, F, Q, H, R, G, u) in enumerate(steps):
            if update_first:
                self.update(z, H=H, R=R)
                means[i, :] = self.x
                cov[i, :, :] = self.P
                self.predict(u=u, G=G, F=F, Q=Q)
                means_p[i, :] = self.x
                cov_p[i, :, :] = self.P
            else:
                self.predict(u=u, G=G, F=F, Q=Q)
                means_p[i, :] = self.x
                cov_p[i, :, :] = self.P
                self.update(z, R=R, H=H)
                means[i, :] = self.x
                cov[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return BatchOutput(means, cov, means_p, cov_p)

    def rts_smoother(
        self,
        Xs: npt.NDArray[np.float64],
        Ps: npt.NDArray[np.float64],
        Fs=None,
        Qs=None,
        inv=linalg.inv,
    ) -> RtsOutput:
        """Rauch-Tung-Striebel smoother over a `batch_filter` run.

        Parameters
        ----------
        Xs, Ps
            Mean / covariance stacks from `batch_filter`.
        Fs, Qs
            Optional per-step transition/process-noise sequences.
        inv
            Matrix inverse callable; defaults to `numpy.linalg.inv`. Pass
            `scipy.linalg.pinv` for numerically degenerate covariances.
        """
        if len(Xs) != len(Ps):
            error_message = "length of Xs and Ps must be the same"
            raise ValueError(error_message)

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        Fs = _default_sequence(self.F, n) if Fs is None else Fs
        Qs = _default_sequence(self.Q, n) if Qs is None else Qs

        K = np.zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            F = Fs[k + 1]
            Pp[k] = F @ P[k] @ F.T + Qs[k + 1]
            PFt = P[k] @ F.T
            K[k] = (
                linalg.solve(Pp[k].T, PFt.T).T
                if inv is linalg.inv
                else PFt @ inv(Pp[k])
            )
            x[k] += K[k] @ (x[k + 1] - F @ x[k])
            P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T

        return RtsOutput(x, P, K, Pp)

    # ------------------------------------------------------------------
    # diagnostic helpers
    # ------------------------------------------------------------------

    def residual_of(self, z: MeasurementLike) -> Vector:
        """Innovation of ``z`` against the prior state; does not mutate."""

        z = reshape_z(z, self.dim_z, self.x.ndim)
        return z - self.H @ self.x_prior

    def measurement_of_state(self, x: Vector) -> Vector:
        """Expected measurement for state ``x`` (i.e. ``H @ x``)."""
        return self.H @ x

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood of the last innovation."""
        if self._log_likelihood is None:
            if not np.any(self.S):
                self._log_likelihood = math.log(sys.float_info.min)
                return self._log_likelihood
            self._log_likelihood = stats.multivariate_normal.logpdf(
                x=self.y, cov=self.S
            )
        return self._log_likelihood

    @property
    def likelihood(self) -> float:
        """Likelihood of the last innovation, floored at `sys.float_info.min`."""
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if not np.isscalar(self._likelihood):
                self._likelihood = self._likelihood[0]
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self) -> float:
        """Mahalanobis distance of the innovation."""
        if self._mahalanobis is None:
            if not np.any(self.S):
                self._mahalanobis = 0.0
                return self._mahalanobis
            try:
                distance = self.y.T @ linalg.solve(self.S, self.y)
            except linalg.LinAlgError:
                distance = self.y.T @ linalg.pinv(self.S) @ self.y
            self._mahalanobis = math.sqrt(float(np.squeeze(distance)))
        return self._mahalanobis

    @property
    def alpha(self) -> float:
        """Square-root of the internal fading-memory scalar."""
        return self._alpha_sq**0.5

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not np.isscalar(value) or value < 1:
            error_message = "alpha must be a float greater than 1"
            raise ValueError(error_message)

        self._alpha_sq = value**2

    def log_likelihood_of(self, z: MeasurementLike | None) -> float:
        """Log-likelihood of a hypothetical measurement ``z``.

        Should only be called after `update`; calling after `predict` yields
        a stale `S` and an incorrect result.
        """

        if z is None:
            return math.log(sys.float_info.min)
        return stats.multivariate_normal.logpdf(z, self.H @ self.x, self.S)

    def __repr__(self) -> str:
        return "\n".join(
            [
                "KalmanFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dim_u", self.dim_u),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("x_post", self.x_post),
                pretty_str("P_post", self.P_post),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("H", self.H),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("M", self.M),
                pretty_str("G", self.G),
                pretty_str("z", self.z),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
                pretty_str("alpha", self.alpha),
            ]
        )
