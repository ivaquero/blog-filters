from __future__ import annotations

import math
import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from numpy import linalg
from scipy import stats

from .helpers import pretty_str
from .transformers import reshape_z

if TYPE_CHECKING:
    from collections.abc import Callable


class ExtendedKalmanFilter:
    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0):
        """Initialize an object of Extended Kalman Filter

        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter.
            This is used to set the default size of P, Q, and u
        dim_z : int
            Number of of measurement inputs.

        dim_u : int (optional)
            Size of the control input, if it is being used.
            Default value 0 indicates it is not used.

        compute_log_likelihood : bool (default = True)
            Computes log likelihood by default, but this can be a slow
            computation, turn this computation off if you never use it.

        Attributes
        ----------
        x : numpy.array(dim_x, 1)
            Current state estimate.
        P : numpy.array(dim_x, dim_x)
            Current state covariance matrix.
        F : numpy.array()
            State Transition matrix.
        Q : numpy.array(dim_x, dim_x)
            Process noise covariance matrix.
        z : numpy.array
            Last measurement used in update(). Read only.
        H : numpy.array(dim_z, dim_x)
            Measurement function.
        R : numpy.array(dim_z, dim_z)
            Measurement noise covariance matrix.
        y : numpy.array
            Residual of the update step. Read only.
        K : numpy.array(dim_x, dim_z)
            Kalman gain of the update step. Read only.
        S :  numpy.array
            System uncertainty. Read only.
        SI :  numpy.array
            Inverse system uncertainty. Read only.
        log_likelihood : float
            log-likelihood of the last measurement. Read only.
        likelihood : float
            likelihood of last measurement. Read only.
        mahalanobis : float
            mahalanobis distance of the innovation. Read only.
        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # state
        self.x = np.zeros((dim_x, 1))
        # state covariance
        self.P = np.eye(dim_x)
        # transition matrix
        self.F = np.eye(dim_x)
        # transition covariance
        self.Q = np.eye(dim_x)

        # measurement
        self.z = np.array([[None] * self.dim_z]).T
        self.H = np.zeros((dim_z, dim_x))
        # measurement covariance
        self.R = np.eye(dim_z)

        # control transition matrix, used to apply control input to the state, initialized as None
        self.G = None
        # residual vector, difference between the measurement and the prediction, initialized to zero
        self.y = np.zeros((dim_z, 1))

        # measurement vector, reshaped to be consistent with the dimensionality of the state vector
        z = np.array([None] * self.dim_z)
        self.z = reshape_z(z, self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step
        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros((dim_z, 1))
        # system uncertainty
        self.S = np.zeros((dim_z, dim_z))
        # inverse system uncertainty
        self.SI = np.zeros((dim_z, dim_z))
        # identity matrix.
        self._I = np.eye(dim_x)

        # only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # save the priors so that in case to inspect them for various purposes
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # save the posts so that in case to inspect them for various purposes
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_x(self, u: np.ndarray | float = 0.0):
        """Predicts the next state of X. You would need to do this when the usual Taylor expansion to generate F is not providing accurate results.

        Parameters
        ----------
        u : np.ndarray | float, optional, by default 0.0
        """

        # x = Fx + Gu
        if self.G is not None and u is not None:
            self.x = self.F @ self.x + self.G @ u
        else:
            self.x = self.F @ self.x

    def predict(self, u: np.ndarray | float = 0):
        """Predict next state (prior) using the Kalman filter state propagation equations.

        Parameters
        ----------
        u : np.ndarray | float, optional, by default 0
        """

        self.predict_x(u)
        self.P = self.F @ self.P @ self.F.T + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(
        self,
        z: np.ndarray,
        HJac: Callable[[np.ndarray], np.ndarray],
        Hx: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray | float | None = None,
        args: tuple = (),
        hx_args: tuple = (),
        residual=np.subtract,
    ):
        """Performs the update innovation of the EKF.

        Parameters
        ----------
        z : np.ndarray
           measurement for this step.
        HJac : function
           function which computes the Jacobian of the H matrix. Takes state variable (self.x) as input, returns H.
        Hx : function
           function which takes as input the state variable (self.x) along with the optional arguments in hx_args, and returns the measurement that would correspond to that state.
        R : np.ndarray | float | None
        args : tuple, optional, default (,)
           arguments to be passed into HJac after the required state variable.
        hx_args : tuple, optional, default (,)
           arguments to be passed into Hx function after the required state variable.
        residual : function (z, z2), optional
           function that computes the residual (difference) between the two measurement vectors.
        """

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        H = HJac(self.x, *args)
        hx = Hx(self.x, *hx_args)

        # update step
        Pxz = self.P @ H.T
        self.S = H @ Pxz + R
        self.SI = linalg.inv(self.S)
        self.K = Pxz @ self.SI

        self.y = residual(z, hx)
        self.x += self.K @ self.y

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update2(
        self,
        z: np.ndarray,
        Hess: Callable[[np.ndarray], np.ndarray],
        Hx: Callable[[np.ndarray], np.ndarray],
        Hxx: Callable[[np.ndarray], np.ndarray],
        HxR: Callable[[np.ndarray], np.ndarray] | None = None,
        R: np.ndarray | float | None = None,
        args: tuple = (),
        hx_args: tuple = (),
        hxx_args: tuple = (),
        hxr_args: tuple = (),
        residual=np.subtract,
    ):
        """Performs the update innovation of the 2nd order EKF.

        Parameters
        ----------
        z : np.ndarray
            measurement for this step.
        Hess : function
            function which computes the Hessian of the H matrix. Takes state variable (self.x) as input, returns H.
        Hx : function
            function which takes as input the state variable (self.x) along with the optional arguments in hx_args, and returns the measurement that would correspond to that state.
        Hxx : function
            derivative of H matrix with respect to state as matrix.
        HxR : function
            derivative of H matrix with respect to noise as matrix, default identity.
        args : tuple, optional, default (,)
            arguments to be passed into Hess after the required state
            variable.
        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state variable.
        hxx_args : tuple, optional, default (,)
            arguments to be passed into Hxx function after the required state variable.
        hxr_args : tuple, optional, default (,)
            arguments to be passed into HxR function after the required state variable.
        residual : function (z, z2), optional
            function that computes the residual between the two measurement vectors. You will normally want to use the built in unless your residual computation is nonlinear (for example, angles)
        """
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)
        if not isinstance(hxx_args, tuple):
            hxx_args = (hxx_args,)
        if not isinstance(hxr_args, tuple):
            hxr_args = (hxr_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        hxr = np.eye(self.dim_z) * R if HxR is None else HxR(self.x, *hxr_args)

        He = Hess(self.x, *args)
        hx = Hx(self.x, *hx_args)
        hxx = Hxx(self.x, *hxx_args)

        # update step
        self.y = residual(z, hx)
        for i in range(He.shape[0]):
            self.y[i] -= 0.5 * np.trace(He[i] @ self.P)

        self.S = hxr @ R @ hxr.T + hxx @ self.P @ hxx.T
        for i in range(He.shape[0]):
            for j in range(He.shape[0]):
                self.S[i, j] += 0.5 * np.trace(He[i] @ self.P @ He[j] @ self.P)

        self.SI = linalg.inv(self.S)
        self.K = self.P @ hxx.T @ self.SI
        self.x += self.K @ self.y

        I_KH = self._I - self.K @ hxx
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_update(
        self,
        z: np.ndarray,
        HJac: Callable[[np.ndarray], np.ndarray],
        Hx: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray | float | None = None,
        args: tuple = (),
        hx_args: tuple = (),
        u: int = 0,
    ):
        """Performs the predict/update innovation of the EKF.

        Parameters
        ----------
        z : np.ndarray
        HJac : Callable[[np.ndarray], np.ndarray]
        Hx : Callable[[np.ndarray], np.ndarray]
        R : np.ndarray | float | None
        args : tuple
        hx_args : tuple
        u : int, optional, by default 0
        """

        # predict step
        self.predict(u)
        # update step
        self.update(z, HJac, Hx, R, args=args, hx_args=hx_args)

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                x=self.y, cov=self.S, allow_singular=True
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """Computed from the log-likelihood. The log-likelihood can be very small, so by default we always return a number >= sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """Mahalanobis distance of innovation.
        e.g. 3 means measurement was 3 standard deviations away from the predicted value.
        """

        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(self.y.T @ self.SI @ self.y))
        return self._mahalanobis

    def __repr__(self):
        return "\n".join(
            [
                "KalmanFilter object",
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("likelihood", self.likelihood),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
            ]
        )
