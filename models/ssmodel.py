"""The state-space model contract shared by every estimator in the toolkit.

A model answers two questions and, optionally, three follow-ups:

============================  ====================================================
member                        meaning
============================  ====================================================
``fx(x, dt, u, w)``           one-step state propagation ``x[t+1] = f(x, u, w)``
``hx(x, v)``                  measurement map ``z = h(x, v)``
``jacobian_x(x)``             ``df/dx`` — numeric fallback provided
``jacobian_h(x)``             ``dh/dx`` — numeric fallback provided
``process_cov()``             ``Q`` priced from the model's own noise input
============================  ====================================================

``fx``/``hx`` deliberately mirror the callables that
:class:`filters.kalman_ukf.UnscentedKalmanFilter` and
:class:`filters.kalman_ekf.ExtendedKalmanFilter` already accept, so a model can
be handed to either filter without an adapter:

>>> from filters.kalman_ukf import UnscentedKalmanFilter
>>> kf = UnscentedKalmanFilter.from_model(
...     ConstantVelocity(dt=0.1), points
... )

The legacy ``state_equation(t, x, u, w)`` / ``observation_equation(t, x, v)``
spellings remain available as thin aliases. Noise arguments default to ``None``,
meaning "noise-free": passing a bare ``0`` used to work only for models that
special-cased it, and blew up with a matmul shape error everywhere else.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from shared import Matrix, Measurement, Vector

from ._common import as_state_batch, numerical_jacobian

__all__ = ["NonlinearStateSpaceModel", "StateSpaceModel"]


class StateSpaceModel(ABC):
    """Abstract base for discrete-time state-space models.

    Subclasses must define :attr:`dim_x`, :attr:`dim_z`, :meth:`fx` and
    :meth:`hx`. Jacobians fall back to central differences, so a nonlinear model
    becomes EKF-ready as soon as its equations are written down; override
    :meth:`jacobian_x` / :meth:`jacobian_h` when the analytic form is cheap.
    """

    @property
    @abstractmethod
    def dim_x(self) -> int:
        """Number of state variables."""

    @property
    @abstractmethod
    def dim_z(self) -> int:
        """Number of measurement variables."""

    @property
    def dim_u(self) -> int:
        """Number of control inputs; zero when the model is uncontrolled."""

        return 0

    @property
    def dim_w(self) -> int:
        """Number of process-noise inputs; defaults to one per state."""

        return self.dim_x

    @property
    def dim_v(self) -> int:
        """Number of measurement-noise inputs; defaults to one per measurement."""

        return self.dim_z

    @property
    def ndim(self) -> dict[str, int]:
        """Dimension lookup keyed by ``x``/``z``/``u``/``w``/``v``."""

        return {
            "x": self.dim_x,
            "z": self.dim_z,
            "u": self.dim_u,
            "w": self.dim_w,
            "v": self.dim_v,
        }

    @abstractmethod
    def fx(
        self,
        x: Any,
        dt: float | None = None,
        *,
        t: float = 0.0,
        u: Any = None,
        w: Any = None,
    ) -> Vector:
        """Propagate one state vector over ``dt``.

        Parameters
        ----------
        x : array_like
            State vector of length :attr:`dim_x`.
        dt : float, optional
            Step length; ``None`` means "use the model's own ``dt``".
        t : float, optional
            Absolute time, for explicitly time-varying models.
        u : array_like, optional
            Control input; ``None`` means zero.
        w : array_like, optional
            Process noise; ``None`` means noise-free.
        """

    @abstractmethod
    def hx(self, x: Any, v: Any = None) -> Measurement:
        """Map a state vector to the expected measurement.

        Parameters
        ----------
        x : array_like
            State vector of length :attr:`dim_x`.
        v : array_like, optional
            Measurement noise; ``None`` means noise-free.
        """

    def fx_many(
        self,
        x: Any,
        dt: float | None = None,
        *,
        t: float = 0.0,
        u: Any = None,
        w: Any = None,
    ) -> Matrix:
        """Propagate a ``(n, dim_x)`` batch of states; a single vector yields one row.

        The default is a loop over :meth:`fx`, which keeps every subclass
        correct without extra work. Linear models override it with a single
        matrix product; particle filters gain their speedup from that override.
        """

        batch = as_state_batch(x, self.dim_x)
        return np.vstack(
            [np.asarray(self.fx(row, dt, t=t, u=u, w=w)).reshape(-1) for row in batch]
        )

    def hx_many(self, x: Any, v: Any = None) -> Matrix:
        """Map a ``(n, dim_x)`` batch of states to ``(n, dim_z)`` measurements."""

        batch = as_state_batch(x, self.dim_x)
        return np.vstack([np.asarray(self.hx(row, v=v)).reshape(-1) for row in batch])

    def jacobian_x(self, x: Any) -> Matrix:
        """Return ``df/dx`` evaluated at ``x`` (numeric by default)."""

        return numerical_jacobian(lambda state: self.fx(state), x)

    def jacobian_h(self, x: Any) -> Matrix:
        """Return ``dh/dx`` evaluated at ``x`` (numeric by default)."""

        return numerical_jacobian(lambda state: self.hx(state), x)

    def process_cov(self) -> Matrix:
        """Return the process-noise covariance ``Q``.

        Models that carry a noise input matrix override this; the base raises so
        a filter never silently runs with an all-zero ``Q``.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not define process noise; "
            "override process_cov() or pass Q explicitly"
        )

    def measurement_cov(self) -> Matrix:
        """Return the measurement-noise covariance ``R``."""

        raise NotImplementedError(
            f"{type(self).__name__} does not define measurement noise; "
            "override measurement_cov() or pass R explicitly"
        )

    def state_equation(self, t: float, x: Any, u: Any = None, w: Any = None) -> Vector:
        """Alias of :meth:`fx` kept for the pre-0.2 call sites."""

        return self.fx(x, t=t, u=u, w=w)

    def observation_equation(self, t: float, x: Any, v: Any = None) -> Measurement:
        """Alias of :meth:`hx` kept for the pre-0.2 call sites."""

        return self.hx(x, v=v)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dim_x={self.dim_x}, dim_z={self.dim_z}, "
            f"dim_u={self.dim_u})"
        )


class NonlinearStateSpaceModel(StateSpaceModel):
    """Additive-noise nonlinear model with explicit Jacobian hooks.

    Provides the ``Jfx``/``Jfw``/``Jhx``/``Jhv``/``Lt``/``Mt`` surface that the
    pre-0.2 subclasses implemented, and forwards :meth:`jacobian_x` /
    :meth:`jacobian_h` to it so the modern filter entry points and the legacy
    hooks stay numerically identical.
    """

    def Jfx(self, t: float, x: Any) -> Matrix:
        """Return ``df/dx`` at time ``t``."""

        raise NotImplementedError

    def Jfw(self, t: float, x: Any) -> Matrix:
        """Return ``df/dw`` at time ``t``."""

        raise NotImplementedError

    def Jhx(self, t: float, x: Any) -> Matrix:
        """Return ``dh/dx`` at time ``t``."""

        raise NotImplementedError

    def Jhv(self, t: float, x: Any) -> Matrix:
        """Return ``dh/dv`` at time ``t``."""

        raise NotImplementedError

    def Lt(self, t: float) -> Matrix:
        """Return the additive process-noise input matrix ``L[t]``."""

        raise NotImplementedError

    def Mt(self, t: float) -> Matrix:
        """Return the additive measurement-noise matrix ``M[t]``."""

        raise NotImplementedError

    def jacobian_x(self, x: Any) -> Matrix:
        return self.Jfx(0.0, np.asarray(x, dtype=float).reshape(-1))

    def jacobian_h(self, x: Any) -> Matrix:
        return self.Jhx(0.0, np.asarray(x, dtype=float).reshape(-1))
