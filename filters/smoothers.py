import numpy as np
from numpy import linalg

from ._typing import RtsOutput
from .helpers import pretty_str


def _as_sequence(value, n: int, default):
    """Broadcast one matrix to ``n`` steps, or pass a per-step stack through."""

    if value is None:
        return [default] * n
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return [value] * n
    return list(value)


def rts_smoother(Xs, Ps, Fs=None, Qs=None, Q=None, inv=linalg.inv):
    """Rauch-Tung-Striebel smoother over a ``batch_filter`` run.

    Parameters
    ----------
    Xs : ndarray, shape (n, dim_x) or (n, dim_x, 1)
        State means from the forward pass. Both layouts are accepted; the
        result keeps whichever one was supplied.
    Ps : ndarray, shape (n, dim_x, dim_x)
        Covariances from the forward pass.
    Fs : ndarray | sequence | None
        A single ``(dim_x, dim_x)`` transition matrix or a per-step stack.
        Defaults to the identity.
    Qs : ndarray | sequence | None
        A single process-noise covariance or a per-step stack. Defaults to
        zeros.
    Q : ndarray | None
        Alias for ``Qs``, kept so the historical ``rts_smoother(..., Q=kf.Q)``
        call sites keep working.
    inv : callable
        Matrix inverse; pass ``scipy.linalg.pinv`` for degenerate covariances.
    """

    Xs = np.asarray(Xs, dtype=float)
    Ps = np.asarray(Ps, dtype=float)

    if Xs.shape[0] != Ps.shape[0]:
        error_message = "length of Xs and Ps must be the same"
        raise ValueError(error_message)

    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    Fs = _as_sequence(Fs, n, np.eye(dim_x))
    Qs = _as_sequence(Qs if Qs is not None else Q, n, np.zeros((dim_x, dim_x)))

    # smoother gain
    K = np.zeros((n, dim_x, dim_x))
    x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        Pp[k] = F @ P[k] @ F.T + Qs[k + 1]
        K[k] = P[k] @ F.T @ inv(Pp[k])
        x[k] += K[k] @ (x[k + 1] - F @ x[k])
        P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T

    return RtsOutput(x, P, K, Pp)


class FixedLagSmoother:
    """Fixed Lag Kalman smoother."""

    def __init__(self, dim_x, dim_z, N=None):
        """Create a fixed lag Kalman filter smoother."""

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = N

        self.x = np.zeros((dim_x, 1))
        self.x_s = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.eye(dim_z, dim_x)
        self.R = np.eye(dim_z)
        self.K = np.zeros((dim_x, 1))
        self.y = np.zeros((dim_z, 1))
        self.G = 0.0
        self.S = np.zeros((dim_z, dim_z))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self.count = 0

        if N is not None:
            self.xSmooth = []

    def smooth(self, z, u=None):
        """Smooths the measurement using a fixed lag smoother.

        On return, self.xSmooth is populated with the N previous smoothed estimates, where self.xSmooth[k] is the kth time step. self.x merely contains the current Kalman filter output of the most recent measurement, and is not smoothed at all (beyond the normal Kalman filter processing).

        self.xSmooth grows in length on each call. If you run this 1 million times. If you want to filter something else, create a new FixedLagSmoother object.
        """

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        G = self.G
        N = self.N
        assert N is not None, "smooth() requires N to be set at construction time"

        k = self.count

        # predict step of normal Kalman filter
        x_pre = F @ x
        if u is not None:
            x_pre += G @ u

        P = F @ P @ F.T + Q

        # update step of normal Kalman filter
        self.y = z - H @ x_pre

        self.S = H @ P @ H.T + R
        SI = linalg.inv(self.S)

        K = P @ H.T @ SI
        x = x_pre + K @ self.y
        I_KH = self._I - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T

        self.xSmooth.append(x_pre.copy())

        # compute invariants
        HTSI = H.T @ SI
        F_LH = (F - K @ H).T

        if k >= N:
            PS = P.copy()  # smoothed P for step i
            for i in range(N):
                K = PS @ HTSI  # smoothed gain
                PS = PS @ F_LH  # smoothed covariance

                si = k - i
                self.xSmooth[si] = self.xSmooth[si] + K @ self.y
        else:
            # Some sources specify starting the fix lag smoother only
            # after N steps have passed, some don't. I am getting far
            # better results by starting only at step N.
            self.xSmooth[k] = x.copy()

        self.count += 1
        self.x = x
        self.P = P

    def smooth_batch(self, zs, N, us=None):
        """Batch smooths the set of measurements using a fixed lag smoother.

        This is a batch processor, so it does not alter any of the object's data. In particular, self.x is NOT modified. All date is returned by the function.
        """

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        G = self.G

        if x.ndim == 1:
            xSmooth = np.zeros((len(zs), self.dim_x))
            xhat = np.zeros((len(zs), self.dim_x))
        else:
            xSmooth = np.zeros((len(zs), self.dim_x, 1))
            xhat = np.zeros((len(zs), self.dim_x, 1))
        for k, z in enumerate(zs):
            # predict step of normal Kalman filter
            x_pre = F @ x
            if us is not None:
                x_pre += G @ us[k]

            P = F @ P @ F.T + Q
            # update step of normal Kalman filter
            y = z - H @ x_pre
            S = H @ P @ H.T + R
            SI = linalg.inv(S)
            K = P @ H.T @ SI

            x = x_pre + K @ y
            I_KH = self._I - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T

            xhat[k] = x.copy()
            xSmooth[k] = x_pre.copy()

            # compute invariants
            HTSI = H.T @ SI
            F_LH = (F - K @ H).T

            if k >= N:
                PS = P.copy()  # smoothed P for step i
                for i in range(N):
                    K = PS @ HTSI  # smoothed gain
                    PS = PS @ F_LH  # smoothed covariance

                    si = k - i
                    xSmooth[si] = xSmooth[si] + K @ y
            else:
                # Some sources specify starting the fix lag smoother only
                # after N steps have passed, some don't. I am getting far
                # better results by starting only at step N.
                xSmooth[k] = xhat[k]

        return xSmooth, xhat

    def __repr__(self):
        return "\n".join(
            [
                "FixedLagSmoother object",
                pretty_str("dim_x", self.x),
                pretty_str("dim_z", self.x),
                pretty_str("N", self.N),
                pretty_str("x", self.x),
                pretty_str("x_s", self.x_s),
                pretty_str("P", self.P),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("H", self.H),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("B", self.G),
            ]
        )
