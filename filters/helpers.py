import contextlib
import inspect
import math
import sys
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import numpy as np


class KFSaver:
    def __init__(
        self,
        kf,
        ignore=(),
        *,
        save_current=False,
        skip_private=False,
        skip_callable=False,
    ):
        self._kf = kf
        self._DL = defaultdict(list)
        self._skip_private = skip_private
        self._skip_callable = skip_callable
        self._ignore = ignore
        self._len = 0

        properties = inspect.getmembers(type(kf), lambda o: isinstance(o, property))
        self.properties = [p for p in properties if p[0] not in ignore]

        if save_current:
            self.save()

    def __getattr__(self, name: str) -> Any:
        # Attributes such as `x`, `y`, `P`, `z` are populated dynamically by
        # `save()` / `to_array()` via `self.__dict__[key] = ...`. This hook exists
        # so that static type checkers can resolve those attribute accesses; it
        # still raises AttributeError at runtime for genuinely missing names.
        msg = f"{type(self).__name__!r} object has no attribute {name!r}"
        raise AttributeError(msg)

    def save(self):
        kf = self._kf

        # force all attributes to be computed.
        for prop in self.properties:
            self._DL[prop[0]].append(getattr(kf, prop[0]))

        v = deepcopy(kf.__dict__)

        if self._skip_private:
            for key in list(v.keys()):
                if key.startswith("_"):
                    del v[key]

        if self._skip_callable:
            for key in list(v.keys()):
                if callable(v[key]):
                    del v[key]

        for ig in self._ignore:
            if ig in v:
                del v[ig]

        for key in list(v.keys()):
            self._DL[key].append(v[key])

        self.__dict__.update(self._DL)
        self._len += 1

    def __getitem__(self, key):
        return self._DL[key]

    def __setitem__(self, key, newvalue):
        self._DL[key] = newvalue
        self.__dict__.update(self._DL)

    def __len__(self):
        return self._len

    @property
    def keys(self):
        """list of all keys"""
        return list(self._DL.keys())

    def to_array(self, *, flatten=False):
        for key in self.keys:
            try:
                self.__dict__[key] = np.array(self._DL[key])
            except Exception as e:
                # get back to lists so we are in a valid state
                self.__dict__.update(self._DL)
                error_message = f"could not convert {key} into np.array"
                raise ValueError(error_message) from e
        if flatten:
            self.flatten()

    def flatten(self):
        """Drop trailing singleton axes, e.g. ``(n, 3, 1)`` to ``(n, 3)``.

        The old ``shape[2]`` lookup raised on 1-D and 2-D arrays and the
        exception was swallowed, so column vectors were never flattened.
        """

        for key in self.keys:
            arr = self.__dict__[key]
            if not isinstance(arr, np.ndarray):
                continue
            while arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            self.__dict__[key] = arr

    def __repr__(self):
        return f"<KFSaver object at {hex(id(self))}\n  Keys: {' '.join(self.keys)}>"


def mahalanobis(y, S) -> float:
    """Mahalanobis distance of the innovation ``y`` under covariance ``S``.

    Returns 0.0 before the first update (``S`` still all zeros) and falls back
    to the pseudo-inverse when ``S`` is singular, so reading the property never
    raises. Accepts ``y`` as either ``(dim_z,)`` or ``(dim_z, 1)``.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)
    if not np.any(S):
        return 0.0
    try:
        squared = float(y @ np.linalg.solve(S, y))
    except np.linalg.LinAlgError:
        squared = float(y @ np.linalg.pinv(S) @ y)
    return math.sqrt(max(squared, 0.0))


def as_covariance(value, dim: int) -> np.ndarray:
    """Expand a scalar variance to ``value * I``; pass matrices through.

    KF/EKF/UKF each carried a private copy; the EKF version also coerced to
    float, which the other two silently skipped.
    """

    return np.eye(dim) * value if np.isscalar(value) else np.asarray(value, dtype=float)


def likelihood_of(filt) -> float:
    """Likelihood of ``filt``'s last innovation, for IMM/MMAE mode weights.

    Filters such as KF/EKF/UKF/CKF expose a ``likelihood`` property. Others
    (notably `HInfinityFilter`) do not, which used to raise ``AttributeError``
    as soon as they were mixed into a bank. Reconstruct a Gaussian likelihood
    from ``f.y`` and the innovation/noise covariance (``S`` or ``V``) when the
    property is missing, and fall back to the smallest positive float when
    no covariance is available — the same floor the native properties use.
    """

    value = getattr(filt, "likelihood", None)
    if value is not None:
        return float(np.asarray(value).reshape(-1)[0])

    y = np.asarray(getattr(filt, "y", [])).reshape(-1)
    S = getattr(filt, "S", None)
    if S is None or not np.any(S):
        S = getattr(filt, "V", None)
    if y.size == 0 or S is None or not np.any(S):
        return sys.float_info.min

    S = np.asarray(S, dtype=float)
    try:
        squared = float(y @ np.linalg.solve(S, y))
        sign, logdet = np.linalg.slogdet(S)
    except np.linalg.LinAlgError:
        return sys.float_info.min
    if sign <= 0:
        return sys.float_info.min

    log_likelihood = -0.5 * (y.size * math.log(2.0 * math.pi) + logdet + squared)
    return max(math.exp(log_likelihood), sys.float_info.min)


def pretty_str(label, arr):
    def is_col(a):
        """return true if a is a column vector"""
        try:
            return a.shape[0] > 1 and a.shape[1] == 1
        except AttributeError, IndexError:
            return False

    # display empty lists correctly
    with contextlib.suppress(TypeError):
        if len(arr) == 0:
            return f"{label} = {type(arr)()!s}"
    if isinstance(arr, list | tuple | deque):
        return "\n".join(
            [pretty_str(f"{label}[{i!s}]", x) for (i, x) in enumerate(arr)]
        )

    if label is None:
        label = ""

    if label:
        label += " = "

    if is_col(arr):
        return label + str(arr.T).replace("\n", "") + ".T"

    rows = str(arr).split("\n")
    if not rows:
        return ""

    s = label + rows[0]
    pad = " " * len(label)
    for line in rows[1:]:
        s = s + "\n" + pad + line

    return s
