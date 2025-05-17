from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from singledispatchmethod import singledispatchmethod
from typing_extensions import Protocol, runtime

T = TypeVar("T")


@runtime
class StateEstimator(Protocol[T]):
    def predict(self, eststate: T, Ts: float) -> T: ...

    def update(
        self, z: np.ndarray, eststate: T, *, sensor_state: dict[str, Any] | None = None
    ) -> T: ...

    def step(
        self,
        z: np.ndarray,
        eststate: T,
        Ts: float,
        *,
        sensor_state: dict[str, Any] | None = None,
    ) -> T: ...

    def estimate(self, eststate: T): ...

    def loglikelihood(
        self, z: np.ndarray, eststate: T, *, sensor_state: dict[str, Any] | None = None
    ) -> float: ...

    def reduce_mixture(self, estimator_mixture) -> T: ...

    def gate(
        self,
        z: np.ndarray,
        eststate: T,
        gate_size_square: float,
        *,
        sensor_state: dict[str, Any] | None = None,
    ) -> bool: ...


@dataclass(init=False)
class GaussParams:
    """A class for holding Gaussian parameters"""

    __slots__ = ["cov", "mean"]
    mean: np.ndarray  # shape=(n,)
    cov: np.ndarray  # shape=(n, n)

    def __init__(self, mean, cov) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)

    def __iter__(self):  # in order to use tuple unpacking
        return iter((self.mean, self.cov))


@dataclass(init=False)
class GaussParamList:
    __slots__ = ["cov", "mean"]
    mean: np.ndarray  # shape=(N, n)
    cov: np.ndarray  # shape=(N, n, n)

    def __init__(self, mean=None, cov=None):
        if mean is not None and cov is not None:
            self.mean = mean
            self.cov = cov
        else:
            # container left empty
            pass

    @classmethod
    def allocate(
        cls,
        shape: int | tuple[int, ...],  # list shape
        n: int,  # dimension
        fill: float | None = None,  # fill the allocated arrays
    ) -> GaussParamList:
        if isinstance(shape, int):
            shape = (shape,)

        if fill is None:
            return cls(np.empty((*shape, n)), np.empty((*shape, n, n)))

        return cls(np.full((*shape, n), fill), np.full((*shape, n, n), fill))

    def __getitem__(self, key):
        theCls = GaussParams if isinstance(key, int) else GaussParamList
        return theCls(self.mean[key], self.cov[key])

    def __setitem__(self, key, value):
        if isinstance(value, GaussParams | tuple):
            self.mean[key], self.cov[key] = value
        elif isinstance(value, GaussParamList):
            self.mean[key] = value.mean
            self.cov[key] = value.cov
        else:
            raise NotImplementedError(f"Cannot set from type {value}")

    def __len__(self):
        return self.mean.shape[0]

    def __iter__(self):
        yield from (self[k] for k in range(len(self)))


@dataclass
class MixtureParameters(Generic[T]):
    __slots__ = ["components", "weights"]
    weights: np.ndarray
    components: Sequence[T]


class Array(Collection[T], Generic[T]):
    def __getitem__(self, key): ...

    def __setitem__(self, key, value): ...


@dataclass
class MixtureParametersList(Generic[T]):
    weights: np.ndarray
    components: Array[Sequence[T]]

    @classmethod
    def allocate(cls, shape: int | tuple[int, ...], component_type: T):
        shape = (shape,) if isinstance(shape, int) else shape
        raise NotImplementedError

    @singledispatchmethod
    def __getitem__(self, key: Any) -> MixtureParametersList[T]:
        return MixtureParametersList(self.weights[key], self.components[key])

    @__getitem__.register
    def _(self, key: int) -> MixtureParameters:
        return MixtureParameters(self.weights[key], self.components[key])

    def __setitem__(
        self, key: int | slice, value: MixtureParameters[T] | MixtureParametersList[T]
    ) -> None:
        self.weights[key] = value.weights
        self.components[key] = value.components

    def __len__(self):
        return self.weights.shape[0]

    def __iter__(self):
        yield from (self[k] for k in range(len(self)))
