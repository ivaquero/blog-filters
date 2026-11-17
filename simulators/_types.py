"""Return shape shared by every batch simulator."""

from typing import NamedTuple

import numpy as np

__all__ = ["Tracks"]


class Tracks(NamedTuple):
    """Truth and measurement samples produced by one simulation run.

    The three fields are always index-aligned: ``t[i]`` is the time of the true
    state ``x[i]`` and of the measurement ``z[i]`` taken from it.

    Attributes
    ----------
    t
        Sample times, shape ``(N,)``. Simulators that advance before measuring
        start at ``dt``; the initial state is not sampled.
    x
        True state, shape ``(N, D)``. ``D`` is the state dimension of the
        scenario (``1`` for a scalar walk, ``2`` for ``[x, v]``, ...).
    z
        Measurement, shape ``(N, M)``, or ``None`` when the scenario has no
        sensor (pure kinematics).

    Examples
    --------
    >>> tracks = linear.DogSimulation(
    ...     vel_var=0.0, meas_var=0.0
    ... ).simulate(num=3)
    >>> tracks.x.shape, tracks.z.shape
    ((3, 1), (3, 1))

    Notes
    -----
    A ``NamedTuple`` rather than a dataclass so callers can keep the historical
    ``tracks.x`` style without losing the ability to index positionally.
    """

    t: np.ndarray
    x: np.ndarray
    z: np.ndarray | None = None
