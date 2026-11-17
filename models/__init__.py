"""State-space models (linear and nonlinear) shared by the notebooks.

The classes are the supported entry point: they satisfy
:class:`models.ssmodel.StateSpaceModel`, which in turn satisfies the
``StateSpaceModelLike`` protocol in ``filters._typing``, so a model can be handed
straight to :meth:`filters.kalman_ukf.UnscentedKalmanFilter.from_model` or
:meth:`filters.kalman_ekf.ExtendedKalmanFilter.from_model`.

The free functions (``constant_velocity_transition_matrix`` and friends) remain
public because the notebooks build filters from matrices; they are no longer the
only way in.
"""

from . import (
    clutter,
    const_acc,
    const_vel,
    coord_turn,
    noise,
    ssmodel,
    ssmodel_linear,
    ssmodel_nonlinear,
)
from .ssmodel import NonlinearStateSpaceModel, StateSpaceModel
from .ssmodel_linear import (
    ConstantAcceleration,
    ConstantVelocity,
    LinearModel,
    LinearStateSpaceModel,
)
from .ssmodel_nonlinear import CoordinatedTurn

__all__ = [
    "ConstantAcceleration",
    "ConstantVelocity",
    "CoordinatedTurn",
    "LinearModel",
    "LinearStateSpaceModel",
    "NonlinearStateSpaceModel",
    "StateSpaceModel",
    "clutter",
    "const_acc",
    "const_vel",
    "coord_turn",
    "noise",
    "ssmodel",
    "ssmodel_linear",
    "ssmodel_nonlinear",
]
