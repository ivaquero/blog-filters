"""Simulators that generate synthetic tracks, measurements, and maneuvers.

Submodules are imported on first access (PEP 562) rather than here. Importing
``simulators.datagen`` used to execute every sibling module, so it needed
``numpy`` plus ``scipy``, ``sympy`` and ``matplotlib`` — measured at 1464 ms, of
which roughly 560 ms was ``radar``/``robot`` dragging in plotting and symbolic
algebra that ``datagen`` never touches.
"""

import importlib

from ._types import Tracks

__all__ = ["Tracks", "datagen", "linear", "maneuver", "radar", "robot", "trajectory"]

_SUBMODULES = frozenset(__all__) - {"Tracks"}


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
