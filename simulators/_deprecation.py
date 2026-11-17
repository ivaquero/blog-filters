"""Deprecation shims for names that moved during the simulator refactor.

Each module keeps an ``_ALIASES`` table mapping a retired name to its
replacement — either a class in the same module (``(None, "NewName")``) or a
fully qualified import path (``("plots.plot_radar", "plot_radar3d")``). The
module-level ``__getattr__`` resolves the table on first access and emits a
``DeprecationWarning`` pointing at the caller.
"""

import importlib
import warnings

__all__ = ["deprecated_lookup"]


def deprecated_lookup(
    name: str,
    aliases: dict[str, tuple[str | None, str]],
    *,
    module_name: str,
    namespace: dict[str, object],
) -> object:
    """Resolve a retired attribute name, or raise ``AttributeError``.

    Parameters
    ----------
    name
        The attribute the caller asked for.
    aliases
        The module's ``_ALIASES`` table.
    module_name
        ``__name__`` of the calling module, used in messages.
    namespace
        The calling module's ``globals()``, so same-module aliases can be
        resolved without a second import.
    """

    target = aliases.get(name)
    if target is None:
        msg = f"module {module_name!r} has no attribute {name!r}"
        raise AttributeError(msg)

    target_module, attr = target
    qualified = (
        f"{module_name}.{attr}" if target_module is None else f"{target_module}.{attr}"
    )
    warnings.warn(
        f"{module_name}.{name} is deprecated; use {qualified} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    if target_module is None:
        return namespace[attr]
    return getattr(importlib.import_module(target_module), attr)
