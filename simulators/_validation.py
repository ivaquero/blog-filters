"""Argument validation shared by the simulators.

The previous modules each carried their own copy of these checks, and
``datagen._validate_variance`` was reused for standard deviations, which made
the error message disagree with the parameter it was guarding.
"""

__all__ = ["require_count", "require_nonnegative", "require_positive"]


def require_nonnegative(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value >= 0``.

    Used for variances, standard deviations and time steps alike — all of them
    are non-negative quantities, and naming the quantity in the message is what
    keeps the traceback readable.
    """

    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def require_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value > 0``."""

    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def require_count(num: int) -> int:
    """Validate a sample count and return it, so call sites can inline it."""

    if num < 0:
        msg = f"num must be non-negative, got {num}"
        raise ValueError(msg)
    return num
