"""SymPy derivation of the slant-range measurement Jacobian.

Moved out of ``simulators/radar.py``: a simulation module that needs SymPy at
import time costs ~210 ms before it can generate a single sample.
"""

import sympy as sy

__all__ = ["radar3d_symbol"]


def radar3d_symbol(*, print_symbol: bool = False):
    """Return ``(H, J, state)`` for the 3-state slant-range measurement.

    ``H`` is the measurement map ``hypot(x, z)`` written in the package's
    ``[x, x_vel, z]`` layout, ``J`` its Jacobian and ``state`` the symbol vector.

    Parameters
    ----------
    print_symbol
        Print ``H``, ``J`` and ``state``. The retired version printed them
        unconditionally from inside a function that returned ``None``.
    """

    x, x_vel, z = sy.symbols("x, x_vel z")
    measurement = sy.Matrix([sy.sqrt(x**2 + z**2)])
    state = sy.Matrix([x, x_vel, z])
    jacobian = measurement.jacobian(state)

    if print_symbol:
        print("state: ", state)
        print("H: ", measurement)
        print("J: ", jacobian)

    return measurement, jacobian, state
