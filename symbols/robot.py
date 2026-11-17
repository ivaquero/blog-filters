"""SymPy derivation of the bicycle-model Jacobians.

Kept separate from the runtime filter so that :mod:`filters.robot_ekf` can carry
hand-written ``numpy`` Jacobians (microseconds) while this module remains the
readable reference that ``tests/test_robot_ekf.py`` checks them against.
"""

import sympy as sy

__all__ = ["robot3d_symbol"]


def robot3d_symbol(*, print_symbol: bool = False):
    """Return ``(fxu, F_j, V_j, vars_)`` for the differential-drive robot.

    Parameters
    ----------
    print_symbol
        Print the three matrices. The retired version printed ``fxu`` under all
        three headings, so ``F_j`` and ``V_j`` were never actually shown.

    Returns
    -------
    tuple
        ``fxu`` is the transition map, ``F_j`` its Jacobian with respect to the
        state ``[x, y, theta]``, ``V_j`` the Jacobian with respect to the
        control ``[v, a]``, and ``vars_`` the symbols ``(a, x, y, v, w, θ, t)``.
    """

    vars_ = sy.symbols("a, x, y, v, w, θ, t")
    a, x, y, v, w, θ, time = vars_
    distance = v * time
    β = (distance / w) * sy.tan(a)
    r = w / sy.tan(a)

    fxu = sy.Matrix(
        [
            [x - r * sy.sin(θ) + r * sy.sin(θ + β)],
            [y + r * sy.cos(θ) - r * sy.cos(θ + β)],
            [θ + β],
        ]
    )
    F_j = fxu.jacobian(sy.Matrix([x, y, θ]))
    V_j = fxu.jacobian(sy.Matrix([v, a]))

    if print_symbol:
        print("fxu: ", fxu)
        print("F_j: ", F_j)
        print("V_j: ", V_j)

    return fxu, F_j, V_j, vars_
