def rk4(y0, x0, dx, func):
    """One classical fourth-order Runge-Kutta step for ``dy/dx = func(y, x)``.

    Parameters
    ----------
    y0 : ndarray
        State at the start of the step.
    x0 : float
        Independent variable at the start of the step.
    dx : float
        Step size.
    func : callable
        Derivative ``func(y, x)`` returning the same shape as ``y0``.

    Returns
    -------
    ndarray
        State advanced by one step of size ``dx``.
    """

    k1 = dx * func(y0, x0)
    k2 = dx * func(y0 + 0.5 * k1, x0 + 0.5 * dx)
    k3 = dx * func(y0 + 0.5 * k2, x0 + 0.5 * dx)
    k4 = dx * func(y0 + k3, x0 + dx)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
