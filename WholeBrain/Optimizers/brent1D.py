from __future__ import division

inf = float("inf")

_eps = 1.4902e-08
_golden = 0.381966011250105097


def brent(f, a=-inf, b=+inf, x0=None, f0=None, rtol=_eps, atol=_eps, maxiter=500):
    """ Seeks a minimum of a function via Brent's method.

    Given a function ``f`` with a minimum in the interval ``a <= b``, seeks a local
    minimum using a combination of golden section search and successive parabolic
    interpolation.

    Let ``tol = rtol * abs(x0) + atol``, where ``x0`` is the best guess found so far.
    It converges if evaluating a next guess would imply evaluating ``f`` at a point that
    is closer than ``tol`` to a previously evaluated one or if the number of iterations
    reaches ``maxiter``.

    Parameters
    ----------
    f : object
        Objective function to be minimized.
    a : float, optional
        Interval's lower limit. Defaults to ``-inf``.
    b : float, optional
        Interval's upper limit. Defaults to ``+inf``.
    x0 : float, optional
        Initial guess. Defaults to ``None``, which implies that::

            x0 = a + 0.382 * (b - a)
            f0 = f(x0)

    f0 : float, optional
        Function evaluation at ``x0``.
    rtol : float
        Relative tolerance. Defaults to ``1.4902e-08``.
    atol : float
        Absolute tolerance. Defaults to ``1.4902e-08``.
    maxiter : int
        Maximum number of iterations.


    Returns
    -------
    float
        Best guess ``x`` for the minimum of ``f``.
    float
        Value ``f(x)``.
    int
        Number of iterations performed.

    References
    ----------
    - http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c
    - Numerical Recipes 3rd Edition: The Art of Scientific Computing
    - https://en.wikipedia.org/wiki/Brent%27s_method
    """
    # a, b: interval within the minimum should lie
    #       no function evaluation will be requested
    #       outside that range.
    # x0: least function value found so far (or the most recent one in
    #                                            case of a tie)
    # x1: second least function value
    # x2: previous value of x1
    # (x0, x1, x2): Memory triple, updated at the end of each interation.
    # u : point at which the function was evaluated most recently.
    # m : midpoint between the current interval (a, b).
    # d : step size and direction.
    # e : memorizes the step size (and direction) taken two iterations ago
    #      and it is used to (definitively) fall-back to golden-section steps
    #      when its value is too small (indicating that the polynomial fitting
    #      is not helping to speedup the convergence.)
    #
    #
    # References: Numerical Recipes: The Art of Scientific Computing
    # http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c

    if a > b:
        raise ValueError("'a' must be equal or smaller than 'b'")

    if x0 is None:
        x0 = a + _golden * (b - a)
    else:
        if not (a <= x0 <= b):
            raise RuntimeError("'x0' didn't fall in-between 'a' and 'b'")
    f0 = f(x0)

    x1 = x0
    x2 = x1
    niters = -1
    d = 0.0
    e = 0.0
    f1 = f0
    f2 = f1

    for niters in range(maxiter):

        m = 0.5 * (a + b)
        tol = rtol * abs(x0) + atol
        tol2 = 2.0 * tol

        # Check the stopping criterion.
        if abs(x0 - m) <= tol2 - 0.5 * (b - a):
            break

        r = 0.0
        q = r
        p = q

        # "To be acceptable, the parabolic step must (i) fall within the
        # bounding interval (a, b), and (ii) imply a movement from the best
        # current value x0 that is less than half the movement of the step
        # before last."
        #   - Numerical Recipes 3rd Edition: The Art of Scientific Computing.

        if tol < abs(e):
            # Compute the polynomial of the least degree (Lagrange polynomial)
            # that goes through (x0, f0), (x1, f1), (x2, f2).
            r = (x0 - x1) * (f0 - f2)
            q = (x0 - x2) * (f0 - f1)
            p = (x0 - x2) * q - (x0 - x1) * r
            q = 2.0 * (q - r)
            if 0.0 < q:
                p = -p
            q = abs(q)
            r = e
            e = d

        if abs(p) < abs(0.5 * q * r) and q * (a - x0) < p and p < q * (b - x0):
            # Take the polynomial interpolation step.
            d = p / q
            u = x0 + d

            # Function must not be evaluated too close to a or b.
            if (u - a) < tol2 or (b - u) < tol2:
                if x0 < m:
                    d = tol
                else:
                    d = -tol
        else:
            # Take the golden-section step.
            if x0 < m:
                e = b - x0
            else:
                e = a - x0
            d = _golden * e

        # Function must not be evaluated too close to x0.
        if tol <= abs(d):
            u = x0 + d
        elif 0.0 < d:
            u = x0 + tol
        else:
            u = x0 - tol

        # Notice that we have u \in [a+tol, x0-tol] or
        #                     u \in [x0+tol, b-tol],
        # (if one ignores rounding errors.)
        fu = f(u)

        # Housekeeping.

        # Is the most recently evaluated point better (or equal) than the
        # best so far?
        if fu <= f0:

            # Decrease interval size.
            if u < x0:
                if b != x0:
                    b = x0
            else:
                if a != x0:
                    a = x0

            # Shift: drop the previous third best point out and
            # include the newest point (found to be the best so far).
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0
            x0 = u
            f0 = fu

        else:
            # Decrease interval size.
            if u < x0:
                if a != u:
                    a = u
            else:
                if b != u:
                    b = u

            # Is the most recently evaluated point at better (or equal)
            # than the second best one?
            if fu <= f1 or x1 == x0:
                # Insert u between (rank-wise) x0 and x1 in the triple
                # (x0, x1, x2).
                x2 = x1
                f2 = f1
                x1 = u
                f1 = fu
            elif fu <= f2 or x2 == x0 or x2 == x1:
                # Insert u in the last position of the triple (x0, x1, x2).
                x2 = u
                f2 = fu

        print(f'x0={x0}, f0={f0}, niter+1={niters+1}')

    return x0, f0, niters + 1

if __name__ == '__main__':
    def func(x):
            # return x**2 - 0.8
            return (x+3)*(x-1)**2

    r = brent(lambda x: func(x),
              x0=2.,
              a=-10., b=10.)
    print('\n\nfinished!!!\n\n')
    print(f'x0={r[0]}, f0={r[1]}, niter={r[2]}')
    # The output should be
    #
    # (0.0, -0.8, 6) for first
    # (Correct : Iter 9 : f(s) = -1.4E-07, Iter 10 : f(s) = 6.96E-12) for second
