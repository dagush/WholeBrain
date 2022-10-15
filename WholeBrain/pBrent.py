# --------------------------------------------------------------
# Based on Brent's method for univariate function optimization.
# Code by Danilo Horta
# available at https://github.com/limix/brent-search
#
# Modified for parallel evaluation by gustavo Patow
# --------------------------------------------------------------
from __future__ import division
import numpy as np

inf = np.inf # float("inf")

_eps = 1.4902e-08
_golden = 0.381966011250105097


def pBrent(f, N=1, a=None, b=None, x0=None, rtol=_eps, atol=_eps, maxiter=5000):
    """ Seeks the minimums of a set of WholeBrain in parallel, each via Brent's method.

    Given a set of WholeBrain ``f``, each with a minimum in the interval ``a <= b``, for each one seeks a local
    minimum using a combination of golden section search and successive parabolic interpolation.

    Let ``tol = rtol * abs(x0) + atol``, where ``x0`` is the best guess found so far.
    It converges if evaluating a next guess would imply evaluating ``f`` at a point that
    is closer than ``tol`` to a previously evaluated one or if the number of iterations
    reaches ``maxiter``.

    Parameters
    ----------
    f : object
        Objective function vector to be minimized.
    a : array of floats, optional
        Interval's lower limits. Defaults to ``-inf``.
    b : array of floats, optional
        Interval's upper limits. Defaults to ``+inf``.
    x0 : array of floats, optional
        Initial guesses. Defaults to ``None``, which implies that::

            x0 = a + 0.382 * (b - a)
            f0 = f(x0)

    rtol : float
        Relative tolerance. Defaults to ``1.4902e-08``.
    atol : float
        Absolute tolerance. Defaults to ``1.4902e-08``.
    maxiter : int
        Maximum number of iterations for the whole set.


    Returns
    -------
    array of floats
        Best guesses ``x`` for the minimums of ``f``.
    array of floats
        Values of ``f(x)`` for each x.
    int
        Number of iterations performed.

    References
    ----------
    - http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c
    - Numerical Recipes 3rd Edition: The Art of Scientific Computing
    - https://en.wikipedia.org/wiki/Brent%27s_method

    based on the code by Danilo Horta
    available at https://github.com/limix/brent-search
    """
    # a, b: intervals within the minimums should lie
    #       no function evaluation will be requested
    #       outside that range.
    # x0: least function values found so far (or the most recent ones in
    #                                            case of ties)
    # x1: second least function values
    # x2: previous values of x1
    # (x0, x1, x2): Memory triples, updated at the end of each interation.
    # u : points at which the function was evaluated most recently.
    # m : midpoints between the current interval (a, b).
    # d : step sizes and directions.
    # e : memorizes the step sizes (and directions) taken two iterations ago
    #      and they are used to (definitively) fall-back to golden-section steps
    #      when their values are too small (indicating that the polynomial fittings
    #      are not helping to speedup the convergence.)
    #
    # References: Numerical Recipes: The Art of Scientific Computing
    # http://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.c
    if a is None:
        a = np.full((N,), -np.inf)
    if b is None:
        b = np.full((N,), np.inf)

    if np.any(a > b):
        raise ValueError("All 'a' values must be equal or smaller than 'b' values")

    if x0 is None:
        x0 = a + _golden * (b - a)
    else:
        if not np.all((a <= x0) & (x0 <= b)):
            raise RuntimeError("An 'x0' didn't fall in-between 'a' and 'b'")
    f0 = f(x0)

    u = x0.copy()
    x1 = x0.copy()
    x2 = x1.copy()
    # niters = -1
    d = np.zeros(N)  # 0.0
    e = np.zeros(N)  # 0.0
    # if f0 is not None:
    f1 = f0.copy()
    f2 = f1.copy()
    success = np.zeros(N, dtype=bool)
    niters = np.zeros(N)
    f0s = np.zeros(N)
    x0s = np.zeros(N)

    # -------------------------------------------------------------
    for niter in range(maxiter):

        m = 0.5 * (a + b)
        tol = rtol * np.max(np.abs(x0)) + atol
        tol2 = 2.0 * tol

        # -------------------------------------------------------------
        # Check the stopping criterion.
        criterion = (np.abs(x0 - m) <= tol2 - 0.5 * (b - a))
        applyChange = np.logical_not(success) & criterion  # These are the ones who have finished
        x0s = np.where(applyChange, x0, x0s)
        f0s = np.where(applyChange, f0, f0s)
        niters = np.where(applyChange, niter, niters)
        success |= np.where(np.abs(x0 - m) <= tol2 - 0.5 * (b - a), True, False)
        if np.all(success):
            break

        # r = np.zeros(N)  #0.0
        # q = r.copy()
        # p = q.copy()

        # "To be acceptable, the parabolic step must (i) fall within the
        # bounding interval (a, b), and (ii) imply a movement from the best
        # current value x0 that is less than half the movement of the step
        # before last."
        #   - Numerical Recipes 3rd Edition: The Art of Scientific Computing.

        # -------------------------------------------------------------
        # if tol < abs(e):
        #     r = (x0 - x1) * (f0 - f2)
        #     q = (x0 - x2) * (f0 - f1)
        #     p = (x0 - x2) * q - (x0 - x1) * r
        #     q = 2.0 * (q - r)
        #     if 0.0 < q:
        #         p = -p
        #     q = abs(q)
        #     r = e
        #     e = d
        condition = tol < abs(e)
        # Compute the polynomial of the least degree (Lagrange polynomial)
        # that goes through (x0, f0), (x1, f1), (x2, f2).
        r = np.where(condition, (x0 - x1) * (f0 - f2), 0.0)
        q = np.where(condition, (x0 - x2) * (f0 - f1), 0.0)
        p = np.where(condition, (x0 - x2) * q - (x0 - x1) * r, 0.0)
        q = np.where(condition, 2.0 * (q - r), q)
        p = np.where(condition & (0.0 < q), -p, p)
        q = np.where(condition, np.abs(q), q)
        r = np.where(condition, e, r)
        e = np.where(condition, d, e)

        # -------------------------------------------------------------
        # if abs(p) < abs(0.5 * q * r) and q * (a - x0) < p and p < q * (b - x0):
        #     # # Take the polynomial interpolation step.
        #     # d = p / q
        #     # u = x0 + d
        #     # Function must not be evaluated too close to a or b.
        #     if (u - a) < tol2 or (b - u) < tol2:
        #         if x0 < m:
        #             d = tol
        #         else:
        #             d = -tol
        #     # d2 = np.where(condition,  # Function must not be evaluated too close to a or b.
        #     #              np.where(((u - a) < tol2) | ((b - u) < tol2),
        #     #                       np.where(x0 < m, tol, -tol),
        #     #                       d),
        #     #              d)
        # else:
        #     # Take the golden-section step.
        #     if x0 < m:
        #         e = b - x0
        #     else:
        #         e = a - x0
        #     d = _golden * e
        #     # d2 = np.where(np.logical_not(condition), _golden * np.where(x0 < m, b - x0, a - x0), d)  # Take the golden-section step.

        condition = (abs(p) < abs(0.5 * q * r)) & (q * (a - x0) < p) & (p < q * (b - x0))
        d = np.where(condition, p/q, d)  # Take the polynomial interpolation step.
        u = np.where(condition, x0 + d, u)
        e = np.where(np.logical_not(condition), np.where(x0 < m, b - x0, a - x0), e)
        d = np.where(condition,  # Function must not be evaluated too close to a or b.
                     np.where(((u - a) < tol2) | ((b - u) < tol2),
                              np.where(x0 < m, tol, -tol),
                              d),
                     _golden * e)
        # print(f'd={d} u={u}   ', end='')

        # -------------------------------------------------------------
        # # Function must not be evaluated too close to x0.
        # if tol <= abs(d):
        #     u = x0 + d
        # elif 0.0 < d:
        #     u = x0 + tol
        # else:
        #     u = x0 - tol
        # Function must not be evaluated too close to x0.
        u = np.where(tol <= abs(d), x0 + d,
                     np.where(0.0 < d, x0 + tol, x0 - tol))
        # print(f'u updated={u} ', end='')

        # Notice that we have u \in [a+tol, x0-tol] or
        #                     u \in [x0+tol, b-tol],
        # (if one ignores rounding errors.)
        fu = f(u)
        # print(f'f(u)={fu}', end='')

        # Housekeeping.

        # Is the most recently evaluated point better (or equal) than the best so far?
        condition = fu <= f0
        # Decrease interval size.
        b = np.where(condition & (u < x0) & (b != x0), x0, b)
        a = np.where(condition & np.logical_not(u < x0) & (a != x0), x0, a)
        # Shift: drop the previous third best point out and include the newest point (found to be the best so far).
        x2 = np.where(condition, x1, x2)
        f2 = np.where(condition, f1, f2)
        x1 = np.where(condition, x0, x1)
        f1 = np.where(condition, f0, f1)
        x0 = np.where(condition, u, x0)
        f0 = np.where(condition, fu, f0)

        notCondition = np.logical_not(condition)
        # Decrease interval size.
        a = np.where(notCondition & (u < x0) & (a != u), x0, a)
        b = np.where(notCondition & np.logical_not(u < x0) & (b != u), x0, b)
        # Is the most recently evaluated point at better (or equal) than the second best one?
        subCondition = notCondition & ((fu <= f1) | (x1 == x0))
        # Insert u between (rank-wise) x0 and x1 in the triple (x0, x1, x2).
        x2 = np.where(subCondition, x1, x2)
        f2 = np.where(subCondition, f1, f2)
        x1 = np.where(subCondition, u, x1)
        f1 = np.where(subCondition, fu, f1)
        notSubCondition = np.logical_not(subCondition) & ((fu <= f2) | (x2 == x0) | (x2 == x1))  # Probably this can be much simplified
        # Insert u in the last position of the triple (x0, x1, x2).
        x2 = np.where(notSubCondition, u, x2)
        f2 = np.where(notSubCondition, fu, f2)

        # # Is the most recently evaluated point better (or equal) than the best so far?
        # if fu <= f0:
        #     pass
        #         # Decrease interval size.
        #         if u < x0:
        #             if b != x0:
        #                 b = x0
        #         else:
        #             if a != x0:
        #                 a = x0
        #         # Shift: drop the previous third best point out and include the newest point (found to be the best so far).
        #         x2 = x1
        #         f2 = f1
        #         x1 = x0
        #         f1 = f0
        #         x0 = u
        #         f0 = fu
        # else:
        #         # Decrease interval size.
        #         if u < x0:
        #             if a != u:
        #                 a = u
        #         else:
        #             if b != u:
        #                 b = u
        #         # Is the most recently evaluated point at better (or equal) than the second best one?
        #         if fu <= f1 or x1 == x0:
        #             # Insert u between (rank-wise) x0 and x1 in the triple (x0, x1, x2).
        #             x2 = x1
        #             f2 = f1
        #             x1 = u
        #             f1 = fu
        #         elif fu <= f2 or x2 == x0 or x2 == x1:
        #             # Insert u in the last position of the triple (x0, x1, x2).
        #             x2 = u
        #             f2 = fu
        #     pass

        print(f'x0={x0}, f0={f0}, niter+1={niter+1}')

    return {'x': x0s, 'fun': f0s, 'nfev': niters + 1, 'status': np.all(success), 'message': 'success' if np.all(success) else 'failure'}


if __name__ == '__main__':
    numDims = 3
    def func(x):
        # return x**2 - 0.8
        return np.array([(x[0])**2 - 0.8, (x[1]+3)*(x[1]-1)**2, ((x[1]+3)*(x[1]-1)**2)**2])
        # return (x+3)*(x-1)**2

    r = pBrent(lambda x: func(x),
               x0=np.array([10., 0., 0.]),
               N=numDims,
               a=np.full((numDims,), -10.), b=np.full((numDims,), 10.))
    print('\n\nfinished!!!\n\n')
    for x0, f0, niter in zip(r[0], r[1], r[2]):
        print(f'x0={x0}, f0={f0}, niter={niter}')
    # The output should be
    #
    # (0.0, -0.8, 6) for first
    # (Correct : Iter 9 : f(s) = -1.4E-07, Iter 10 : f(s) = 6.96E-12) for second
