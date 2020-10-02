# ==========================================================================
# ==========================================================================
# Set of functions to analyse and plot a system of ODEs with numerical tools.
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.integrate
from scipy.optimize import fsolve
from numpy import linalg as LA

# Plotting modules
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as mpatches #used to write custom legends

trajectoyLength = 500

# ==========================================================================
# ==========================================================================
# EVALUATION FUNCTIONS
# ==========================================================================
# ==========================================================================
def Jac(f, x, dx=1e-8):
    # n = len(x)
    # t = 0.
    # fx = f(x, t)
    # jac = np.zeros((n, n))
    # for j in range(n): #through columns to allow for vector addition
    #     Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
    #     x_plus = [(xi if k != j else xi+Dxj) for k, xi in enumerate(x)]
    #     jac[:, j] = (f(x_plus, t)-fx)/Dxj
    # return jac
    import numdifftools as nd
    J = nd.Jacobian(f)
    return J(x)


# -----------------------------------------------------------------------------
# finds root of equation function(x)=0
# Uses scipy.optimize.fsolve
# -----------------------------------------------------------------------------
def findroot(func, init):
    """ Find root of equation function(x)=0
    Args:
        - the system (function),
        - the initial values (type list or np.array)

    return: correct equilibrium (type np.array)
            if the numerical method converge or return nan
    Adapted from:
        https://www.normalesup.org/~doulcier/teaching/modeling/bistable_systems.html
    """
    sol, info, convergence, sms = fsolve(func, init, args=0., full_output=1)
    if convergence == 1:
        return sol
    return np.array([np.nan]*len(init))


# -----------------------------------------------------------------------------
# finds all rots of an equation, numerically using the Method of Bisection
# (it is ideal for the roots forming curves, as in nullclines, not for points)
# -----------------------------------------------------------------------------
def findAllRoots(eq, interval, n_grid=30):
    import math

    # It's based on chapter 4.3, Method of Bisection, from the book
    # Kiusalaas, Jaan; Numerical Methods in Engineering with Python, 3rd Ed.
    def rootsearch(f, a, b, dx):
        x1 = a; f1 = f(a)
        x2 = a + dx; f2 = f(x2)
        while f1*f2 > 0.0:
            if x1 >= b:
                return None,None
            x1 = x2; f1 = f2
            x2 = x1 + dx; f2 = f(x2)
        return x1,x2

    def bisect(f, x1, x2, switch=0, epsilon=1.0e-9):
        f1 = f(x1)
        if f1 == 0.0:
            return x1
        f2 = f(x2)
        if f2 == 0.0:
            return x2
        if f1*f2 > 0.0:
            print('Root is not bracketed')
            return None
        n = int(math.ceil(math.log(abs(x2 - x1)/epsilon)/math.log(2.0)))
        for i in range(n):
            x3 = 0.5*(x1 + x2); f3 = f(x3)
            if (switch == 1) and (abs(f3) >abs(f1)) and (abs(f3) > abs(f2)):
                return None
            if f3 == 0.0:
                return x3
            if f2*f3 < 0.0:
                x1 = x3
                f1 = f3
            else:
                x2 =x3
                f2 = f3
        return (x1 + x2)/2.0

    def roots(f, a, b, eps=1e-6):
        print('The roots on the interval [%f, %f] are:' % (a,b))
        rSet = []
        while 1:
            x1,x2 = rootsearch(f,a,b,eps)
            if x1 != None:
                a = x2
                root = bisect(f,x1,x2,1)
                if root != None:
                    rSet.append(round(root,-int(math.log(eps, 10))))
                    print("  >", round(root,-int(math.log(eps, 10))))
            else:
                print('  > Done')
                break
        return rSet

    roots0 = []
    roots1 = []
    multiple0 = False
    multiple1 = False

    uSpace = np.linspace(interval['left'], interval['right'], n_grid)
    for u in uSpace:
        f = lambda v: eq([u, v], None)[0]
        print("u:{}[0]".format(u), end=' ')
        rootSet = roots(f, interval['bottom'], interval['top'])
        if len(rootSet) > 1: multiple0 = True
        for r in rootSet: roots0.append([u,r])

        f = lambda v: eq([u, v], None)[1]
        print("u:{}[1]".format(u), end=' ')
        rootSet = roots(f, interval['bottom'], interval['top'])
        if len(rootSet) > 1: multiple1 = True
        for r in rootSet: roots1.append([u,r])

    vSpace = np.linspace(interval['bottom'], interval['top'], n_grid)
    for v in vSpace:
        f = lambda u: eq([u, v], None)[0]
        print("v:{}[0]".format(v), end=' ')
        rootSet = roots(f, interval['left'], interval['right'])
        if len(rootSet) > 1: multiple0 = True
        for r in rootSet: roots0.append([r,v])

        f = lambda u: eq([u, v], None)[1]
        print("v:{}[1]".format(v), end=' ')
        rootSet = roots(f, interval['left'], interval['right'])
        if len(rootSet) > 1: multiple1 = True
        for r in rootSet: roots1.append([r,v])

    # sort them, to give some order
    roots0.sort()
    roots1.sort()
    return roots0, multiple0, roots1, multiple1


# --------------------------------------------------------------------------
# Classifies the Fixed Points
# --------------------------------------------------------------------------
UNSTABLE_FP = 0
STABLE_FP = 1
def classify_fixedPoint(f, fp):
    J = Jac(f, fp)
    evalues, evectors = LA.eig(J)
    if np.any(evalues.real > 0):
        # Here we only care about stable vs. unstable fixed points
        # However, if one evalue > 0 and the other < 0, we have a saddle fp...
        #          if both evalues > 0, we have a source
        return UNSTABLE_FP
    else:  # Both evalues < 0 => a sink!
        return STABLE_FP


def stability_fixedPoint(f, fp):
    """ Stability of the equilibrium (computes its associated 2x2 jacobian matrix).
    Args:
        The funciton to evaluate and the equilibrium point.
    Return:
        (string) status of equilibrium point 
    Adapted from:
        https://www.normalesup.org/~doulcier/teaching/modeling/bistable_systems.html
    """
    J = Jac(f, fp)
    determinant = np.linalg.det(J)
    trace = np.matrix.trace(J)
    if np.isclose(trace,0) and np.isclose(determinant,0):
        nature = "Center (Hopf)"
    elif np.isclose(determinant,0):
        nature = "Transcritical (Saddle-Node)"
    elif determinant < 0:
        nature = "Saddle"
    else:
        nature = "Stable" if trace < 0 else "Unstable"
        nature += " focus" if (trace**2 - 4 * determinant) < 0 else " node"
    return nature


# --------------------------------------------------------------------------
# finds the Fixed Points
# --------------------------------------------------------------------------
def fixedPoints(func, interval, n_grid=30, TOLERANCE = 1e-5):
    def isKnown(x, equilibriaSet):
        return any(np.isnan(x)) or \
               any([all(np.isclose(x, e)) for e in equilibriaSet])

    # Set up u,v space
    u = np.linspace(interval['left'], interval['right'], n_grid)
    v = np.linspace(interval['bottom'], interval['top'], n_grid)
    uu, vv = np.meshgrid(u, v)

    equilibria = []

    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            x = findroot(func, np.array([uu[i,j], vv[i,j]]))  # fsolve(eq, np.array([uu[i,j], vv[i,j]]), args=0.)
            if not isKnown(x, equilibria):
                equilibria.append(x)
    return equilibria


# --------------------------------------------------------------------------
# finds limit cycles
# --------------------------------------------------------------------------
def findLimitCycles(model, fixedPoints, interval, n_grid=10):
    def isFixedPoint(x):
        return any(np.isnan(x)) or \
               any([all(np.isclose(x, e, rtol=1e-3, atol=1e-3)) for e in fixedPoints])
    def alreadySeen(bbox):
        return [np.isclose(bbox, e, rtol=1e-3) for e in limitCycles]

    # Set up u,v space
    u = np.linspace(interval['left'], interval['right'], n_grid)
    v = np.linspace(interval['bottom'], interval['top'], n_grid)
    uu, vv = np.meshgrid(u, v)

    limitCycles = []

    t = np.linspace(0, 100, trajectoyLength)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            y = computeTrajectory(model.dfun, np.array([uu[i,j], vv[i,j]]), t=t)  # fsolve(eq, np.array([uu[i,j], vv[i,j]]), args=0.)
            lastPoint = y[-1]
            minValues = np.min(y[300:], axis=0); maxValues = np.max(y[300:], axis=0)
            bbox = np.array([minValues, maxValues])
            if not isFixedPoint(lastPoint):
                if not alreadySeen(bbox):
                    limitCycles.append(bbox)
    return limitCycles


# --------------------------------------------------------------------------
# Given an initial position, computes the continuation of `F(u, lambda) = 0`
# --------------------------------------------------------------------------
def numerical_continuation(f, set_lbda_func, initial_u, lbda_values):
    """ Find the roots of the parametrised non linear equation.

    Iteratively find approximate solutions of `F(u, lambda) = 0`
    for several values of lambda. The solution of the step i is
    used as initial guess for the numerical solver at step i+1.
    The first inital guess is initial_u (for lbda_values[0]).

    Args:
        f (function): Function of u and lambda.
        initial_u (float): Starting point for the contiunation.
        lbda_values (array): Values of the parameter lambda (in the order of the continuation process).

    Return:
        (numpy.array) output[i] is the solutions of f(u,lbda_values[i]) = 0
         NaN if the algorithm did not converge.
    Adapted from:
        https://www.normalesup.org/~doulcier/teaching/modeling/bistable_systems.html
    """
    def func(x,lbda):
        set_lbda_func(lbda)
        return f(x,lbda)
    eq = []
    for lbda in lbda_values:
        eq.append(findroot(lambda x,t: func(x,lbda),
                           eq[-1] if eq else initial_u))
    return eq


def get_branches(func, set_lbda_func, starting_points, lbda_space):
    def stability_fixedPoint_lbda(x, lbda):
        set_lbda_func(lbda)
        return stability_fixedPoint(func, x)

    branches = []
    for init in starting_points:
        # Perform numerical continuation.
        equilibrium = numerical_continuation(func, set_lbda_func, np.array(init), lbda_space)
        nature = [stability_fixedPoint_lbda(x, lbda) for (x, lbda) in zip(equilibrium, lbda_space)]
        branches.append((equilibrium, nature))
    return branches


def computeTrajectory(f, y0, t, args=()):
    # print('===================================== computing trajectories:', y0)
    """
    computes a trajectory on a phase portrait.

    Parameters
    ----------
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    y0 : array_like, shape (2,)
        Initial condition.
    t : array_like
        Time points for trajectory.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.

    Returns
    -------
    output : the integrated trajectory
    """
    def rhs(ab, t):
        # Unpack variables
        return f(ab, t)

    y = scipy.integrate.odeint(rhs, y0, t, args=args)
    return y


# ==========================================================================
# ==========================================================================
# PLOTTING FUNCTIONS
# ==========================================================================
# ==========================================================================
def desperatePlotImplicit(ax, eq, parms, interval, color='magenta', n_grid=300, PLOTTING_THRESHOLD = 1e-5):
    print("     ---> desperate implicit plot, the last resource... ([on -->", eq, "<--)")
    # Set up u,v space
    u = np.linspace(interval['left'], interval['right'], n_grid)
    v = np.linspace(interval['bottom'], interval['top'], n_grid)
    uu, vv = np.meshgrid(u, v)

    # values = np.empty_like(uu)
    # evaluableF = eq(parms, eq, modules=['numpy', 'sympy'])
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            val = eq(np.array([uu[i,j], vv[i,j]]), None)  # values[i,j]
            if np.all(np.abs(val) < PLOTTING_THRESHOLD):
                ax.scatter(uu[i,j], vv[i,j], color=color, marker='.')
    # ax = plotImplicitOnGrid(ax, values, interval, color)
    return ax


# --------------------------------------------------------------------------
# Plots a trajectory on a phase portrait.
# --------------------------------------------------------------------------
def plot_traj(ax, f, y0, t, args=(), color='black', lw=2):
    print('===================================== Plotting trajectories:', y0)
    """
    Plots a trajectory on a phase portrait.

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    y0 : array_like, shape (2,)
        Initial condition.
    t : array_like
        Time points for trajectory.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """
    y = computeTrajectory(f, y0, t, args=args)
    ax.plot(*y.transpose(), color=color, lw=lw)
    return ax


# --------------------------------------------------------------------------
# plots limit cycles
# --------------------------------------------------------------------------
def plotLimitCycles(ax, model, fixedPts, interval, lmbda):
    limits = findLimitCycles(model, fixedPts, interval, n_grid=10)
    for lim in limits:
        ax.scatter(lmbda, model.selectObservationVar(lim[0]))
        ax.scatter(lmbda, model.selectObservationVar(lim[1]))
    return ax


# --------------------------------------------------------------------------
# Plots the separatrix on phase portrait.
# --------------------------------------------------------------------------
def plot_separatrix(ax, f, interval, t_max=300, eps=1e-6,
                    color='tomato', lw=3, shadeUnderCurve = False):
    print('===================================== Plotting Separatrix')
    # Negative time function to integrate to compute separatrix
    def rhs(ab, t):
        # Unpack variables
        a, b = ab

        # Stop integrating if we get the edge of where we want to integrate
        if interval['left'] < a < interval['right'] and interval['bottom'] < b < interval['top']:
            return -1*f(ab, t)
        else:
            return np.array([0, 0])

    """
    Plot separatrix on phase portrait.
    """
    # Compute fixed points
    fps = fixedPoints(f, interval)

    # If only one fixed point, no separatrix
    if len(fps) == 1:
        return ax

    # Parameters for building separatrix
    t = np.linspace(0, t_max, trajectoyLength)

    for fp in fps:
        if classify_fixedPoint(f, fp) == UNSTABLE_FP:
            # Build upper right branch of separatrix
            ab0 = fp + eps
            ab_upper = scipy.integrate.odeint(rhs, ab0, t)

            # Build lower left branch of separatrix
            ab0 = fp - eps
            ab_lower = scipy.integrate.odeint(rhs, ab0, t)

            # Concatenate, reversing lower so points are sequential
            sep_a = np.concatenate((ab_lower[::-1,0], ab_upper[:,0]))
            sep_b = np.concatenate((ab_lower[::-1,1], ab_upper[:,1]))

            # Plot
            ax.plot(sep_a, sep_b, '-', color=color, lw=lw)
            if shadeUnderCurve:
                section = interval['bottom'] * np.ones_like(sep_b)
                plt.fill_between(sep_a, section, sep_b, alpha=.5, facecolor='lightgray')

    return ax


# --------------------------------------------------------------------------
# Plots the flow field with line thickness proportional to speed.
# --------------------------------------------------------------------------
def plot_flow_field(ax, f, interval, args=(), n_grid=100):
    print('===================================== Plotting Flow Field')
    """
    Plots the flow field with line thickness proportional to speed.

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    u_range : array_like, shape (2,)
        Range of values for u-axis.
    v_range : array_like, shape (2,)
        Range of values for v-axis.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """

    # Set up u,v space
    u = np.linspace(interval['left'], interval['right'], n_grid)
    v = np.linspace(interval['bottom'], interval['top'], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i,j], v_vel[i,j] = f(np.array([uu[i,j], vv[i,j]]), None)
    # u_vel, v_vel = f([uu, vv], None)

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    lw = 0.5 + 2.5 * speed / speed.max()

    # Make stream plot
    ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2,
                  density=1, color='thistle')

    return ax

# --------------------------------------------------------------------------
# Plots the Quiverplot with line color proportional to speed.
# --------------------------------------------------------------------------
def plot_quiverplot(ax, f, interval, n_grid=25, normalizeSpeed=True):
    print('===================================== Plotting Quiverplot')
    """
    Plots the Quiverplot with line color proportional to speed.

    Parameters
    ----------
    ax : Matplotlib Axis instance
        Axis on which to make the plot
    f : function for form f(y, t, *args)
        The right-hand-side of the dynamical system.
        Must return a 2-array.
    u_range : array_like, shape (2,)
        Range of values for u-axis.
    v_range : array_like, shape (2,)
        Range of values for v-axis.
    args : tuple, default ()
        Additional arguments to be passed to f
    n_grid : int, default 100
        Number of grid points to use in computing
        derivatives on phase portrait.

    Returns
    -------
    output : Matplotlib Axis instance
        Axis with streamplot included.
    """

    # Set up u,v space
    u = np.linspace(interval['left'], interval['right'], n_grid)
    v = np.linspace(interval['bottom'], interval['top'], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i,j], v_vel[i,j] = f(np.array([uu[i,j], vv[i,j]]), None)
    # u_vel, v_vel = f(np.array([uu, vv]), None)

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)
    speed[speed == 0] = 1.
    if normalizeSpeed:
        u_vel /= speed
        v_vel /= speed
        ax.quiver(uu, vv, u_vel, v_vel, speed, pivot='mid')
    else:
        ax.quiver(uu, vv, u_vel, v_vel, pivot='mid')

    # Make stream plot


    return ax


# --------------------------------------------------------------------------
# Plot nullclines
# --------------------------------------------------------------------------
def plot_nullclines(ax, f, interval,
                    colors=['#1f77b4', '#1fb477'],
                    n_grid = 30):
    print('===================================== Computing nullclines ({}*2 samples)'.format(n_grid))
    """Add nullclines to ax"""
    nc1, mult1, nc2, mult2 = findAllRoots(f, interval, n_grid=n_grid)
    if mult1:
        for point in nc1:
            ax.scatter(point[0], point[1], color=colors[0])
    else:
        xs = [x for x, y in nc1]
        ys = [y for x, y in nc1]
        ax.plot(xs, ys, color=colors[0])
    if mult2:
        for point in nc2:
            ax.scatter(point[0], point[1], color=colors[1])
    else:
        xs = [x for x, y in nc2]
        ys = [y for x, y in nc2]
        ax.plot(xs, ys, color=colors[1])

    return ax


# --------------------------------------------------------------------------
# Plots fixed points
# --------------------------------------------------------------------------
def plot_equilibrium_points(ax, f, interval):
    print('===================================== Plotting Fixed Points')
    """Add fixed points to plot."""
    fps = fixedPoints(f, interval)

    # Plot
    for fp in fps:
        if classify_fixedPoint(f, fp) == UNSTABLE_FP:
            print("Fixed Point:", fp, "is UNSTABLE")
            ax.plot(*fp, '.', markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=2, markersize=20)
        else:
            print("Fixed Point:", fp, "is stable")
            ax.plot(*fp, '.', color='black', markersize=20)
    return ax


# This is an utility function you can use to make your own graph prettier. 
EQUILIBRIUM_COLOR = {'Stable node':'C0',
                    'Unstable node':'C1', 
                    'Saddle':'C4',
                    'Stable focus':'C3',
                    'Unstable focus':'C2',
                    'Center (Hopf)':'C5',
                    'Transcritical (Saddle-Node)':'C6'}
def plot_equilibrium_points2(ax, f, interval, legend=True):
    """Draw equilibrium points at position with the color
       corresponding to their nature
    Adapted from:
        https://www.normalesup.org/~doulcier/teaching/modeling/bistable_systems.html
    """
    fps = fixedPoints(f, interval)
    equilibria_nature = []
    for fp in fps:
        equilibria_nature.append(stability_fixedPoint(f, fp))
        print("{} in ({} {})".format(equilibria_nature[-1], *fp))
    for pos, nat in zip(fps, equilibria_nature):
        ax.scatter(pos[0], pos[1],
                   color= (EQUILIBRIUM_COLOR[nat] 
                           if nat in EQUILIBRIUM_COLOR
                           else 'k'),
                   zorder=100)
    if legend:
        # Draw a legend for the equilibrium types that were used.
        labels = list(frozenset(equilibria_nature))
        ax.legend([mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels], labels)
    return ax


# --------------------------------------------------------------------------
# Plots a Bifurcation Plot
# --------------------------------------------------------------------------
def get_segments(values):
    """Return a dict listing the interval where values are constant.
    Return:
        A dict mapping (start, finish) index to value"""
    start = 0
    segments = {}
    for i,val in enumerate(values[1:],1):
        if val != values[start] or i == len(values)-1:
            segments[(start,i)] = values[start]
            start = i
    return segments


def plot_bifurcation(ax, model, branches, lbdaspace):
    """Function to draw nice bifurcation graph
    Args:
        ax: object of the plt.subplots
        branches: a list of two lists giving the position and
        the nature of the equilibrium.
        lbda_space: bifurcation parameter space
    """
    labels = frozenset()
    for equilibrium, nature in branches:
        labels = labels.union(frozenset(nature))
        segments = get_segments(nature)
        for idx, n in segments.items():
            # xdebug = equilibrium[idx[0]]
            # ydebug = model.getObservationVar(xdebug)[0]
            # xdebug2 = equilibrium[idx[1]]
            # ydebug2 = model.getObservationVar(xdebug2)[0]
            yValues = [model.getObservationVar(x,lbda)[0] for x,lbda in zip(equilibrium[idx[0]:idx[1]+1], lbdaspace[idx[0]:idx[1]+1])]
            ax.plot(lbdaspace[idx[0]:idx[1]+1], yValues,
                     color=EQUILIBRIUM_COLOR[n] if n in EQUILIBRIUM_COLOR else 'k')
    handles = [mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels]
    return ax, handles, labels


# ==========================================================================
# ==========================================================================
# GENERAL PLOTTING FUNCTIONS
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------
# Plotting eq variables
# --------------------------------------------------------------------------
def plotODEInt(f, parms, initialCond):
    print('=====================================')
    print('==        Plotting eq variables    ==')
    print('=====================================')

    # Solve
    #parms, eq = f()
    t = np.linspace(0, 100, trajectoyLength)
    ab = scipy.integrate.odeint(f, initialCond, t)

    # Plot
    plt.plot(t, ab)
    plt.xlabel('t')
    plt.ylabel(parms[0]+','+parms[1])
    plt.legend((parms[0], parms[1]))
    plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Main plotting method, just an utility to quickly plot ODE Phase Planes
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def plotPhasePlane(ax, model, interval,
                   trajectories=[],
                   background='flow',
                   drawNullclines=True,
                   labelLoc='best', eqPoints='b&w',
                   shadeUnderSeparatrix=False):
    print('=====================================')
    print('==        Phase plane analysis     ==')
    print('=====================================')
    # Set up the figure
    # plt.rcParams.update({'font.size': 15})
    # fig, ax = plt.subplots(1, 1)
    # parms, eq = f()
    parms = model.parmNames()
    ax.set_xlabel(parms[0])
    ax.set_ylabel(parms[1])
    custom_lines = [lines.Line2D([0], [0], color='#1f77b4', lw=3),
                    lines.Line2D([0], [0], color='#1fb477', lw=3),
                    lines.Line2D([0], [0], color='tomato', lw=3),
                    lines.Line2D([0], [0], marker='.', linestyle='None', color='black', markersize=20),
                    lines.Line2D([0], [0], marker='.', linestyle='None', markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, markersize=20)]
    ax.legend(custom_lines, [parms[0]+'-nullcline', parms[1]+'-nullcline', 'separatrix', 'stable fp', 'unstable fp'],
              loc=labelLoc)

    # Build the plot
    ax.grid()
    # ax.set_xlim(interval['left'], interval['right'])
    # ax.set_ylim(interval['bottom'], interval['top'])
    if background == 'flow':
        ax = plot_flow_field(ax, model.dfun, interval)
    elif background == 'quiver':
        ax = plot_quiverplot(ax, model.dfun, interval)
    elif background == 'quiver-B&W':
        ax = plot_quiverplot(ax, model.dfun, interval, normalizeSpeed=False)
    if drawNullclines:
        ax = plot_nullclines(ax, model.dfun, interval)
    ax = plot_separatrix(ax, model.dfun, interval, shadeUnderCurve=shadeUnderSeparatrix)
    if eqPoints == 'b&w':
        ax = plot_equilibrium_points(ax, model.dfun, interval)
    else:
        ax = plot_equilibrium_points2(ax, model.dfun, interval)

    # Add some trajectories
    t = np.linspace(0, 100, trajectoyLength)
    for origin in trajectories:
        ax = plot_traj(ax, model.dfun, np.array(origin), t)

    return ax


def plot_PhasePlane_Only(model, interval, trajectories=[], background='flow', drawNullclines=True, labelLoc='best'):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1,1,figsize=(12,5))
    ax.set_aspect('equal')
    ax = plotPhasePlane(ax, model, interval, trajectories=trajectories, background=background, drawNullclines=drawNullclines, labelLoc=labelLoc)
    plt.show()

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Main plotting method, just an utility to quickly plot ODE Phase Planes
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def plotBifurcationDiagram(ax, model, interval, lbda_space, fullEvaluations=10):
    handles = []; labels = []
    real_lbda_space = lbda_space[::fullEvaluations]
    real_lbda_space = np.append(real_lbda_space, lbda_space[-1])
    model.setControlParm(real_lbda_space[0])
    lastFixedPoints = fixedPoints(model.dfun, interval)
    lastFixedPointCount = len(lastFixedPoints)
    print("Start with {} fixed points".format(lastFixedPoints))
    for pos, lbda in enumerate(real_lbda_space[1:], 1):  # Start from the second element, as the first one is already done!
        model.setControlParm(lbda)
        fps = fixedPoints(model.dfun, interval)
        # Perhaps this is a bit redundant (i.e., computing the fixed points and then the limit cycles), but I
        # think it is not, as a trajectory may converge to a STABLE fixed point, to a limit cycle or simply diverge.
        plotLimitCycles(ax, model, fps, interval, lbda)
        reduced_lbda_space = lbda_space[(pos-1)*fullEvaluations:pos*fullEvaluations+1]
        if len(fps) > lastFixedPointCount:
            print("Change at {} to {}".format(lbda, len(fps)))
            reduced_lbda_space = reduced_lbda_space[::-1]
            fixedPointsToUse = fps
        else:
            fixedPointsToUse = lastFixedPoints
        # OK, let's compute the branches starting from each fixed point...
        branches = get_branches(model.dfun, model.setControlParm, fixedPointsToUse, reduced_lbda_space)
        ax, handleslbda, labelslbda = plot_bifurcation(ax, model, branches, reduced_lbda_space)
        # Now, let's CAREFULLY merge all the labels and handles...
        handles = handles + [i for (i,j) in zip(handleslbda, labelslbda) if j not in labels]
        labels = labels + [j for (i,j) in zip(handleslbda, labelslbda) if j not in labels]
        lastFixedPoints = fps
        lastFixedPointCount = len(fps)
    ax.legend(handles, labels)
    ax.set(xlabel=model.getControlParmName(), ylabel=model.getObservationVarName())
    return ax


def plot_BifurcationDiagram_Only(model, interval, lbda_space, fullBifurcationEvaluations=10):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1,1,figsize=(12,5))
    ax = plotBifurcationDiagram(ax, model, interval, lbda_space, fullEvaluations=fullBifurcationEvaluations)
    plt.show()


def plotFancyBifurcationDiagram(model, interval, lbda_space,
                                trajectories=[], background='flow', drawNullclines=True, fullBifurcationEvaluations=10,
                                phaseLabelLoc='best'):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(constrained_layout=True)
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    ax1 = fig.add_subplot(grid[0,0])
    model.setControlParm(lbda_space[0])
    ax1 = plotPhasePlane(ax1, model, interval, trajectories=trajectories, background=background, drawNullclines=drawNullclines, labelLoc=phaseLabelLoc)

    ax2 = fig.add_subplot(grid[0,1])
    model.setControlParm(lbda_space[int(len(lbda_space)/2)])
    ax2 = plotPhasePlane(ax2, model, interval, trajectories=trajectories, background=background, drawNullclines=drawNullclines, labelLoc=phaseLabelLoc)

    ax3 = fig.add_subplot(grid[0,2])
    model.setControlParm(lbda_space[-1])
    ax3 = plotPhasePlane(ax3, model, interval, trajectories=trajectories, background=background, drawNullclines=drawNullclines, labelLoc=phaseLabelLoc)

    ax4 = fig.add_subplot(grid[1, :])
    ax4 = plotBifurcationDiagram(ax4, model, interval, lbda_space,
                                 fullEvaluations=fullBifurcationEvaluations)
    ax4.axvline(x=lbda_space[0], color='k', linestyle='--')
    ax4.axvline(x=lbda_space[int(len(lbda_space)/2)], color='k', linestyle='--')
    ax4.axvline(x=lbda_space[-1], color='k', linestyle='--')

    plt.show()

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------EOF
