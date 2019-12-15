# ==========================================================================
# ==========================================================================
# Set of functions to analyse and plot a system of ODEs with symbolic tools.
#
# Some code taken from Some dynamical systems approaches, by Justin Bois, 2017
# http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
from numpy import linalg as LA
import scipy.integrate
import scipy.optimize
import sympy as sm
from sympy import ccode

# Plotting modules
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import sympy.plotting as splt

# ==========================================================================
# ==========================================================================
# EVALUATION FUNCTIONS
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------
# Symbolic evaluation
# --------------------------------------------------------------------------
def evaluableFunc(func):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()
    return sm.lambdify(parms, eqs, "numpy")  #([a, b], [eq1, eq2], "numpy")


def evalFunc(func, ab, t):
    f = evaluableFunc(func)
    res = f(*ab)  #f(ab[0], ab[1])
    return np.array(res)


# --------------------------------------------------------------------------
# Compute Fixed Points
# --------------------------------------------------------------------------
def fixedPoints(func):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()

    # use sympy's way of setting equations to zero
    aEqual = sm.Eq(eqs[0], 0)
    bEqual = sm.Eq(eqs[1], 0)
    print("Computing Fixed Points:")
    print("      Eq. "+ccode(parms[0])+":", aEqual)
    print("      Eq. "+ccode(parms[1])+":", bEqual)

    # compute fixed points
    try:
        equilibria = sm.solve((aEqual, bEqual), *parms, rational=False)  # We need rational=False because it hangs otherwise. But, if it fails, exits with an error...
        print(' ===> Computed symbolically compute Fixed Points!!!')
        # Now, convert to an array of numpy.array, filtering out the complex solutions...
        realFP = [fp for fp in equilibria if fp[0].is_real]
        return [np.array([fp[0].evalf(), fp[1].evalf()], dtype=np.float64) for fp in realFP]
    except:
        print(' ===> Failed to symbolically compute Fixed Points... Please, try numerical methods!')
        return []

# --------------------------------------------------------------------------
# Compute Nullclines
# --------------------------------------------------------------------------
def solve_nullcline(eq, parms):
    # use SymPy's way of setting equations to zero. not really needed, but more elegant! ;-)
    equal = sm.Eq(eq, 0)
    print("Solving Nullcline:", equal, flush=True)
    nullclines = sm.solve(equal, parms, set=True, dict=True, rational=False)  # We have to add because it hangs otherwise... But, if it fails, exits with an error!!!
    for nc in nullclines:
        print("           result:", nc, flush=True)
    return nullclines


# --------------------------------------------------------------------------
# Compute the Jacobian
# --------------------------------------------------------------------------
def Jac(func, ab, t):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()
    J = sm.Matrix(eqs).jacobian(parms)
    J_func = sm.lambdify(parms, J, "numpy")
    res = J_func(ab[0], ab[1])
    return res


# --------------------------------------------------------------------------
# Classifies the Fixed Points
# --------------------------------------------------------------------------
UNSTABLE_FP = 0
STABLE_FP = 1
def classify_fixedPoint(f, fp):
    J = Jac(f, fp, None)
    evalues, evectors = LA.eig(J)
    if np.any(evalues.real > 0):
        # Here we only care about stable vs. unstable fixed points
        # However, if one evalue > 0 and the other < 0, we have a saddle fp...
        #          if both evalues > 0, we have a source
        return UNSTABLE_FP
    else:  # Both evalues < 0 => a sink!
        return STABLE_FP


# ==========================================================================
# ==========================================================================
# PLOTTING FUNCTIONS
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------
# Plot implicit, when we cannot do better...
# --------------------------------------------------------------------------
def plotImplicitOnGrid(ax, points, interval, color='magenta'):
    from skimage import measure
    # Using skimage.measure...
    dimX, dimY = points.shape
    intervalX = interval['right'] - interval['left']
    intervalY = interval['top'] - interval['bottom']
    contours = measure.find_contours(points, 0)
    for n, contour in enumerate(contours):
       ax.plot(interval['left']   + contour[:, 1]*intervalX/dimX,
               interval['bottom'] + contour[:, 0]*intervalY/dimY,
               linewidth=2, color=color)
    return ax


def plotImplicit(ax, eq, parms, interval, color='magenta'):
    # To plot, one option is to lambdify nc[var] and then, if its NOT a number, evaluate with the array
    # of values, and then append the result to the list, being careful to add taking into account if we
    # solved for x or y. The problem is that this produces problems when plotting implicit function like
    # a circle (too steep at the intersections between the circle and the plotting axis). The same happens
    # if we provide values that evaluate to a complex number (e.g., again, outside of the circle).
    # However, SymPy has a plotting library that we can use... ;-)
    # Thanks stackoverflow:
    # https://stackoverflow.com/questions/31747210/python-sympy-implicit-function-get-values-instead-of-plot
    plotf = splt.plot_implicit(eq,
                               (parms[0], interval['left'], interval['right']),
                               (parms[1], interval['bottom'], interval['top']), show=False,
                               # parameters for when the method falls back to a contour plot... and wait!!!
                               depth=4, points=300)
    series = plotf[0]
    points = series.get_points()
    if points[-1] == 'fill':
        data, action = points
        data = np.array([(x_int.mid, y_int.mid) for x_int, y_int in data])
        ax.scatter(data.T[0], data.T[1], color=color, marker='.')
    elif points[-1] == 'contour':
        #print("points:", points[2])
        #ax.pcolormesh(points[0], points[1], points[2], alpha=0.33)

        # ax.plotImplicitOnGrid(ax, points[2], interval, color)

        ax.contourf(points[0], points[1], points[2], alpha=0.33)
    else:
        print("Wrong option @ plotImplicit:", points[-1])
    return ax


def tryPlotImplicit(ax, f, var, parms, interval, color='magenta'):
    import warnings
    def rhs(ab, t):
        eq = sm.lambdify(parms, f[var]-var, modules=['sympy', 'numpy'])
        return eq(*ab)

    # I hate this conversion of warnings into errors, but plot_implicit is a marvelous tool that does pretty bad when
    # the implicit curve has self intersections. Then, the adaptive meshing cannot be applied to the expression, and
    # then it resorts to a simple contour algorithm, which obviously do pretty horribly...
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            print("     ---> trying to plot implicit:", f)
            equality = sm.Eq(f[var], var)
            return plotImplicit(ax, equality, parms, interval, color)
        except Warning as e:
            print('     ---> error (warning) found:', e, "for", f)
            warnings.filterwarnings("default")
            warnings.warn(' ===> It seems impossible to plot Implicit... Please, try numerical methods!\nSwitching to a lame (but fast) numerical method')
            import functions.NumAnalysis as numA
            print(rhs(np.array([0,0]), None))
            numA.desperatePlotImplicit(ax, rhs, parms, interval, color=color, PLOTTING_THRESHOLD = 1e-3)
    return ax


def tryPlotExplicit(ax, eq, var, parms, interval, color='magenta', n_grid=300):
    if var == parms[1]:  # if var is Y (second var)
        low = interval['left']; high = interval['right']  # If it is Y, iterate over X (first var)
    else:
        low = interval['bottom']; high = interval['top']  # otherwise
    tSpace = np.linspace(low, high, n_grid)
    res = []
    for t in tSpace:
        f = sm.lambdify(parms, eq, modules=['sympy', "numpy"])
        if var == parms[1]:
            res.append(f(t,0))
        else:
            res.append(f(0,t))
    if var == parms[1]:  # if var is Y (second var)
        ax.plot(tSpace, res, color=color)
    else:
        ax.plot(res, tSpace, color=color)
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
            u_vel[i,j], v_vel[i,j] = evalFunc(f, np.array([uu[i,j], vv[i,j]]), None)

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
def plot_quiverplot(ax, f, interval, n_grid=20):
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
            u_vel[i,j], v_vel[i,j] = evalFunc(f, np.array([uu[i,j], vv[i,j]]), None)

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)
    speed[speed == 0] = 1.
    u_vel /= speed
    v_vel /= speed

    # Make stream plot
    ax.quiver(uu, vv, u_vel, v_vel, speed, pivot='mid')

    return ax


# --------------------------------------------------------------------------
# Plot nullclines
# --------------------------------------------------------------------------
def plot_nullclines(ax, f, interval,
                    colors=['#1f77b4', '#1fb477'],
                    type="explicit"):
    print('===================================== Computing nullclines ('+type+' if possible)')
    """Add nullclines to ax."""
    parms, eqs = f()
    for i, eq in enumerate(eqs):
        if type.find('solve') != -1:
            print(' ===> Trying to solve')
            try:
                ncs = solve_nullcline(eq, parms)
                print(' ===> Solve succeeded, trying to plot analytic nullclines')
            except:
                ncs = [{0:eq}]
                print(' ===> Solve failed, directly plotting the whole nullcline, implicitly')
        else:
            ncs = [{0:eq}]
        # Now, I am sure we have a list of {var = eq} dicts, for all cases...
        for nc in ncs:
            for var in nc:
                if type.find('implicit') != -1:
                    ax = tryPlotImplicit(ax, nc, var, parms, interval, color=colors[i])
                else:
                    ax = tryPlotExplicit(ax, nc[var], var, parms, interval, color=colors[i])
                    # Here we should also plot x/y = var (a line) in case the (explicit) equation was of
                    # type {var = eq}, so we can SEE the intersections...
    return ax


# --------------------------------------------------------------------------
# Plots fixed points
# --------------------------------------------------------------------------
def plot_fixed_points(ax, f):
    print('===================================== Plotting Fixed Points')
    """Add fixed points to plot."""
    fps = fixedPoints(f)

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
    def rhs(ab, t):
        # Unpack variables
        return evalFunc(f, ab, t)

    y = scipy.integrate.odeint(rhs, y0, t, args=args)
    ax.plot(*y.transpose(), color=color, lw=lw)
    return ax


# --------------------------------------------------------------------------
# Plots the separatrix on phase portrait.
# --------------------------------------------------------------------------
def plot_separatrix(ax, f, interval, t_max=30, eps=1e-6,
                           color='tomato', lw=3):
    print('===================================== Plotting Separatrix')
    # Negative time function to integrate to compute separatrix
    def rhs(ab, t):
        # Unpack variables
        a, b = ab

        # Stop integrating if we get the edge of where we want to integrate
        if interval['left'] < a < interval['right'] and interval['bottom'] < b < interval['top']:
            return -1*evalFunc(f, ab, t)
        else:
            return np.array([0, 0])

    """
    Plot separatrix on phase portrait.
    """
    # Compute fixed points
    fps = fixedPoints(f)

    # If only one fixed point, no separatrix
    if len(fps) == 1:
        return ax

    # Parameters for building separatrix
    t = np.linspace(0, t_max, 400)

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

    return ax


# --------------------------------------------------------------------------
# Plotting eq variables
# --------------------------------------------------------------------------
def plotODEInt(f, initialCond):
    print('=====================================')
    print('==        Plotting eq variables    ==')
    print('=====================================')

    def rhs(ab, t):
        # Unpack variables
        return evalFunc(f, ab, t)

    # Solve
    parms, eqs = f()
    t = np.linspace(0, 100, 600)
    ab0 = np.array(initialCond)
    ab = scipy.integrate.odeint(rhs, ab0, t)

    # Plot
    plt.plot(t, ab)
    plt.xlabel('t')
    plt.ylabel(ccode(parms[0])+','+ccode(parms[1]))
    plt.legend((parms[0], parms[1]))
    plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Main plotting method, just an utility to quickly plot ODEs
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def plotPhasePlane(f, interval, trajectories=[], background='flow', type="explicit"):
    print('=====================================')
    print('==        Phase plane analysis     ==')
    print('=====================================')
    # Set up the figure
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1, 1)
    parms, eqs = f()
    ax.set_xlabel(parms[0])
    ax.set_ylabel(parms[1])
    ax.set_aspect('equal')
    custom_lines = [lines.Line2D([0], [0], color='#1f77b4', lw=3),
                    lines.Line2D([0], [0], color='#1fb477', lw=3),
                    lines.Line2D([0], [0], color='tomato', lw=3),
                    lines.Line2D([0], [0], marker='.', linestyle='None', color='black', markersize=20),
                    lines.Line2D([0], [0], marker='.', linestyle='None', markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, markersize=20)]
    ax.legend(custom_lines, [ccode(parms[0])+'-nullcline', ccode(parms[1])+'-nullcline', 'separatrix', 'stable fp', 'unstable fp'])

    # Build the plot
    ax.grid()
    ax.set_xlim(interval['left'], interval['right'])
    ax.set_ylim(interval['bottom'], interval['top'])
    if background == 'flow':
        ax = plot_flow_field(ax, f, interval)
    elif background == 'quiver':
        ax = plot_quiverplot(ax, f, interval)
    ax = plot_nullclines(ax, f, interval, type=type)
    ax = plot_separatrix(ax, f, interval)
    ax = plot_fixed_points(ax, f)

    # Add some trajectories
    t = np.linspace(0, 100, 400)
    for origin in trajectories:
        ax = plot_traj(ax, f, np.array(origin), t)

    plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------EOF
