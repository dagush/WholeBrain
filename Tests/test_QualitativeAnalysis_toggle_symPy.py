# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of the toggle system
# Taken from Some dynamical systems approaches, by Justin Bois, 2017
# http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html
#
# Translated to SymPy by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
from numpy import linalg as LA
import scipy.integrate
import scipy.optimize
import sympy as sm

# Plotting modules
import matplotlib.pyplot as plt

# ==========================================================================
# ODE: Bistable gene circuit (genetic toggle)
# from
# Biomolecular Feedback Systems, chapter 3
# Domitilla Del Vecchio and Richard M. Murray
# http://www.cds.caltech.edu/~murray/BFSwiki/index.php/Main_Page
# ==========================================================================
gamma = 2
beta = 5
n = 2

# gamma = 1
# beta = 5
# n = 2

def toggle():
    a, b = sm.symbols('a, b')  # negative=False)
    # beta, gamma, n = sm.symbols('beta, gamma, n', real=True, constant=True)
    eq1 = beta / (1 + b**n) - a
    eq2 = gamma * (beta / (1 + a**n) - b)
    return [a,b], [eq1,eq2]


# ==========================================================================
# Symbolic evaluation
# ==========================================================================
def evaluableFunc(func):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()
    return sm.lambdify(parms, eqs, "numpy")  #([a, b], [eq1, eq2], "numpy")


def evalFunc(func, ab, t):
    f = evaluableFunc(func)
    res = f(*ab)  #f(ab[0], ab[1])
    return np.array(res)


def fixedPoints(func):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()

    # use sympy's way of setting equations to zero
    aEqual = sm.Eq(eqs[0], 0)
    bEqual = sm.Eq(eqs[1], 0)

    # compute fixed points
    equilibria = sm.solve((aEqual, bEqual), *parms) #a, b)
    # Now, convert to an array of numpy.array, filtering out the complex solutions...
    realFP = [fp for fp in equilibria if fp[0].is_real]
    return [np.array([fp[0].evalf(), fp[1].evalf()], dtype=np.float64) for fp in realFP]


def nullclines(func, ab, t):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()

    # use sympy's way of setting equations to zero
    aEqual = sm.Eq(eqs[0], 0)
    bEqual = sm.Eq(eqs[1], 0)

    # print("Parms: beta={}, gamma={}, n={}".format(beta, gamma, n))
    a_nullcline = sm.solve(aEqual, parms[0])
    # print("Solve:", aEqual, "==> a=", a_nullcline)
    b_nullcline = sm.solve(bEqual, parms[1])
    # print("Solve:", bEqual, "==> b=", b_nullcline)
    f = sm.lambdify(parms, [a_nullcline, b_nullcline], "numpy")
    res = f(ab[0], ab[1])
    return np.array([res[0][0], res[1][0]])


def Jac(func, ab, t):
    parms, eqs = func()  # [a,b], [eq1,eq2] = func()
    J = sm.Matrix(eqs).jacobian(parms)
    J_func = sm.lambdify(parms, J, "numpy")
    res = J_func(ab[0], ab[1])
    return res


UNSTABLE_FP = 0
STABLE_FP = 1
def classify_fixedPoint(f, fp):
    J = Jac(f, fp, None)
    evalues, evectors = LA.eig(J)
    if np.any(evalues.real > 0):
        return UNSTABLE_FP
    else:
        return STABLE_FP

# ==========================================================================
# Many trajectories and streamplots
# ==========================================================================
def plot_flow_field(ax, f, u_range, v_range, args=(), n_grid=100):
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
    u = np.linspace(u_range[0], u_range[1], n_grid)
    v = np.linspace(v_range[0], v_range[1], n_grid)
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


def plot_quiverplot(ax, f, u_range, v_range, args=(), n_grid=20):
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
    u = np.linspace(u_range[0], u_range[1], n_grid)
    v = np.linspace(v_range[0], v_range[1], n_grid)
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

# ==========================================================================
# Many trajectories and streamplots
# ==========================================================================
def plot_nullclines(ax, f, a_range, b_range,
                    colors=['#1f77b4', '#1f77b4'], lw=3):
    """Add nullclines to ax."""

    b = np.linspace(b_range[0], b_range[1], 200)  # a-nullcline
    a = np.linspace(a_range[0], a_range[1], 200)  # b-nullcline
    [nca, ncb] = nullclines(f, np.array([a, b]), None)

    # Plot
    ax.plot(nca, b, lw=lw, color=colors[0])
    ax.plot(a, ncb, lw=lw, color=colors[1])

    return ax


# ==========================================================================
# Fixed points
# ==========================================================================
def plot_fixed_points(ax, f):
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


# ==========================================================================
# Putting it together: streamlines with nullclines and fixed points
# ==========================================================================
def plot_traj(ax, f, y0, t, args=(), color='black', lw=2):
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


# ==========================================================================
# The separatrix
# ==========================================================================
def plot_separatrix(ax, f, a_range, b_range, t_max=30, eps=1e-6,
                           color='tomato', lw=3):
    # Negative time function to integrate to compute separatrix
    def rhs(ab, t):
        # Unpack variables
        a, b = ab

        # Stop integrating if we get the edge of where we want to integrate
        if a_range[0] < a < a_range[1] and b_range[0] < b < b_range[1]:
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


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Set up the figure
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_aspect('equal')
ax.grid()

# Build the plot
a_range = [0, 6]
b_range = [0, 6]
ax = plot_flow_field(ax, toggle, a_range, b_range)
# ax = plot_quiverplot(ax, toggle, a_range, b_range)
ax = plot_nullclines(ax, toggle, a_range, b_range)
ax = plot_separatrix(ax, toggle, a_range, b_range)
ax = plot_fixed_points(ax, toggle)

# t = np.linspace(0, 30, 200)
# ax = plot_traj(ax, toggle, np.array([0.01, 1]), t)
# ax = plot_traj(ax, toggle, np.array([1, 0.01]), t)
# ax = plot_traj(ax, toggle, np.array([3, 6]), t)
# ax = plot_traj(ax, toggle, np.array([6, 3]), t)

plt.show()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
