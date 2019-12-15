# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of the toggle system
# Taken from Some dynamical systems approaches, by Justin Bois, 2017
# http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.integrate
import scipy.optimize

# Plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
# sns.set(style='whitegrid', context='notebook', palette=colors,
#         rc={'axes.labelsize': 16})
plt.grid()

#%matplotlib inline
#%config InlineBackend.figure_formats = {'png', 'retina'}

# ==========================================================================
# The phase portrait for the activator-repressor clock
# ==========================================================================
def act_rep_clock(ab, t, alpha, beta, gamma, n):
    """Right hand side of ODEs for activator-repressor clock."""
    a, b = ab
    return np.array([alpha + beta * b**n / (1 + b**n) - a,
                     gamma * (alpha + beta * b**n / (1 + a**n + b**n) - b)])


def b_nullcline(a_vals, b_range):
    """Find b-nullcline for values of a."""
    # Set up output array
    b_nc = np.empty((len(a_vals), 3))
    b = np.linspace(b_range[0], b_range[1], 10000)

    # For each value of a, find where rhs of ODE is zero
    for i, a in enumerate(a_vals):
        s = np.sign(alpha + beta * b**n / (1 + a**n + b**n) - b)

        # Values of b for sing switches
        b_vals = b[np.where(np.diff(s))]

        # Make sure we put numbers in correct branch
        if len(b_vals) == 0:
            b_nc[i,:] = np.array([np.nan, np.nan, np.nan])
        elif len(b_vals) == 1:
            if b_vals[0] > 2*alpha:
                b_nc[i,:] = np.array([np.nan, np.nan, b_vals[0]])
            else:
                b_nc[i,:] = np.array([b_vals[0], np.nan, np.nan])
        elif len(b_vals) == 2:
            b_nc[i,:] = np.array([b_vals[0], b_vals[1], np.nan])
        else:
            b_nc[i,:] = b_vals

    return b_nc


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
            u_vel[i,j], v_vel[i,j] = f(np.array([uu[i,j], vv[i,j]]), None, *args)

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    lw = 0.5 + 2.5 * speed / speed.max()

    # Make stream plot
    ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2,
                  density=1, color='thistle')

    return ax


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

    y = scipy.integrate.odeint(f, y0, t, args=args)
    ax.plot(*y.transpose(), color=color, lw=lw)
    return ax


# We can now use it to make our nullclines.
def plot_null_clines_act_rep_clock(ax, a_range, b_range, alpha, beta, gamma, n,
                                   colors=['#1f77b4', '#1f77b4'], lw=3):
    """Add nullclines to ax."""
    # a-nullcline
    nca_b = np.linspace(b_range[0], b_range[1], 200)
    nca_a = alpha + beta * nca_b**n / (1 + nca_b**n)

    # b-nullcline
    ncb_a = np.linspace(a_range[0], a_range[1], 20000)
    ncb_b = b_nullcline(ncb_a, b_range)

    # Plot
    ax.plot(nca_a, nca_b, lw=lw, color=colors[0])
    ax.plot(ncb_a, ncb_b, lw=lw, color=colors[1])

    return ax


# ==========================================================================
# ==========================================================================
beta = 10
alpha = 0.1
gamma = 5
n = 2
args = (alpha, beta, gamma, n)

# Solve
t = np.linspace(0, 30, 200)
ab0 = np.array([1.2, 0.5])
ab = scipy.integrate.odeint(act_rep_clock, ab0, t, args=args)

# Plot
plt.plot(t, ab)
plt.xlabel('t')
plt.ylabel('a, b')
plt.legend(('a', 'b'))
plt.show()

# ==========================================================================
# Nullclines
# ==========================================================================
a = 4.5
b = np.linspace(0, 10, 200)
plt.plot(b, b)
plt.plot(b, alpha + beta * b**n / (1 + a**n + b**n), color=colors[0])
plt.xlabel('b')
plt.show()

# Now, we can make out plot. We will put a few trajectories on to highlight the limit cycle.
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_aspect('equal')

t = np.linspace(0, 15, 400)
ax = plot_flow_field(ax, act_rep_clock, [0, 10], [0, 10], args=args)
ax = plot_null_clines_act_rep_clock(ax, [0, 10], [0, 10], alpha, beta, gamma, n)
ax = plot_traj(ax, act_rep_clock, np.array([0.01, 1]), t, args=args)
ax = plot_traj(ax, act_rep_clock, np.array([0.1, 10]), t, args=args)
ax = plot_traj(ax, act_rep_clock, np.array([1, 0.1]), t, args=args)
ax = plot_traj(ax, act_rep_clock, np.array([10, 10]), t, args=args)
plt.show()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
