# ==========================================================================
# ==========================================================================
# Simple implementation of
# [Meisl21] G. Meisl, E. Hidari, K. Allinson, T. Rittman, S. L. DeVos, J. S. Sanchez, C. K. Xu, K. E. Duff,
# K. A. Johnson, J. B. Rowe, B. T. Hyman, T. P. J. Knowles, D. Klenerman, In vivo rate-determining
# steps of tau seed accumulation in Alzheimer’s disease. Sci. Adv. 7, eabh1448 (2021).
#
# by Gustavo Patow
#
# ==========================================================================
# ==========================================================================

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm


D = None
kappa = None
dx = None
def dfun(f):
    df = D * ndimage.laplace(f) / dx**2 + kappa * f * (1-f)
    return df


dt = None
def eqIintegrate(u, nt):
    # Loop
    for t in range(0, nt - 1):
        print('.', end='')
        u[t+1, :] = u[t, :] + dfun(u[t,:]) * dt
        print(end='')
    print()
    return u


def diffusion(nt, nx, tmax):
    # Initialisation of data structures
    u = np.zeros((nt, nx))

    # Boundary Conditions
    u[:, 0] = 1.
    u[:, nx - 1] = 0.
    # Initial Conditions
    for i in range(0, nx-1):
        if i < nx/10:
            u[0,i] = 1.
        else:
            u[0,i] = 0.

    # And... integrate!!!
    u = eqIintegrate(u, nt)

    return u


def plot(u, x, time, nt, nx, tmax, xmax):
    """
    Plots the Velocity Field Results in 1D
    """
    factor = 5
    plt.figure()

    ax = plt.subplot(1,2,1)
    colour = iter(cm.rainbow(np.linspace(0, nt/factor, nt)))
    for t in range(0, nt, int(nt/factor)):
        c = next(colour)
        ax.plot(x, u[t, :], c=c, label=f'{np.int(t/nt*tmax)}')
        ax.set_xlabel('Relative distance')
        ax.set_ylabel('Tau')
        # ax.set_ylim([0, 1.2])
        ax.grid(True, linestyle='-.', linewidth='0.5', color='black')
        ax.legend()
        ax.set_title('2B,S4) Tau @ different years')

    ax = plt.subplot(1,2,2)
    colour = iter(cm.rainbow(np.linspace(0, nx/factor, nx)))
    for pos in range(int(nx / factor), nx, int(nx / factor)):
        c = next(colour)
        ax.plot(time, u[:, pos], c=c, label=f'{np.round(pos/nx*xmax,2)}')
        ax.set_xlabel('Years since onset')
        ax.set_ylabel('Tau')
        # ax.set_ylim([0, 1.2])
        ax.grid(True, linestyle='-.', linewidth='0.5', color='black')
        ax.legend()
        ax.set_title('3A-E) Tau @ positions from origin')

    # plt.title(f'Figure 1: D={D}, k={kappa}, nt={nt}, nx={nx}, tmax={tmax}y')
    plt.show()


def integrate():
    global kappa; kappa = 0.14  # years^−1
    global D; D = 0.0025 * kappa

    xmin = 0.
    xmax = 1.0
    tmax = 70
    nx = 51
    nt = 151
    global dx; dx = (xmax-xmin) / (nx - 1)
    global dt; dt = tmax / (nt - 1)
    u = diffusion(nt, nx, tmax)
    # x, only for plotting...
    x = np.arange(xmin, xmin+nx*dx, dx)
    t = np.arange(0, tmax+dt, dt)
    plot(u, x, t, nt, nx, tmax, xmax)


# ======================================================================
# ======================================================================
# ======================================================================
if __name__ == '__main__':
    # Set the size of the networks to be simulated
    integrate()

# ======================================================================
# ======================================================================
# ======================================================================EOF