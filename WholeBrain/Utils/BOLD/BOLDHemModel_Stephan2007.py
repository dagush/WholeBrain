# ----------------------------------------
# ----------------------------------------
# BOLD model parameters. In general, these equations are from:
#
# * K.J. Friston, L. Harrison, and W. Penny,
#   Dynamic causal modelling, NeuroImage 19 (2003) 1273–1302
# * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
#   Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
#
# ----------------------------------------
# ----------------------------------------
import numpy as np
from numba import jit

print("Going to use Stephan2007 BOLD model...")

t_min = 20  # (s)
dt = 0.001  # (s)
n_min = int(np.round(t_min / dt))


def computeRequiredVectorLength(T):
    global t_min, dt, n_min
    n_min = int(np.round(t_min / dt))
    n_t = int(T/dt)
    return n_t - n_min


@jit(nopython=True)
def BOLDModel(T, r):
    # The Hemodynamic model with one simplified neural activity
    #     by Stephan et al. 2007
    #
    # T          : total time (s)
    # resolution : dt in (s)
    #
    # Code from Deco et al. 2018
    #
    # Comparing hemodynamic models with DCM
    #
    # Based on the paper:
    # Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
    # NeuroImage 38 (2007) 387–401

    # global itaus, itauf, itauo, ialpha, Eo, dt

    # BOLD model parameters. In general, these values are from:
    # Dynamic causal modelling,
    # K.J. Friston, L. Harrison, and W. Penny,
    # NeuroImage 19 (2003) 1273–1302
    # ----------------------------------------

    # global t_min, dt, n_min
    taus = 0.65  # 0.8;    # time unit (s)  --> kappa in the paper
    tauf = 0.41  # 0.4;    # time unit (s)  --> gamma in the paper
    tauo = 0.98  # 1;      # mean transit time (s)  --> tau in the paper
    alpha = 0.32 # 0.2;    # a stiffness exponent   --> alpha in the paper
    itaus = 1. / taus
    itauf = 1. / tauf
    itauo = 1. / tauo
    ialpha = 1. / alpha

    Eo = 0.4  # This value is from Obata et al. (2004)
    TE = 0.04  # --> TE, from Stephan et al. 2007
    vo = 0.04  # ???
    r0 = 25  # (s)^-1 --> r0, from Stephan et al. 2007
    theta0 = 40.3  # (s)^-1
    # Part of equation (12) in Stephan et al. 2007:
    k1 = 4.3*theta0*Eo*TE
    k2 = r0*Eo*TE  # Shouldn't it be epsilon*r0*Eo*TE ???
    k3 = 1  # Shouldn't it be 1-epsilon ???

    # dt = resolution    # (s)
    #t0 = np.arange(0,T,dt).reshape(-1,1)
    #n_t = t0.size
    n_t = int(T/dt)

    n_min = int(np.round(t_min / dt))

    # Initial conditions
    x0 = np.array([0, 1, 1, 1])

    # Euler method
    # t = t0
    x = np.zeros((n_t, 4))
    x[0,:] = x0
    for n in range(n_t-1):
        # Equation (9) for s in Stephan et al. 2007
        x[n + 1, 0] = x[n, 0] + dt * (r[n] - itaus * x[n, 0] - itauf * (x[n, 1] - 1))  # Shouldn't it be (0.5 r[n] + 3) instead of r[n] ??? also, shouldn't it be taus and tauf instead of itaus and itauf???
        # Equation (10) for f in Stephan et al. 2007
        x[n + 1, 1] = x[n, 1] + dt * x[n, 0]
        # Equation (8) for v and q in Stephan et al. 2007
        x[n + 1, 2] = x[n, 2] + dt * itauo * (x[n, 1] - x[n, 2] ** ialpha)
        x[n + 1, 3] = x[n, 3] + dt * itauo * (x[n, 1] * (1-(1 - Eo)**(1/x[n, 1]))/Eo - (x[n, 2]**ialpha) * x[n, 3]/x[n, 2])

    # The Balloon-Windkessel model, originally from Buxton et al. 1998:
    # t = t[n_min:t.size]
    # s = x[n_min:n_t, 0]
    # fi = x[n_min:n_t, 1]
    v = x[n_min:n_t, 2]
    q = x[n_min:n_t, 3]
    b = vo * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))  # Equation (12) in Stephan et al. 2007

    # Code to check whether we have a nan in the arrays...
    # Probably, it comes from a negative value in x[:,1], x[:,2] or x[:,3].
    # If it happens, then directly us the [Stephan2008] implementation:
    # * Klaas Enno Stephan, Lars Kasper, Lee M. Harrison, Jean Daunizeau, Hanneke E.M. den Ouden, Michael Breakspear, and Karl J. Friston
    #   Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2008 Aug 15; 42(2): 649–662.
    #
    # array_sum = np.sum(b)
    # array_has_nan = np.isnan(array_sum)
    # if array_has_nan:
    #     print(f"NAN!!!")

    return b
