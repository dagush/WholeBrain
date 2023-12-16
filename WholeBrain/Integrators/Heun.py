# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of the Heun Integrator
# Based on the code from TVB:
#     It is a simple example of a predictor-corrector method. It is also known as
#     modified trapezoidal method, which uses the Euler method as its predictor.
#     And it is also a implicit integration scheme.
#
#     [1] Kloeden and Platen, Springer 1995, *Numerical solution of stochastic
#         differential equations.
#
#     From [1]:
#         .. math::
#             X_{n+1} &= X_n + dt (dX(t_n, X_n) +
#                                  dX(t_{n+1}, \tilde{X}_{n+1})) / 2 \\
#             \tilde{X}_{n+1} &= X_n + dt dX(t_n, X_n)
#         cf. Equation 1.11, page 283.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
# from WholeBrain.Utils import numTricks as iC
from WholeBrain.Integrators.integr_utils import *
from numba import jit

print("Going to use the Heun integration scheme...")

neuronalModel = None  # To be able to choose the model externally...


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Heun Integration
# --------------------------------------------------------------------------
# sigma = 0.01
@jit(nopython=True)
def integrationStep(simVars, dt, coupling, stimulus):
    dvars_obsVars = neuronalModel.dfun(simVars, coupling, stimulus)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...

    inter = simVars + dt * dvars

    dvars_obsVars = neuronalModel.dfun(inter, coupling, stimulus)
    dvars2 = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    dX = (dvars + dvars2) * dt / 2.0

    simVars = simVars + dX

    return simVars, obsVars


# ======================================================================
# Debug/test code
# To use it, comment the @jit(nopython=True) line at integrationStep
# Otherwise you'll get weird numba errors
# ======================================================================
if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    import sys
    mySelf = sys.modules[__name__]

    import Integrator
    Integrator.integrationScheme = mySelf

    class dummyNeuronalModel:
        def __init__(self):
            pass
        # we will use the differential equation y'(t) = y(t).
        def dfun(self, simVars, stimulus):
            y = simVars
            return y, y
        def recompileSignatures(self):
            pass
        def numObsVars(self):
            return 1

    # For this to work, COMMENT the @jit decorator above
    neuronalModel = dummyNeuronalModel()
    Integrator.neuronalModel = neuronalModel

    # The analytic solution is y = e^t.
    def asol(t):
        return math.exp(t)

    clamping = False
    dt = 0.5
    Integrator.ds = dt
    Tmax = 5.0
    y0 = np.array([[1.0]])

    Integrator.initStimuli(dt, Tmax)

    simVars, obsVars = Integrator.integrate(dt, Tmax, y0)

    t = np.arange(0.0, Tmax, dt)
    yasol = np.vectorize(asol)
    plt.plot(t,obsVars.flatten()[:-1],'r-',label="Heun's")
    plt.plot(t,yasol(t),'b-', label='analytical')
    plt.legend()
    plt.show()

# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF
