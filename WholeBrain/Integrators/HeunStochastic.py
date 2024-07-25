# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of the Heun Stochastic Integrator
# Based on the code from TVB:
#     It is a simple example of a predictor-corrector method. It is also known as
#     modified trapezoidal method, which uses the Euler method as its predictor.
#     And it is also a implicit integration scheme.
#
#     [1] Kloeden and Platen, Springer 1995, *Numerical solution of stochastic
#         differential equations.
#     [2] Riccardo Mannella, *Integration of Stochastic Differential Equations
#         on a Computer*, Int J. of Modern Physics C 13(9): 1177--1194, 2002.
#
#          From [2]_:
#         .. math::
#             X_i(t) = X_i(t-1) + dX(X_i(t)/2 + dX(X_i(t-1))) dt + g_i(X) Z_1
#         in our case, :math:`noise = Z_1`
#         See page 1180.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from WholeBrain.Integrators.integr_utils import *
from numba import jit

print("Going to use the Heun stochastic integration scheme...")

neuronalModel = None  # To be able to choose the model externally...


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Heun Stochastic Integration
# --------------------------------------------------------------------------
sigma = 0.01
# @jit(nopython=True)
def integrationStep(simVars, dt, coupling, stimulus):
    numSimVars = simVars.shape[0]; N = simVars.shape[1]
    dvars_obsVars = neuronalModel.dfun(simVars, coupling, stimulus)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...

    noise = np.sqrt(dt) * sigma * randn(numSimVars,N)

    inter = simVars + dt * dvars + noise

    dvars_obsVars = neuronalModel.dfun(inter, coupling, stimulus)
    dvars2 = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    dX = (dvars + dvars2) * dt / 2.0

    simVars = simVars + dX + noise

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
        # We will use the differential equation y"(t) = -y(t).
        # This is written as
        #     y" = v' -> y' = v
        #                v' = -y
        def dfun(self, simVars, stimulus):
            y = simVars[0]
            v = simVars[1]
            dy = v
            dv = -y
            return np.stack((dy,dv)), np.stack((y, v))
        def recompileSignatures(self):
            pass
        def numObsVars(self):
            return 2

    # For this to work, COMMENT the @jit decorator above
    neuronalModel = dummyNeuronalModel()
    Integrator.neuronalModel = neuronalModel

    # The analytic solution is y = sin(t) because we initialize at 0.0.
    def asol(t):
        return math.sin(t)

    # Note that we are setting a CONSTANT sigma of 0.5, so it's better to use something like a sinus to visualize
    # its effects...
    clamping = False
    sigma = 0.05
    dt = 0.1
    Integrator.ds = dt
    Tmax = 20.0
    y0 = np.array([[0.0], [1.0]])  # y0=0, v0=cos(0)=1

    Integrator.initStimuli(dt, Tmax)

    simVars, obsVars = Integrator.integrate(dt, Tmax, y0)

    t = np.arange(0.0, Tmax, dt)
    yasol = np.vectorize(asol)
    plt.plot(t,obsVars[:,0,:].flatten()[:-1],'r-',label="Heun Stochastic's")
    plt.plot(t,yasol(t),'b-', label='analytical')
    plt.legend()
    plt.show()

# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF
