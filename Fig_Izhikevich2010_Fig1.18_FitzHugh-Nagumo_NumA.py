# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of several ODEs
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import functions.NumAnalysis as numA

# ==========================================================================
# ODE: FitzHugh–Nagumo model
# Eugene M. Izhikevich and Richard FitzHugh (2006), Scholarpedia, 1(9):1349. doi:10.4249/scholarpedia.1349
# http://www.scholarpedia.org/article/FitzHugh-Nagumo_model
# Also at:
# https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
# values from [Izhikevich, Dynamical Systems in Neuroscience, 2010, fig 1.18]
# ==========================================================================
class FitzHugh_Nagumo_model:
    def __init__(self):
        self.I_ext = 0.1
        self.a = 0.7
        self.b = 0.8  # 1.8
        self.tau = 12.5  # 1/tau = 0.08
        self.index = 0

    def dfun(self, x, t=None):
        v, w = x
        dVdt = v - v**3/3 - w + self.I_ext
        dWdt = (v + self.a - self.b*w)/self.tau
        return np.array([dVdt, dWdt])

    def parmNames(self):
        return ['u', 'w']

    def getObservationVarName(self):
        return self.parmNames()[self.index]

    def setControlParm(self, lmbd):
        self.I_ext = lmbd

    def getControlParmName(self):
        return '$I_{ext}$'

    def getObservationVar(self, simVars, lmbd):
        return np.array([simVars[self.index]])


# ==========================================================================
# ==========================================================================
# ==========================================================================

# ======================== FitzHugh-Nagumo... ==========================
print("=====================================================")
print("=  FitzHugh-Nagumo Equation to compute numerically  =")
print("=      (from [Izhikevich2010, fig 1.18])            =")
print("=====================================================")
# print("Equation:", FitzHugh_Nagumo())
print("=====================================================")
model = FitzHugh_Nagumo_model()
interval = {'left': -2.5, 'right': 2.5, 'bottom': -1.2, 'top': 1.2}
# numA.plotODEInt(model, parms=('label1', 'label2'), initialCond=[1, 0.01])
numA.plot_PhasePlane_Only(model, interval,
                          trajectories=[[-0.5, -0.75], [-2.5,-0.75], [0, 0.4]],
                          background='flow')
# lbda_space = np.linspace(0, 1, 100)
# numA.plotFancyBifurcationDiagram(model,
#                                  interval, lbda_space,
#                                  drawNullclines=False, fullBifurcationEvaluations=20)

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
