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
# ==========================================================================
class FitzHugh_Nagumo_model:
    def __init__(self):
        self.I_ext = 0.5
        self.a = 0.7
        self.b = 1.8  # 0.8
        self.tau = 12.5  # 1/tau = 0.08
        self.index = 0

    def dfun(self, x, t):
        # v, w = sm.symbols('v, w')
        v, w = x
        eq1 = v - v**3/3 - w + self.I_ext
        eq2 = (v + self.a - self.b*w)/self.tau
        return np.array([eq1, eq2])

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
print("=====================================================")
# print("Equation:", FitzHugh_Nagumo())
print("=====================================================")
model = FitzHugh_Nagumo_model()
interval = {'left': -2.5, 'right': 2.5, 'bottom': -2.5, 'top': 2.5}
# numA.plotODEInt(FitzHugh_Nagumo, parms_FitzHugh_Nagumo(), [1, 0.01])
# numA.plot_PhasePlane_Only(FitzHugh_Nagumo, parms_FitzHugh_Nagumo(), interval, trajectories=[[1, 0.01], [-2.5,-0.75]], background='flow')
lbda_space = np.linspace(0, 1, 100)
numA.plotFancyBifurcationDiagram(model,
                                 interval, lbda_space,
                                 drawNullclines=False, fullBifurcationEvaluations=20)

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
