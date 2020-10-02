# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of the
# Normal form of a supercritical Hopf bifurcation (supHopf)
# model from
#
#     .. [Deco_2017]  Deco, G., Kringelbach, M.L., Jirsa, V.K. et al.
#     The dynamics of resting fluctuations in the brain: metastability and its
#     dynamical cortical core. Sci Rep 7, 3095 (2017).
#     https://doi.org/10.1038/s41598-017-03073-5
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================

# ==========================================================================
# numeric definitions...
# ==========================================================================

class supHopfModel:
    def __init__(self):
        # import functions.Stimuli.constant as stimuli
        # self.stimuli = stimuli
        # self.stimuli.onset = 0.
        # self.stimuli.amp = 0.
        # integrator.stimuli = stimuli
        self.index = 0

    def dfun(self, simVars, t=None):
        [x, y] = simVars
        vx, vy = np.array([x]), np.array([y])
        N = 1
        supHopf.SC = np.zeros((N,N))
        supHopf.SCT = np.zeros((N,N))
        supHopf.ink = supHopf.SCT.sum(axis=1)
        if not t: t = 0.
        dvars = supHopf.dfun([vx, vy], 0.)  # self.stimuli.stimulus(t))
        return np.array(dvars).reshape((2,))

    def parmNames(self):
        return ['x', 'y']

    def getObservationVarName(self):
        return 'x' if self.index == 0 else 'y'

    def setControlParm(self, lmbd):
        supHopf.a = lmbd

    def getControlParmName(self):
        return '$a$'

    def getObservationVar(self, simVars, lmbd):
        # return simVars[self.index]
        self.setControlParm(lmbd)
        dvars = self.dfun(simVars, 0.)
        return supHopf.x if self.index == 0 else supHopf.y

    def selectObservationVar(self, simVars):
        return simVars[0] if self.index == 0 else simVars[1]


# ==========================================================================
# ==========================================================================
# ==========================================================================
import numpy as np
import functions.NumAnalysis as numA
import functions.Models.supHopf as supHopf
model = supHopfModel()
supHopf.G = 0.
N = 1
supHopf.SC = np.zeros((N,N))
supHopf.SCT = np.zeros((N,N))
supHopf.ink = supHopf.SCT.sum(axis=1)
interval = {'left': -0.5, 'right': 0.5, 'bottom': -0.5, 'top': 0.5}
print("=============================================")
print("=  supHopf Equations to compute numerically =")
print("=============================================")
# initialcond = [0.001, 0.001]
# lbda_space = np.linspace(-0.5, 0.5, 100)
numA.plot_PhasePlane_Only(model, interval)
# numA.plot_BifurcationDiagram_Only(model, interval, lbda_space)
# numA.plotFancyBifurcationDiagram(model, interval, lbda_space,
#                                  # trajectories=[[-0.5, -0.6], [0.75,0.25]],
#                                  drawNullclines=False, fullBifurcationEvaluations=10,
#                                  phaseLabelLoc='lower right')

# ====================== DEBUG CODE
# model.setControlParm(0.5)
# numA.plot_PhasePlane_Only(model, interval,
#                           trajectories=[[-0.5, -0.6], [0.75,0.25]],
#                           drawNullclines=False,
#                           labelLoc='lower right')
# fps = numA.fixedPoints(model.dfun, interval)
# limits = numA.findLimitCycles(model, fps, interval, n_grid=10)
# print(limits)
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
