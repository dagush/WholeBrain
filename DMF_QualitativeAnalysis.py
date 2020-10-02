# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of the Dynamic Mean Field
# (DMF) model (a.k.a., Reduced Wong-Wang), from
#
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================

# ==========================================================================
# numeric definitions...
# ==========================================================================

class DMFModel:
    def __init__(self):
        import functions.Stimuli.constant as stimuli
        self.stimuli = stimuli
        self.stimuli.onset = 0.
        self.stimuli.amp = 0.
        # integrator.stimuli = stimuli
        self.index = 0

    def dfun(self, simVars, t=None):
        [sn, sg] = simVars
        vsn, vsg = np.array([sn]), np.array([sg])
        N = 1
        DMF.SC = np.zeros((N,N))
        DMF.J = np.ones(N)
        DMF.recompileSignatures()
        if t is None: t = 0.
        dvars = DMF.dfun([vsn, vsg], self.stimuli.stimulus(t))
        return np.array(dvars).reshape((2,))

    def parmNames(self):
        return ['sn', 'sg']

    def getObservationVarName(self):
        # return 'sn' if self.index == 0 else 'sg'
        return '$r_n$'

    def setControlParm(self, lmbd):
        self.stimuli.amp = lmbd

    def getControlParmName(self):
        return '$I_{ext}$'

    def getObservationVar(self, simVars, lmbd):
        # return simVars[self.index]
        self.setControlParm(lmbd)
        dvars = self.dfun(simVars, 0.)
        return DMF.rn

# ==========================================================================
# symbolic definitions...
# ==========================================================================
import functions.Stimuli.constant as stimuli
def symDMF():
    C = 0.
    DMF.J = 1.
    sn, sg = sm.symbols('sn, sg')
    [dsn, dsg] = DMF.dfun([sn, sg], C, stimuli.stimulus(0.))
    return [sn, sg], [dsn, dsg]


# ==========================================================================
# ==========================================================================
# ==========================================================================
symbolic = False
if symbolic:
    # ==========================================================================
    # Symbolic
    # ==========================================================================
    import sympy as sm
    import functions.SymAnalysis as symA
    import functions.Models.Sym_DynamicMeanField as DMF
    print("==========================================")
    print("=  DMF Equations to compute symbolically =")
    print("==========================================")
    [sn, sg], [dsn, dsg] = symDMF()
    print("dsn=", dsn)
    print("dsg=", dsg)
    print("==========================================")
    interval = {'left': -0.5, 'right': 2, 'bottom': -0.5, 'top': 2}
    symA.plotPhasePlane(symDMF, interval, background='flow', type="solve explicit")
else:
    # ==========================================================================
    # Numeric
    # ==========================================================================
    import numpy as np
    import functions.NumAnalysis as numA
    import functions.Models.DynamicMeanField as DMF
    model = DMFModel()
    interval = {'left': -0.5, 'right': 2, 'bottom': -0.5, 'top': 2}
    print("=========================================")
    print("=  DMF Equations to compute numerically =")
    print("=========================================")
    initialcond = [0.001, 0.001]
    # numA.plotODEInt(numDMF, parmsDMF, initialcond)
    # numA.plot_PhasePlane_Only(numDMF, parmsDMF(), interval, background='flow')  #'quiver'    trajectories=[initialcond])
    lbda_space = np.linspace(0, 1, 100)
    # numA.plot_BifurcationDiagram_Only(model, interval, lbda_space, fullBifurcationEvaluations=20)

    numA.plotFancyBifurcationDiagram(model, interval, lbda_space,
                                     drawNullclines=False, fullBifurcationEvaluations=20)

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
