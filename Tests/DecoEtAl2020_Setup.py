# ==========================================================================
# ==========================================================================
#  Setup for the code from the paper
#
#  Taken from the code from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito, "Dynamical consequences of regional heterogeneity
#  in the brain’s transcriptional landscape", 2021, biorXiv
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================


import numpy as np
import scipy.io as sio
from numba import jit


# --------------------------------------------------------------------------
#  Begin modules setup...
# --------------------------------------------------------------------------
# Setup for ...-based DMF simulation!!!
# import WholeBrain.Models.DynamicMeanField as neuronalModel
import WholeBrain.Models.Transcriptional as neuronalModel
# ----------------------------------------------
import WholeBrain.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import WholeBrain.BOLDHemModel_Stephan2007 as Stephan2007
import WholeBrain.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007
# import WholeBrain.simulateFCD as simulateFCD
# simulateFCD.integrator = integrator
# simulateFCD.BOLDModel = Stephan2007

import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.GBC as GBC
import WholeBrain.Observables.swFCD as swFCD

import WholeBrain.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

# set BOLD filter settings
import WholeBrain.BOLDFilters as filters
filters.k = 2       # 2nd order butterworth filter
filters.flp = .008  # lowpass frequency of filter
filters.fhi = .08   # highpass
filters.TR = 0.754  # sampling interval
# --------------------------------------------------------------------------
#  End modules setup...
# --------------------------------------------------------------------------


# ==================================================================================
#  some useful WholeBrain
# ==================================================================================
@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # originally set to 13


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    # serotonin2A.recompileSignatures()
    integrator.recompileSignatures()


def transformEmpiricalSubjects(tc_aal, tcrange, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = tc_aal[:, tcrange, s].T
    return transformed

# ==================================================================================
# ==================================================================================
#  initialization
# ==================================================================================
# ==================================================================================
initRandom()
baseInPath = '../Data_Raw/DecoEtAl2020'
baseOutPath = '../Data_Produced/DecoEtAl2020'


N = 68
NSUB = 389
NumTrials = 15  # should be NSUB...
simulateBOLD.Tmax = 616
simulateBOLD.TR = 0.754  # sampling interval
simulateBOLD.dtt = 1e-3
simulateBOLD.Toffset = 14.
simulateBOLD.recomputeTmaxneuronal()

# =============================================================
# load genetic info
# =============================================================
print("Loading DKcortex_selectedGenes.mat")
DKGenes = sio.loadmat(baseInPath+'/DKcortex_selectedGenes.mat')
expMeasures = DKGenes['expMeasures']
print(f'raw expMeasures from DKGenes shape is {expMeasures.shape}')

coefe = np.sum(expMeasures[:,17:25],1) # / np.sum(expMeasures[:,1:6],1)  # ampa+nmda/gaba
ratioE = np.zeros(N)
ratioI = np.zeros(N)
ratioE[0:34] = coefe/(np.max(coefe))
ratioE[34:68] = ratioE[0:34]

coefrange = np.union1d(np.arange(1,9), np.arange(11,14))
coefi = np.sum(expMeasures[:,coefrange], 1)  # 18:21 ampa+ 22:25 nmda/gaba
ratioI[0:34] = coefi/(np.max(coefi))
ratioI[34:68] = ratioI[0:34]
ratio = ratioE/ratioI
ratio = ratio/(np.max(ratio)-np.min(ratio))
ratio = ratio - np.max(ratio) + 1

neuronalModel.ratio = ratio

# =============================================================
# Read data..SC FC and time-series of BOLD
# =============================================================
print('loading SC_GenCog_PROB_30.mat')
GrCV = sio.loadmat(baseInPath+'/SC_GenCog_PROB_30.mat')['GrCV']
print(f'Raw GrCV shape is {GrCV.shape}')
tcrange = np.union1d(np.arange(0,34), np.arange(41,75))  # [1:34 42:75]
C = GrCV[:, tcrange][tcrange, ]
C = C/np.max(C)*0.2
print(f'C shape is {C.shape}')
neuronalModel.setParms({'SC': C})  # Set the neuronal model with the SC

print('loading DKatlas_noGSR_timeseries.mat')
ts = sio.loadmat(baseInPath+'/DKatlas_noGSR_timeseries.mat')['ts']
print(f'ts shape is {ts.shape}')

# ====================== By default, we set up the parameters for the DEFAULT mode:
tc_transf = transformEmpiricalSubjects(ts, tcrange, NSUB)
print(f'tc_transf shape is {tc_transf[0].shape} for {NSUB} subjects')

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
