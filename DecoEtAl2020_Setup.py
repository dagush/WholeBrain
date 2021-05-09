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
import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
# ----------------------------------------------
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007
# import functions.simulateFCD as simulateFCD
# simulateFCD.integrator = integrator
# simulateFCD.BOLDModel = Stephan2007

import functions.Observables.FC as FC
import functions.Observables.GBC as GBC
import functions.Observables.swFCD as swFCD

import functions.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

import functions.Optimizers.Optim1D as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator
# optim1D.neuronalModel = ... # Leave this for the specific implementations...

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2       # 2nd order butterworth filter
filters.flp = .008  # lowpass frequency of filter
filters.fhi = .08   # highpass
filters.TR = 0.754  # sampling interval
# --------------------------------------------------------------------------
#  End modules setup...
# --------------------------------------------------------------------------


# ==================================================================================
#  some useful functions
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
baseInPath = 'Data_Raw/DecoEtAl2020'
baseOutPath = 'Data_Produced/DecoEtAl2020'

N = 68
NSUB = 389
NumTrials = 15  # should be NSUB...
simulateBOLD.Tmax = 616
simulateBOLD.TR = 0.754  # sampling interval
simulateBOLD.dtt = 1e-3
simulateBOLD.Toffset = 14.
simulateBOLD.recomputeTmaxneuronal()

# load genetic info
print("Loading DKcortex_selectedGenes.mat")
DKGenes = sio.loadmat(baseInPath+'/DKcortex_selectedGenes.mat')
expMeasures = DKGenes['expMeasures']
print(f'raw expMeasures from DKGenes shape is {expMeasures.shape}')

coefei = np.sum(expMeasures[:,17:25],1) / np.sum(expMeasures[:,1:6],1)  # ampa+nmda/gaba
ratioEI = np.zeros(N)
ratioEI[:coefei.size] = coefei/(max(coefei)-min(coefei))
ratioEI = ratioEI-max(ratioEI)+1
ratioEI[34:68] = ratioEI[0:34]
print(f'ratioEI shape is {ratioEI.shape}')

# Read data..SC FC and time series of BOLD
# =============================================================
print('loading SC_GenCog_PROB_30.mat')
GrCV = sio.loadmat(baseInPath+'/SC_GenCog_PROB_30.mat')['GrCV']
print(f'Raw GrCV shape is {GrCV.shape}')
tcrange = np.union1d(np.arange(0,34), np.arange(41,75))  # [1:34 42:75]
C = GrCV[:, tcrange][tcrange, ]
C = C/np.max(C)*0.2
print(f'C shape is {C.shape}')
# print(np.sum(C))
# indexsub=1:NSUB;

print('loading DKatlas_noGSR_timeseries.mat')
ts = sio.loadmat(baseInPath+'/DKatlas_noGSR_timeseries.mat')['ts']
print(f'ts shape is {ts.shape}')

# ====================== By default, we set up the parameters for the DEFAULT mode:
# serotonin2A.wgaine = 0.
# serotonin2A.wgaini = 0.
# recompileSignatures()
#
tc_transf = transformEmpiricalSubjects(ts, tcrange, NSUB)
print(f'tc_transf shape is {tc_transf[0].shape} for {NSUB} subjects')
# FCemp_cotsampling = G_optim.processEmpiricalSubjects(tc_transf, distanceSettings, baseOutPath+"fNeuro_emp.mat")
# FCemp = FCemp_cotsampling['FC']; cotsampling = FCemp_cotsampling['swFCD'].flatten(); GBCemp = FCemp_cotsampling['GBC'];

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
