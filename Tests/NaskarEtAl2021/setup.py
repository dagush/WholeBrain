# ==========================================================================
# ==========================================================================
#  Setup for the code from the paper
#
#  [NaskarEtAl_2018] Amit Naskar, Anirudh Vattikonda, Gustavo Deco,
#      Dipanjan Roy, Arpan Banerjee; Multiscale dynamic mean field (MDMF)
#      model relates resting-state brain dynamics with local cortical
#      excitatory–inhibitory neurotransmitter homeostasis.
#      Network Neuroscience 2021; 5 (3): 757–782. doi: https://doi.org/10.1162/netn_a_00197
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
# Setup for Serotonin 2A-based DMF simulation!!!
# This is a wrapper for the DMF (calls it internally, but before switches the
# two gain functions phie and phii for the right ones...
# import WholeBrain.Models.DynamicMeanField as DMF
import WholeBrain.Models.Naskar as Naskar
import WholeBrain.Models.Couplings as Couplings
# ----------------------------------------------
# import WholeBrain.Integrators.EulerMaruyama as scheme
import WholeBrain.Integrators.Euler as scheme
scheme.sigma = 0.001  # np.array([0.001, 0.001, 0.])
scheme.neuronalModel = Naskar
import WholeBrain.Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = Naskar
integrator.verbose = False
import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2008 as Stephan2008
import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2008

simulateBOLD.TR = 1.  # sampling interval
simulateBOLD.dtt = 1e-3
simulateBOLD.Toffset = 1*60.  # in seconds
simulateBOLD.Tmax = 8*60.  # in seconds
simulateBOLD.recomputeTmaxneuronal()
simulateBOLD.warmUp = True

# --------------------------------------------------------------------------
# Import optimizer (ParmSweep)
import WholeBrain.Optimizers.ParmSweep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

# --------------------------------------------------------------------------
# chose a FIC mechanism
# import Utils.FIC.BalanceFIC as BalanceFIC
# BalanceFIC.integrator = integrator
# import Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
# BalanceFIC.balancingMechanism = Deco2014Mechanism  # default behaviour for this project


# --------------------------------------------------------------------------
# Filters and Observables
# --------------------------------------------------------------------------
# set BOLD filter settings
import WholeBrain.Observables.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass
filters.TR = 1.                           # TR

# import observables
import WholeBrain.Observables.measures as measures
import WholeBrain.Observables.FC as FC
# import Observables.swFCD as swFCD

FC.defaultMeasure = measures.L2()
FC.distance = FC.defaultMeasure.distance
FC.findMinMax = FC.defaultMeasure.findMinMax

# --------------------------------------------------------------------------
#  End modules setup...
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# File loading…
# --------------------------------------------------------------------------
inFilePath = 'data/'
outFilePath = '../../Data_Produced/Tests/NaskarEtAl2021/'


# ==================================================================================
#  some useful WholeBrain
# ==================================================================================
@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # originally set to 13


# def recompileSignatures():
#     # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
#     # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
#     print("\n\nRecompiling signatures!!!")
#     serotonin2A.recompileSignatures()
#     integrator.recompileSignatures()


# def LR_version_symm(TC):
#     # returns a symmetrical LR version of AAL 90x90 matrix
#     odd = np.arange(0,90,2)
#     even = np.arange(1,90,2)[::-1]  # sort 'descend'
#     symLR = np.zeros((90,TC.shape[1]))
#     symLR[0:45,:] = TC[odd,:]
#     symLR[45:90,:] = TC[even,:]
#     return symLR


# def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
#     transformed = {}
#     for s in range(NumSubjects):
#         # transformed[s] = np.zeros(tc_aal[0,cond].shape)
#         transformed[s] = LR_version_symm(tc_aal[s,cond])
#     return transformed


# ==================================================================================
# ==================================================================================
#  initialization
# ==================================================================================
# ==================================================================================
initRandom()

# ------------ Load Structural Connectivity Matrix
print(f"Loading {inFilePath}/avgSC68.mat")
sc68 = sio.loadmat(inFilePath+'/avgSC68.mat')['avgSC40']
# C = sc68/np.max(sc68)*0.2  # Normalization...
N = sc68.shape[0]
# sc68[1:N+1:N*N] = 0
Naskar.setParms({'SC': sc68})  # Set the model with the SC
Naskar.couplingOp.setParms(sc68)

# Load Regional Drug Receptor Map
# print(f'Loading {inFilePath}/mean5HT2A_bindingaal.mat')
# mean5HT2A_aalsymm = sio.loadmat(inFilePath+'/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
# serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()

# ------------ load fMRI data
print(f"Loading {inFilePath}/avgFC68.mat")
avgFC = sio.loadmat(inFilePath+'/avgFC68.mat')['rs_FC']  #load LSDnew.mat tc_aal
fc_all = {'FC': avgFC}
# (N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
# print(f'tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}')

NumSubjects = 1  # Number of Subjects in empirical fMRI dataset, originally 20...
# print(f"Simulating {NumSubjects} subjects!")


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
