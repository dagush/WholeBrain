# ==========================================================================
# ==========================================================================
#  Setup for the code from the paper
#
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., Logothetis,N.K. & Kringelbach,M.L.
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
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
import WholeBrain.Models.serotonin2A as serotonin2A
# ----------------------------------------------
import WholeBrain.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = serotonin2A
integrator.verbose = False
import WholeBrain.BOLDHemModel_Stephan2007 as Stephan2007
import WholeBrain.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007

import WholeBrain.Optimizers.ParmSeep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.swFCD as swFCD

import WholeBrain.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

# set BOLD filter settings
import WholeBrain.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass

PLACEBO_cond = 4; LSD_cond = 1   # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...
# --------------------------------------------------------------------------
#  End modules setup...
# --------------------------------------------------------------------------

inFilePath = '../Data_Raw/'
outFilePath = '../Data_Produced/SC90'


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
    serotonin2A.recompileSignatures()
    integrator.recompileSignatures()


def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,90,2)
    even = np.arange(1,90,2)[::-1]  # sort 'descend'
    symLR = np.zeros((90,TC.shape[1]))
    symLR[0:45,:] = TC[odd,:]
    symLR[45:90,:] = TC[even,:]
    return symLR


def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = LR_version_symm(tc_aal[s,cond])
    return transformed

# ==================================================================================
# ==================================================================================
#  initialization
# ==================================================================================
# ==================================================================================
initRandom()

# Load Structural Connectivity Matrix
print(f"Loading {inFilePath}/all_SC_FC_TC_76_90_116.mat")
sc90 = sio.loadmat(inFilePath+'/all_SC_FC_TC_76_90_116.mat')['sc90']
C = sc90/np.max(sc90[:])*0.2  # Normalization...
serotonin2A.setParms({'SC': C})  # Set the model with the SC

# Load Regional Drug Receptor Map
print(f'Loading {inFilePath}/mean5HT2A_bindingaal.mat')
mean5HT2A_aalsymm = sio.loadmat(inFilePath+'/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()
recompileSignatures()

#load fMRI data
print(f"Loading {inFilePath}/LSDnew.mat")
LSDnew = sio.loadmat(inFilePath+'/LSDnew.mat')  #load LSDnew.mat tc_aal
tc_aal = LSDnew['tc_aal']
(N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
print(f'tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}')

NumSubjects = 15  # Number of Subjects in empirical fMRI dataset, originally 20...
print(f"Simulating {NumSubjects} subjects!")

# ====================== By default, we set up the parameters for the DEFAULT mode:
# Sets the wgaine and wgaini to 0, but using the standard protocol...
# We initialize both to 0, so we have Placebo conditions.
serotonin2A.setParms({'S_E':0., 'S_I':0.})
recompileSignatures()

tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
# FCemp_cotsampling_PLA = G_optim.processEmpiricalSubjects(tc_transf_PLA, distanceSettings, "Data_Produced/SC90/fNeuro_emp_PLA.mat")
# FCemp_PLA = FCemp_cotsampling_PLA['FC']; cotsampling_PLA = FCemp_cotsampling_PLA['swFCD'].flatten()

tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD
# FCemp_cotsampling_LSD = G_optim.processEmpiricalSubjects(tc_transf_LSD, distanceSettings, "Data_Produced/SC90/fNeuro_emp_LCD.mat")  # LCD
# FCemp_LSD = FCemp_cotsampling_LSD['FC']; cotsampling_LSD = FCemp_cotsampling_LSD['swFCD'].flatten()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
