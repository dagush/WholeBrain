# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#   
#  Computes simulations with the Dynamic Mean Field Model (DMF) using 
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#
#  - the optimal coupling (we=2.1) for fitting the placebo condition 
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#  Taken from the code (FCD_LSD_simulated.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
from pathlib import Path
from numba import jit
import time

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Models.DynamicMeanField as neuronalModel
import functions.Models.serotonin2A as serotonin2A
import functions.Integrator_EulerMaruyama as integrator
# import functions.Integrator_Euler as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import functions.simulateFCD as simulateFCD
simulateFCD.integrator = integrator
import functions.BOLDHemModel_Stephan2007 as Stephan2007
simulateFCD.BOLDModel = Stephan2007

from functions import BalanceFIC
BalanceFIC.integrator = integrator

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .02                         # lowpass frequency of filter
filters.fhi = 0.1                         # highpass
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    serotonin2A.recompileSignatures()
    integrator.recompileSignatures()


@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # originally set to 13


def my_hist(x, bin_centers):
    bin_edges = np.r_[-np.Inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]), np.Inf]
    counts, edges = np.histogram(x, bin_edges)
    return [counts, bin_centers]


# Load Structural Connectivity Matrix
print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
C=sc90/np.max(sc90[:])*0.2  # Normalization...

NumSubjects = 15  # Number of Subjects in empirical fMRI dataset
print("Simulating {} subjects!".format(NumSubjects))


# Setup for Serotonin 2A-based DMF simulation!!!
# ----------------------------------------------
neuronalModel.He = serotonin2A.phie
neuronalModel.Hi = serotonin2A.phii

# Load Regional Drug Receptor Map
print('Loading Data_Raw/mean5HT2A_bindingaal.mat')
mean5HT2A_aalsymm = sio.loadmat('Data_Raw/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()


# ============================================================================
# ============= Compute the J values for Balance conditions ==================
# ============================================================================
# Define optimal parameters
neuronalModel.we = 2.1  # Global Coupling parameter
serotonin2A.wgaini = 0.  # Placebo conditions, to calibrate the J's...
serotonin2A.wgaine = 0.
# ==== J is calculated this only once, then saved
baseName = "Data_Produced/SC90/J_Balance_we{}.mat".format(np.round(neuronalModel.we, decimals=2))
neuronalModel.J = BalanceFIC.Balance_J9(neuronalModel.we, C, baseName)['J'].flatten()
integrator.recompileSignatures()
# if not Path(fileName).is_file():
#     print("Computing {} !!!".format(fileName))
#     # recompileSignatures()
#     J, nodeCount = BalanceFIC.JOptim(C)
#     neuronalModel.J = J.flatten()  # This is the Feedback Inhibitory Control
#     sio.savemat(fileName, {'J': neuronalModel.J})  # save J_Balance J
# else:
#     print("Loading {} !!!".format(fileName))
#     # ==== J can be calculated only once and then load J_Balance J
#     neuronalModel.J = sio.loadmat(fileName)['J'].flatten()


initRandom()


# ============================================================================
# ============= Simulate Placebo =============================================
# ============================================================================
if True: #not Path("FCD_values_placebo.mat").is_file():
    # SIMULATION OF OPTIMAL PLACEBO
    print("\n\nSIMULATION OF OPTIMAL PLACEBO")
    wge = 0. # 0 for placebo, 0.2 for LSD
    serotonin2A.wgaini = 0.
    serotonin2A.wgaine = wge

    recompileSignatures()
    start_time = time.clock()
    cotsampling_pla_s = simulateFCD.simulate(NumSubjects, C)
    print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))

    sio.savemat('Data_Produced/FCD_values_placebo.mat', {'cotsampling_pla_s': cotsampling_pla_s})  # save FCD_values_placebo cotsampling_pla_s
else:
    print("LOADING OPTIMAL PLACEBO")
    cotsampling_pla_s = sio.loadmat('Data_Produced/FCD_values_placebo.mat')['cotsampling_pla_s']


# ============================================================================
# ============= Simulate LSD =================================================
# ============================================================================
if True: #not Path("Data_Produced/FCD_values_lsd.mat").is_file():
    # SIMULATION OF OPTIMAL LSD fit
    print("\n\nSIMULATION OF OPTIMAL LSD fit ")
    wge = 0.2 # 0 for placebo, 0.2 for LSD
    serotonin2A.wgaini = 0.
    serotonin2A.wgaine = wge

    recompileSignatures()
    start_time = time.clock()
    cotsampling_lsd_s = simulateFCD.simulate(NumSubjects, C)
    print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))

    sio.savemat('Data_Produced/FCD_values_lsd.mat', {'cotsampling_lsd_s': cotsampling_lsd_s})  # save FCD_values_lsd cotsampling_lsd_s
else:
    print("LOADING OPTIMAL LSD fit")
    cotsampling_lsd_s = sio.loadmat('Data_Produced/FCD_values_lsd.mat')['cotsampling_lsd_s']

# ============================================================================
# Plot
# ============================================================================
[h_pla, x1] = my_hist(cotsampling_pla_s[:].T.flatten(), np.arange(-.1, 1.025, .025))
[h_lsd, x] = my_hist(cotsampling_lsd_s[:].T.flatten(), np.arange(-.1, 1.025, .025))

import matplotlib.pyplot as plt

width=0.01
plaBar = plt.bar(x, h_pla, width=width, color="red", label="Placebo")
lsdBar = plt.bar(x+width, h_lsd, width=width, color="blue", label="LSD")
plt.xlabel('FCD values')
plt.ylabel('Count')
plt.legend(handles=[plaBar, lsdBar], loc='upper right')
plt.title('Simulated data')
plt.show()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
