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
import time
import functions.Models.DynamicMeanField as neuronalModel
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import functions.simulateFCD as simulateFCD

from functions import BalanceFIC
BalanceFIC.integrator = integrator

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .02                         # lowpass frequency of filter
filters.fhi = 0.1                         # highpass


def my_hist(x, bin_centers):
    bin_edges = np.r_[-np.Inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]), np.Inf]
    counts, edges = np.histogram(x, bin_edges)
    return [counts, bin_centers]


# Load Structural Connectivity Matrix
print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']  #load LSDnew.mat tc_aal
C=sc90/np.max(sc90[:])*0.2

NumSubjects = 10  # Number of Subjects in empirical fMRI dataset
print("Simulating {} subjects!".format(NumSubjects))


# ============================================================================
# ============= Compute the J values for Balance conditions ==================
# ============================================================================
# Define optimal parameters
neuronalModel.we = 2.1  # Global Coupling parameter
# ==== J is calculated this only once, then saved
BalanceFIC.baseName = "Data_Produced/SC90/J_Balance_we{}.mat"
BalanceFIC.Balance_J9(neuronalModel.we, C)
# if not Path("Data_Produced/J_Balance_SC90.mat").is_file():
#     from functions import BalanceFIC
#     BalanceFIC.integrator = integrator
#     print("Computing Data_Produced/J_Balance_SC90 !!!")
#     neuronalModel.J=BalanceFIC.JOptim(C).flatten()  # This is the Feedback Inhibitory Control
#     sio.savemat('Data_Produced/J_Balance_SC90.mat', {'J': neuronalModel.J})  # save J_Balance J
# else:
#     print("Loading Data_Produced/J_Balance_SC90 !!!")
#     # ==== J can be calculated only once and then load J_Balance J
#     neuronalModel.J = sio.loadmat('Data_Produced/J_Balance_SC90.mat')['J'].flatten()

np.random.seed(13)

# ============================================================================
# ============= Simulate =====================================================
# ============================================================================
wge = 0. # 0 for placebo, 0.2 for LSD

print("\n\nSTARTING TIME MEASUREMENT\n\n")
start_time = time.clock()
cotsampling_pla_s = simulateFCD.simulate(NumSubjects, C)
print("\n\n--- TOTAL TIME: {} seconds (was: {} seconds, initial: {} seconds)---\n\n".format(time.clock() - start_time, 1871.0590252101958, 6327.425539))

max=np.max(cotsampling_pla_s)
min=np.min(cotsampling_pla_s)
avg=np.average(cotsampling_pla_s)
var=np.var(cotsampling_pla_s)
print("Max={}, min={}, avg={}, var={}".format(max, min, avg, var))
print("Was=(0.9920881163830183, 0.003266796847684554, 0.4743004646754597, 0.03813424749641985)")
# sio.savemat('Data_Produced/FCD_values_placebo.mat', {'cotsampling_pla_s': cotsampling_pla_s})  # save FCD_values_placebo cotsampling_pla_s

# ============================================================================
# Plot
# ============================================================================
[h_pla, x1] = my_hist(cotsampling_pla_s[:].T.flatten(), np.arange(-.1, 1.025, .025))

import matplotlib.pyplot as plt

width=0.01
plaBar = plt.bar(x1, h_pla, width=width, color="red", label="Placebo")
plt.xlabel('FCD values')
plt.ylabel('Count')
plt.legend(handles=[plaBar], loc='upper right')
plt.title('Simulated data')
plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
