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
#  Taken from the code (Code_Figure3.m) from:
#
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
from pathlib import Path
import time

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
from DecoEtAl2018_Setup import *

import DecoEtAl2018_simulateFCD as simulateFCD
simulateFCD.integrator = integrator
simulateFCD.simModel = simulateBOLD
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


def my_hist(x, bin_centers):
    bin_edges = np.r_[-np.Inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]), np.Inf]
    counts, edges = np.histogram(x, bin_edges)
    return [counts, bin_centers]


# ============================================================================
# ============= Compute the J values for Balance conditions ==================
# ============================================================================
# Define optimal parameters
# ==== J is calculated this only once, then saved
baseName = outFilePath + "/J_Balance_we2.1.mat"
balancedG = BalanceFIC.Balance_J9(2.1, C, False, baseName)
balancedG['J'] = balancedG['J'].flatten()
balancedG['we'] = balancedG['we']  #.flatten()
serotonin2A.setParms(balancedG)
# sets the wgaine and wgaini, but using the standard protocol...
# Placebo conditions (both are 0), to calibrate the J's.
serotonin2A.setParms({'S_E':0., 'S_I':0.})
recompileSignatures()

initRandom()


# ============================================================================
# ============= Simulate Placebo =============================================
# ============================================================================
pla_path = outFilePath + '/FCD_values_placebo.mat'
if not Path(pla_path).is_file():
    # SIMULATION OF OPTIMAL PLACEBO
    print("\n\nSIMULATION OF OPTIMAL PLACEBO")
    # sets the wgaine and wgaini, but using the standard protocol... S_E = 0 for placebo, 0.2 for LSD
    serotonin2A.setParms({'S_E':0., 'S_I':0.})
    recompileSignatures()

    start_time = time.clock()
    cotsampling_pla_s = simulateFCD.simulate(NumSubjects, C)
    print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))

    sio.savemat(pla_path, {'cotsampling_pla_s': cotsampling_pla_s})  # save FCD_values_placebo cotsampling_pla_s
else:
    print("LOADING OPTIMAL PLACEBO")
    cotsampling_pla_s = sio.loadmat(pla_path)['cotsampling_pla_s']


# ============================================================================
# ============= Simulate LSD =================================================
# ============================================================================
lsd_path = outFilePath + '/FCD_values_lsd.mat'
if True: #not Path(lsd_path).is_file():
    # SIMULATION OF OPTIMAL LSD fit
    print("\n\nSIMULATION OF OPTIMAL LSD fit ")
    # sets the wgaine and wgaini, but using the standard protocol... S_E = 0 for placebo, 0.2 for LSD
    serotonin2A.setParms({'S_E':0.2, 'S_I':0.})
    recompileSignatures()

    # start_time = time.clock()
    cotsampling_lsd_s = simulateFCD.simulate(NumSubjects)
    # print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))

    sio.savemat(lsd_path, {'cotsampling_lsd_s': cotsampling_lsd_s})  # save FCD_values_lsd cotsampling_lsd_s
else:
    print("LOADING OPTIMAL LSD fit")
    cotsampling_lsd_s = sio.loadmat(lsd_path)['cotsampling_lsd_s']

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
