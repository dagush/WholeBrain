# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  Computes simulations with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#
#  - the optimal coupling (we=2.1) for fitting the placebo condition
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#  Taken from the code (FCD_LSD_empirical.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import WholeBrain.Observables.swFCD as FCD

from DecoEtAl2018_Setup import *


# set BOLD filter settings
import WholeBrain.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .04                         # lowpass frequency of filter
filters.fhi = 0.07                        # highpass


def my_hist(x, bin_centers):
    bin_edges = np.r_[-np.Inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]), np.Inf]
    counts, edges = np.histogram(x, bin_edges)
    return [counts, bin_centers]


Subjects = 15
Conditions = [1, 4]  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

#load fMRI data
print("Loading Data_Raw/LSDnew.mat")
LSDnew = sio.loadmat(inFilePath+'/LSDnew.mat')  #load LSDnew.mat tc_aal
tc_aal = LSDnew['tc_aal']
(N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time


N_windows = len(range(0,190,3))  # This shouldn't be done like this in Python!!!
cotsampling = np.zeros([len(Conditions), Subjects, int(N_windows*(N_windows-1)/2)])


# Loop over conditions and subjects
for task in range(len(Conditions)):
    print("Task:", task, "(", Conditions[task], ")")
    for s in range(Subjects):
        print('   Subject: ', s)
        signal = tc_aal[s, Conditions[task]]
        cotsampling[task, s, :] = FCD.from_fMRI(signal)


# Save & Plot
# ----------------------------------------------------
print("Saving Data_Raw/FCD_values_Empirical.mat")
sio.savemat(inFilePath + '/FCD_values_Empirical.mat', {'cotsampling': cotsampling})  # Save all project variables!

cots = cotsampling[0, :]
cotsf = cots.T.flatten()
[h_lsd, x] = my_hist(cotsampling[0, :].T.flatten(), np.arange(-.1, 1.025, .025))
[h_pla, x1] = my_hist(cotsampling[1, :].T.flatten(), np.arange(-.1, 1.025, .025))

import matplotlib.pyplot as plt

width=0.01
plaBar = plt.bar(x, h_pla, width=0.01, color="red", label="Placebo")
lsdBar = plt.bar(x+width, h_lsd, width=0.01, color="blue", label="LSD")
plt.xlabel('FCD values')
plt.ylabel('Count')
plt.legend(handles=[lsdBar, plaBar], loc='upper right')
plt.title('Empirical data')
plt.show()

