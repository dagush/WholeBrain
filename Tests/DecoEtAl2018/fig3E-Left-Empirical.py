# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  Computes empirical histograms from the placebo/LCD fMRIs
#
#  This code can be run without other previous file.
#
#  Taken from the code (FCD_LSD_empirical.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import hdf5storage as sio
import WholeBrain.Observables.swFCD as FCD

from setup import *


# set BOLD filter settings
import Observables.BOLDFilters as filters
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
fMRIFile = inFilePath + '/LSDnew.mat'
print(f"Loading {fMRIFile}")
LSDnew = sio.loadmat(fMRIFile)  #load LSDnew.mat tc_aal
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
fileName = outFilePath + '/FCD_values_Empirical.mat'
print(f"Saving {fileName}")
sio.savemat(fileName, {'cotsampling': cotsampling})  # Save all project variables!

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

