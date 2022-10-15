# ==========================================================================
# ==========================================================================
#  Computes the Fitting of the Serotonin 5-HT2AR Model to the LSD data
#
#  For the LSD condition, when using this optimal coupling point of the placebo
# condition and systematically scaling the excitatory gain function in each
# region with the empirical 5-HT2AR data, we find that there is an optimum at
# around (0.2,0.045) (minimum of blue line). In contrast, varying the scaling of the
# neuronal gain for the placebo condition does not yield an optimum (see
# monotonically rising green line), and thus the fit is not improved by changing
# the scaling of the neuronal gain by 5-HT2AR density. This clearly demonstrates
# that the LSD brain activity is dependent on the precise 5-HT2A density distribution
# maps.
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
#  Code written by Josephine Cruzat josephine.cruzat@upf.edu
#
#  Translated to Python by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pathlib import Path

from DecoEtAl2018_Setup import *

def loadFittingData(suffix):
    filePath = outFilePath + '/DecoEtAl2018_fitting'+suffix+'.mat'
    print('Loading {}'.format(filePath))
    data = sio.loadmat(filePath)
    return data['fitting_FCD'].flatten(), data['S_E'].flatten()


fitting_PLA, S_E_PLA = loadFittingData('_PLA')
fitting_LSD, S_E_LSD = loadFittingData('_LSD')

minPLA = S_E_PLA[np.argmin(fitting_PLA)]
minLSD = S_E_LSD[np.argmin(fitting_LSD)]
print("\n\n#####################################################################################################")
print(f"# Max FC({minPLA}) = {np.max(fitting_PLA)}             ** Min FCD({minLSD}) = {np.min(fitting_LSD)} **")
print("#####################################################################################################\n\n")

plt.rcParams.update({'font.size': 15})
plotFCDpla, = plt.plot(S_E_PLA, fitting_PLA)
plotFCDpla.set_label("Placebo")
plotFCpla, = plt.plot(S_E_LSD, fitting_LSD)
plotFCpla.set_label("LSD")
plt.title("Whole-brain fitting")
plt.ylabel("FCD Fitting")
plt.xlabel("Exhitatory Gain Modulation")
plt.legend()
plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
