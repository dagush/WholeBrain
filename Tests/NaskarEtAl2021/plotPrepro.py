# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#
#  Computes simulations with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) with an adaptation mechanism
#
#  Before this, needs the results computed in
#   - prepro.py
#
#  Taken from the code from:
#  [NaskarEtAl_2018] Amit Naskar, Anirudh Vattikonda, Gustavo Deco,
#      Dipanjan Roy, Arpan Banerjee; Multiscale dynamic mean field (MDMF)
#      model relates resting-state brain dynamics with local cortical
#      excitatory–inhibitory neurotransmitter homeostasis.
#      Network Neuroscience 2021; 5 (3): 757–782. doi: https://doi.org/10.1162/netn_a_00197
#
#  Translated to Python by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import hdf5storage as sio
import matplotlib.pyplot as plt

from pathlib import Path

from setup import *

filePath = outFilePath + '/NaskarEtAl2021_fneuro.mat'
if not Path(filePath).is_file():
    import prepro as prepro
    prepro.prepro_G_Optim()

print('Loading {}'.format(filePath))
fNeuro = sio.loadmat(filePath)
Gs = fNeuro['G'].flatten()
fitting = fNeuro['fitting'].flatten()
# FCDfitt_PLA = fNeuro['FCDfitt_PLA'].flatten()

# mFCDfitt5   = np.mean(FCDfitt5,2);
# stdFCDfitt5 = np.std(FCDfitt5,[],2);
# mfitting5   = np.mean(fitting5,2);
# stdfitting5 = np.std(fitting5,[],2);

minFC = Gs[np.argmin(fitting)]
# minFCD = WEs[np.argmin(FCDfitt_PLA)]
print("\n\n#####################################################################################################")
print(f"# Max FC({minFC}) = {np.min(fitting)}")  #             ** Min FCD({minFCD}) = {np.min(FCDfitt_PLA)} **")
print("#####################################################################################################\n\n")

plt.rcParams.update({'font.size': 15})
# plotFCDpla, = plt.plot(WEs, FCDfitt_PLA)
# plotFCDpla.set_label("FCD placebo")
plotFC, = plt.plot(Gs, fitting)
plotFC.set_label("FC placebo")
plt.title("Whole-brain fitting")
plt.ylabel("Fitting")
plt.xlabel("Global Coupling G")
plt.legend()
plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
