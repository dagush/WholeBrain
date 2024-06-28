# ==========================================================================
# ==========================================================================
#  Plots Figura 2A in the paper
#
# --------------------------------------------------------------------------
#
#  Taken from the code (read_G.m) from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito, "Dynamical consequences of regional heterogeneity
#  in the brain’s transcriptional landscape", Sci. Adv.7,eabf4752(2021).
#  DOI:10.1126/sciadv.abf4752
#
#  Before this, needs the results computed in
#   - prepro_G_Optim.py
#
#  Code by Gustavo Deco and Kevin Aquino
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import hdf5storage as sio
import matplotlib.pyplot as plt

from pathlib import Path

from setup import *

# baseInPath = 'Data_Raw/DecoEtAl2021'
filePath = baseOutPath + '/DecoEtAl2020_fneuro.mat'
if not Path(filePath).is_file():
    import prepro_G_Optim as prepro
    prepro.prepro()

print('Loading {}'.format(filePath))
fNeuro = sio.loadmat(filePath)
WEs = fNeuro['we'].flatten()
swFCDfitt = fNeuro['swFCDfitt'].flatten()
FCfitt = fNeuro['FCfitt'].flatten()
GBCfitt = fNeuro['GBCfitt'].flatten()

maxFC = WEs[np.argmax(FCfitt)]
minFCD = WEs[np.argmin(swFCDfitt)]
print("\n\n#####################################################################################################")
print(f"# Max FC({maxFC}) = {np.max(FCfitt)}             ** Min swFCD({minFCD}) = {np.min(swFCDfitt)} **")
print("#####################################################################################################\n\n")

plt.rcParams.update({'font.size': 15})
plotFCD, = plt.plot(WEs, swFCDfitt)
plotFCD.set_label("FCD")
plotFC, = plt.plot(WEs, FCfitt)
plotFC.set_label("FC")
plotGBC, = plt.plot(WEs, GBCfitt)
plotGBC.set_label("GBC")
plt.title("Whole-brain fitting")
plt.ylabel("Functional Fitting")
plt.xlabel("Global Coupling (G = we)")
plt.legend()
plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
