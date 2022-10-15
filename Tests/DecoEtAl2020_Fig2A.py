# ==========================================================================
# ==========================================================================
#  Plots Figura 2A in the paper
#
# --------------------------------------------------------------------------
#
#  Taken from the code (read_G.m) from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito, "Dynamical consequences of regional heterogeneity
#  in the brain’s transcriptional landscape", 2020, biorXiv
#  https://doi.org/10.1101/2020.10.28.359943
#
#  Code by Gustavo Deco and Kevin Aquino
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pathlib import Path

from DecoEtAl2020_Setup import *

# baseInPath = 'Data_Raw/DecoEtAl2020'
filePath = baseOutPath + '/DecoEtAl2020_fneuro.mat'
if not Path(filePath).is_file():
    import DecoEtAl2020_Prepro_G_Optim as prepro
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
