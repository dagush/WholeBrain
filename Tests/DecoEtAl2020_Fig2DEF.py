# ==========================================================================
# ==========================================================================
#  Plots Figure 2.D-E-F in the paper
#
# --------------------------------------------------------------------------
#
#  Taken from the code (read_gain.m) from:
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

def plotData(fig, DataFitt, title, pos, optim):
    Values = DataFitt.reshape(grid[0].shape)
    ax = fig.add_subplot(spec[pos*2])
    plot = ax.pcolormesh(Betas, Alphas, Values.T)
    ax.invert_yaxis()
    plt.colorbar(plot, ax=ax, cmap=plt.get_cmap(name))  # label="Number of regions"
    ax.set_ylabel("Bias (B)")
    ax.set_title(title)

    ax = fig.add_subplot(spec[pos*2+1])
    AOptim = np.where(np.isclose(Alphas, optim[0]))
    plot = ax.plot(Betas, (Values[:,AOptim]).flatten())
    ax.set_ylabel("Level")
    ax.set_xlabel("Heterogeneity Scaling (Z)")
    # return ax


# baseInPath = 'Data_Raw/DecoEtAl2020'
filePath = baseOutPath + '/DecoEtAl2020_fittingBZ.mat'
if not Path(filePath).is_file():
    import DecoEtAl2020_Fitting_genes_balanced_gain as fitting
    fitting.Fitting()

print('Loading {}'.format(filePath))
fNeuro = sio.loadmat(filePath)
Alphas = fNeuro['Alphas'].flatten()
Betas = fNeuro['Betas'].flatten()
swFCDfitt = fNeuro['swFCDfitt'].flatten()
FCfitt = fNeuro['FCfitt'].flatten()
GBCfitt = fNeuro['GBCfitt'].flatten()

grid = np.meshgrid(Alphas,Betas)
grid = np.round(grid[0],3), np.round(grid[1],3)
flatGrid = [a for a in np.nditer(grid)]

maxFC = flatGrid[np.argmax(FCfitt)]
minFCD = flatGrid[np.argmin(swFCDfitt)]
maxGBC = flatGrid[np.argmax(GBCfitt)]
print("\n\n#####################################################################################################")
print(f"# Max FC({maxFC}) = {np.max(FCfitt)}")
print(f"# Max swGBC({maxGBC}) = {np.max(GBCfitt)}")
print(f"# Min swFCD({minFCD}) = {np.min(swFCDfitt)}")
print("#####################################################################################################\n\n")

plt.rcParams.update({'font.size': 12})
# fig, axs = plt.subplots(3, 1)
fig = plt.figure(constrained_layout=True)
# widths = [2, 3, 1.5]
heights = [3, 1, 3, 1, 3, 1]
spec = fig.add_gridspec(ncols=1, nrows=6, # width_ratios=widths,
                        height_ratios=heights,
                        hspace=1)
name = 'YlGnBu'

plotData(fig, GBCfitt, "GBC", 0, maxGBC)

plotData(fig, FCfitt, "FC", 1, maxFC)

plotData(fig, swFCDfitt, "swFCD", 2, minFCD)

# axs[0].title("swFCD")
# plt.legend()
plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
