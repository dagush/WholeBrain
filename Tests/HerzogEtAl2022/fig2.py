# ================================================================================================================
#
# This code plots Figure 2 in [HerzogEtAl2022]
#
# see:
# [HerzogEtAl2022] Neural mass modelling for the masses: Democratising access to
#     whole-brain biophysical modelling with FastDMF, Rubén Herzog, Pedro A.M. Mediano,
#     Fernando E. Rosas, Andrea I. Luppi, Yonatan Sanz Perl, Enzo Tagliazucchi, Morten
#     Kringelbach, Rodrigo Cofré, Gustavo Deco, bioRxiv
#     doi: https://doi.org/10.1101/2022.04.11.487903
# [DecoEtAl2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#     How local excitation-inhibition ratio impacts the whole brain dynamics
#     J. Neurosci., 34 (2014), pp. 7886-7898
#     http://www.jneurosci.org/content/34/23/7886.long
#
# Observation: In my dataset, things look a lot messier than on the original manuscript. However,
# the linear patterns can be clearly observed, and they follow what is described there.
# Second, the original fit in [DecoEtAl2014] is for a frequency of 3Hz, while [HerzogEtAl2022]
# tune their model for 3.4Hz. Not a big difference, but...
#
# By Gustavo Patow
# ================================================================================================================
import os, re

import numpy as np
from scipy import stats
import scipy.io as sio
import matplotlib.pyplot as plt

# ============== chose a model
import Models.DynamicMeanField as DMF
# ============== chose and setup an integrator
import Integrators.EulerMaruyama as integrator
integrator.neuronalModel = DMF
integrator.verbose = False
# ============== chose a FIC mechanism
import Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
import Utils.FIC.Balance_Herzog2022 as Herzog2022Mechanism
import Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
BalanceFIC.balancingMechanism = Herzog2022Mechanism  # default behaviour for this project


np.random.seed(42)  # Fix the seed for debug purposes...

# Integration parms...
dt = 0.1
tmax = 10000.
Tmaxneuronal = int((tmax + dt))


def purgeTempFiles():
    dir = outFilePath + '/Human_66/'
    pattern = 'Herzog_Benji_Human66_*.mat'
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def plotGvsSlopeG(ax, C, wes):
    BalanceFIC.balancingMechanism = Deco2014Mechanism
    betas = np.sum(C, axis=0)
    res = np.zeros(len(wes))
    for pos, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        # if it was pre-computed, this simply loads the result...
        balancedJ = BalanceFIC.Balance_J9(we, C, fileNameDeco2014.format(np.round(we, decimals=2)))['J'].flatten()

        x = betas; y = balancedJ
        # obtain m (slope) and b(intercept) of linear regression line
        res[pos] = stats.linregress(x, y).slope
    ax.plot(wes, res)
    ax.plot(wes, 0.75 * wes * np.average(betas), label=r'$\alpha=0.75$')
    ax.plot(wes, 0.725 * wes * np.average(betas), label=r'$\alpha=0.725$')
    ax.set_xlabel('Global Coupling (G)')
    ax.set_ylabel('Slope(G)')
    ax.legend()


# Although this has been verified through other means (TVB C++), I cannot seem to reproduce it...
def plotFICvsBeta(ax, C, wes):
    BalanceFIC.balancingMechanism = Deco2014Mechanism
    betas = np.sum(C, axis=0)
    for we in wes[::5]:  # iterate over the weight range (G in the paper, we here)
        # if it was pre-computed, this simply loads the result...
        balancedJ = BalanceFIC.Balance_J9(we, C, fileNameDeco2014.format(np.round(we, decimals=2)))['J'].flatten()

        x = betas; y = balancedJ
        # create basic scatterplot
        ax.plot(x, y, '.')
        # obtain m (slope) and b(intercept) of linear regression line
        res = stats.linregress(x, y)
        # add linear regression line to scatterplot
        ax.plot(x, 1 + res.slope*x, label=f'G={np.round(we, decimals=1)}')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="8")


def plotMaxFreq(ax, wes, label, shuffle=False, averaging=False, fileName=None):
    print("======================================")
    print(f"=    simulating {label}             =")
    print("======================================")
    maxRate = np.zeros(len(wes))
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print(f"Processing: {we} ", end='')
        DMF.setParms({'we': we})
        # if it was pre-computed, this simply loads the result...
        balancedJ = BalanceFIC.Balance_J9(we, C, fileName.format(np.round(we, decimals=2)))['J'].flatten()
        if shuffle:
            np.random.shuffle(balancedJ)
        if averaging:
            balancedJ = np.average(balancedJ) * np.ones(C.shape[0])
        integrator.neuronalModel.setParms({'J': balancedJ})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
        maxRate[kk] = np.max(np.mean(v,0))
        print(f"MaxRate: {label} => {maxRate[kk]}")
    ee, = ax.plot(wes, maxRate)
    ee.set_label(label)


def plotMaxFrecForAllWe(ax, C, wes):
    DMF.setParms({'SC': C})
    N = C.shape[0]
    # DMF.lambda = 0.  # make sure no long-range feedforward inhibition (FFI) is computed

    # ======================================================
    # ================= FIC (alpha=0.725)
    BalanceFIC.balancingMechanism = Herzog2022Mechanism
    Herzog2022Mechanism.alpha = 0.725
    # BalanceFIC.Balance_AllJ9(C, wes, baseName=fileNameHerzog2022)
    plotMaxFreq(ax, wes, rf"$\alpha={Herzog2022Mechanism.alpha}$", fileName=fileNameHerzog2022)
    purgeTempFiles()  # we do not want to keep these files, computed with a=0.725

    # ================= FIC (alpha=0.75)
    Herzog2022Mechanism.alpha = 0.75  # default value
    BalanceFIC.balancingMechanism = Herzog2022Mechanism
    # BalanceFIC.Balance_AllJ9(C, wes, baseName=fileNameHerzog2022)
    plotMaxFreq(ax, wes, rf"$\alpha={Herzog2022Mechanism.alpha}$", fileName=fileNameHerzog2022)
    # purgeTempFiles()

    # # ======================================================
    # ================= Original [Deco2014] FIC $J^{opt}_n$
    # Always precompute:
    BalanceFIC.balancingMechanism = Deco2014Mechanism
    # BalanceFIC.Balance_AllJ9(C, wes, baseName=fileNameDeco2014)
    plotMaxFreq(ax, wes, r"$J^{opt}_n$", fileName=fileNameDeco2014)

    # ================= Average J^{opt}_n
    BalanceFIC.balancingMechanism = Deco2014Mechanism
    plotMaxFreq(ax, wes, r"Average $J^{opt}_n$", averaging=True, fileName=fileNameHerzog2022)
    # purgeTempFiles()

    # ================= Random J^{opt}_n
    BalanceFIC.balancingMechanism = Deco2014Mechanism
    plotMaxFreq(ax, wes, r"Randomized $J^{opt}_n$", shuffle=True, fileName=fileNameHerzog2022)
    # purgeTempFiles()

    # ======================================================
    # ================= finish plot! =======================
    for line, color in zip([1.47, 4.45], ['r','b']):
        plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    ax.set_ylabel("Maximum rate (Hz)")
    ax.set_xlabel("Global Coupling (G = we)")
    ax.legend()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})

    # Load connectome:
    # --------------------------------
    inFilePath = '../../Data_Raw'
    outFilePath = '../../Data_Produced'
    CFile = sio.loadmat(inFilePath + '/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']

    wStart = 0; wEnd = 6 + 0.001; wStep = 0.05
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.

    fileNameHerzog2022 = outFilePath + '/Human_66/Herzog_Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'
    fileNameDeco2014 = outFilePath + '/Human_66/Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'

    # ================================================================
    ax1 = plt.subplot(2, 1, 2)
    plotMaxFrecForAllWe(ax1, C, wes)

    ax2 = plt.subplot(2, 2, 1)
    plotFICvsBeta(ax2, C, wes)

    ax3 = plt.subplot(2, 2, 2)
    plotGvsSlopeG(ax3, C, wes)

    plt.show()



# ==========================================================================
# ==========================================================================
# ==========================================================================EOF