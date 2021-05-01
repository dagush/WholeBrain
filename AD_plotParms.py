# --------------------------------------------------------------------------------------
# Plotting for AD (MCI, HC) fitting
#
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from pathlib import Path

fontSize = 10
import functions.Utils.p_values as p_values
p_values.fontSize = fontSize
import AD_Auxiliar

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Utils.plotSC as plotSC
from functions.Models import Abeta_StefanovskiEtAl2019 as Abeta
# from functions.Models import JansenRit as JR
AD_modality = 'A'
if AD_modality == 'A':
    import functions.Models.AD_DMF_A as adDMF
else:
    import functions.Models.AD_DMF_B as adDMF
neuronalModel = adDMF

base_folder = "./Data_Raw/from_Ritter"
save_folder = "./Data_Produced/AD"

import functions.Integrator_EulerMaruyama
integrator = functions.Integrator_EulerMaruyama
integrator.neuronalModel = neuronalModel
integrator.verbose = False
# Integration parms...
# dt = 5e-5
# tmax = 20.
# ds = 1e-4
# Tmaxneuronal = int((tmax+dt))

# import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.BOLDHemModel_Stephan2008 as Stephan2008
import functions.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2008
from functions import BalanceFIC
BalanceFIC.integrator = integrator

import functions.Observables.FC as FC
import functions.Observables.swFCD as swFCD
import functions.Observables.phFCD as phFCD
import functions.Observables.indPhDyn as indPhDyn

import functions.G_optim as G_optim
G_optim.simulateBOLD = simulateBOLD
G_optim.integrator = integrator

import AD_functions
AD_functions.neuronalModel = neuronalModel
AD_functions.integrator = integrator
AD_functions.simulateBOLD = simulateBOLD
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


posA = 1; posB = 2; posC = 3
def plotParmComparisonAcrossGroups(dataAD, dataMCI, dataHC, selectedParms):  # plot comparison between the parms of AD, MCI and HC
    # print(result)
    fig = plt.figure()

    for pos, parm in enumerate(selectedParms):
        ax = fig.add_subplot(1,len(selectedParms),pos+1)
        points = {'AD': dataAD[:,parm], 'MCI': dataMCI[:,parm], 'HC': dataHC[:,parm]}
        positions = {'AD': posA, 'MCI': posB, 'HC': posC}
        p_values.plotMeanVars(ax, points, positions, title=f'Parm Comparison ({AD_functions.parmLabels[parm]})')
        test = p_values.computeWilcoxonTests(points)
        p_values.plotWilcoxonTest(ax, test, positions, plotOrder=['AD_MCI', 'MCI_HC', 'AD_HC'])
    plt.show()


def plotParms(allParms, groupName):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(allParms, labels=AD_functions.parmLabels)  # positions=[1,2,3,4,5,6],  # notch='True', patch_artist=True,
    ax.set_title(f'Optimized Parameters ({groupName})')
    plt.show()


def loadParms(subjects):
    results = {}
    optMethod = 'gp_minimize-cheat'
    print("# Optimized Parms:")
    print("#" * 104)
    for s in subjects:
        fileName = f'Data_Produced/AD/AD_{s}_fittingResult-{optMethod}.mat'
        if Path(fileName).is_file():
            optimizedParms = sio.loadmat(fileName)
            results[s] = optimizedParms['parms']
            print(f"#  subject {s} parms: {optimizedParms['parms']}")
        else:
            print(f"#  NOT processed: {s}")
    allParms = np.zeros((len(results),6))
    for pos, s in enumerate(results):
        allParms[pos,:] = results[s].flatten()
    return allParms


if __name__ == '__main__':
    import sys
    # aParmRange, bParmRange, studyParms = processParmValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 16})

    # ------------------------------------------------
    # Load individual classification
    # ------------------------------------------------
    subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    classification = AD_Auxiliar.checkClassifications(subjects)
    HCSubjects = [s for s in classification if classification[s] == 'HC']
    MCISubjects = [s for s in classification if classification[s] == 'MCI']
    ADSubjects = [s for s in classification if classification[s] == 'AD']
    print(f"We have {len(HCSubjects)} HC, {len(MCISubjects)} MCI and {len(ADSubjects)} AD \n")

    # -------------------------------------------------------
    # Plot all the parms for subjects classified as the label
    # -------------------------------------------------------
    # label = 'HC'  #AD, MCI, HC
    # subjectsToPlot = [s for s in classification if classification[s] == label]
    # subjectData = loadParms(subjectsToPlot)
    # plotParms(subjectData, label)

    # -------------------------------------------------------
    # Plot a comparison of some parms across all labels
    # -------------------------------------------------------
    # subjectsToPlot = [s for s in classification]
    AD_Data = loadParms(ADSubjects)
    MCI_Data = loadParms(MCISubjects)
    HC_Data = loadParms(HCSubjects)
    plotParmComparisonAcrossGroups(AD_Data, MCI_Data, HC_Data, selectedParms=[1,3,5])

    # ------------------------------------------------
    # Done !!!
    # ------------------------------------------------
    print("DONE !!!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
