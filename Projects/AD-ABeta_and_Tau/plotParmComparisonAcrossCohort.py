# --------------------------------------------------------------------------------------
# Plotting for AD (MCI, HC) fitting
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# The variable cohortType can either be 'MMSE' or 'AT(N)', resulting in the figures
# in the main paper for the first case, or the figures in the Supplemental material
# for the second.
#
# --------------------------------------------------------------------------------------
import numpy as np
import hdf5storage as sio
import matplotlib.pyplot as plt
import os
from pathlib import Path

fontSize = 10
import WholeBrain.Utils.p_values as p_values
p_values.fontSize = fontSize

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
cohortType = 'MMSE'  #'MMSE'/'AT(N)'
if cohortType == 'MMSE':
    from setup import *
elif cohortType == 'AT(N)':
    from setup_ATN import *
else:
    raise Exception(f'Uncknown cohortType: {cohortType}')
import functions_AD  # ============= we need functions_AD for the labels defined there...
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


posA = 1; posB = 2; posC = 3

# --------------------------------------------------------------------------
# plotParmComparisonAcrossGroups: plot comparison between some (selected) parms of AD, MCI and HC
# This method probably should go at p_values, but right now it is too specific to move there...
# --------------------------------------------------------------------------
def plotParmComparisonAcrossGroups(dataAD, dataMCI, dataHC, selectedParms, labels):
    # print(result)
    fig = plt.figure()

    for pos, parm in enumerate(selectedParms):
        ax = fig.add_subplot(1, len(selectedParms), pos+1)
        points = {labels[0]: dataAD[:,parm], labels[1]: dataMCI[:,parm], labels[2]: dataHC[:,parm]}
        positions = {labels[0]: posA, labels[1]: posB, labels[2]: posC}
        p_values.plotMeanVars(ax, points, positions, title=f'Parm Comparison ({functions_AD.parmLabels[parm]})')
        test = p_values.computeWilcoxonTests(points)
        p_values.plotSignificanceStars(ax, test, positions, plotOrder=[labels[0]+'_'+labels[1], labels[1]+'_'+labels[2], labels[0]+'_'+labels[2]])
    plt.show()


# --------------------------------------------------------------------------
# plotErrorComparisonAcrossBurden: plots the errors for the fittings
# across all burdens
# --------------------------------------------------------------------------
def plotErrorComparisonAcrossBurden():
    for condLabel in dataSetLabels:
        subjects = [s for s in classification if classification[s] == condLabel]
        print(f"Now, plotting values across all targets (ABeta+Tau, ABeta, Tau) for {condLabel}")
        AD_ABeta_Tau_values = loadValues(subjects, variant='')
        AD_Default = loadValues(subjects, parmToCollect='default')
        AD_ABeta_values = loadValues(subjects, variant='full_ABeta')
        AD_Tau_values = loadValues(subjects, variant='full_tau')
        # min, posMin, max, posMax = p_values.findMinMaxSpan(AD_ABeta_Tau_values, AD_Default)
        # print(f'Maximum span at: max={max}, posMax={posMax} for {subjects[posMax]}')
        dataToTest = {'ABeta+Tau': AD_ABeta_Tau_values, 'ABeta': AD_ABeta_values, 'Tau': AD_Tau_values,
                      'BEI': AD_Default}
        dataLabels = list(dataToTest.keys())
        p_values.plotComparisonAcrossLabels2(dataToTest,
                                             # AD_ABeta_Tau_values, AD_ABeta_values, AD_Tau_values, AD_Default,
                                             dataLabels,  # labels=['ABeta+Tau', 'ABeta', 'Tau', 'BEI'],
                                             graphLabel=f'Fitting Comparison ({condLabel})')


# --------------------------------------------------------------------------
# utility funcitons
# --------------------------------------------------------------------------
def plotParms(allParms, groupName, yLimits = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if yLimits is not None:
        ax.set_ylim(yLimits)
    ax.boxplot(allParms, labels=functions_AD.parmLabels)  # positions=[1,2,3,4,5,6],  # notch='True', patch_artist=True,
    ax.set_title(f'Optimized Parameters ({groupName})')
    plt.show()


showLoadingProgress = True
def loadParms(subjects, variant='', nullValues=[]):
    parms = {}
    optMethod = 'gp_minimize-cheat'
    variantString = '-' + variant if variant != '' else ''
    if showLoadingProgress:
        print("# Optimized Parms:")
        print("#" * 104)
    for s in subjects:
        fileName = save_folder + f'/AD_{s}_fittingResult-{optMethod}{variantString}.mat'
        if Path(fileName).is_file():
            optimizedParms = sio.loadmat(fileName)
            parms[s] = optimizedParms['parms']
            if showLoadingProgress:
                print(f"#  subject {s} parms: {optimizedParms['parms']}")
        else:
            if showLoadingProgress:
                print(f"#  NOT processed: {s}")
    allParms = np.zeros((len(parms),6))
    for pos, s in enumerate(parms):
        allParms[pos,:] = parms[s].flatten()
    allParms[:, nullValues] = 0.
    return allParms


def loadValues(subjects, parmToCollect='value', variant=''):
    values = {}
    optMethod = 'gp_minimize-cheat'
    variantString = '-' + variant if variant != '' else ''
    print(f"# Optimized Parms: {parmToCollect} @ {variant}")
    print("#" * 104)
    for s in subjects:
        fileName = save_folder + f'/AD_{s}_fittingResult-{optMethod}{variantString}.mat'
        if Path(fileName).is_file():
            optimizedParms = sio.loadmat(fileName)
            values[s] = optimizedParms[parmToCollect]
            print(f"#  subject {s} values: {optimizedParms[parmToCollect]}")
        else:
            print(f"#  NOT processed: {s}")
    allParms = np.zeros((len(values)))
    for pos, s in enumerate(values):
        allParms[pos] = float(values[s])
    return allParms


# =================================================================
# Loading and plotting parameters and errors
# Use True to select the graph to plot
# Meanings:
#   * When we say "across", it means the horizontal axis
#   * Cohort: HC, MCI, AD
#   * Error: phFCD
#   * BurdenTest: ABeta+Tau, ABeta, Tau
# =================================================================
plotParmValues = False  # For Figure 4 in the paper
printAsJSON = True  # if plotParmValues == True, prints the output as json, or normal otherwise
plotErrorComparisonAcrossBurdenTest = True  # for Figure 2D,E,F in the paper
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 12})

    # -------------------------------------------------------
    # Plot all the parms for subjects classified as the label
    # -------------------------------------------------------
    if plotParmValues:
        showLoadingProgress = False
        print("Plotting parameters for AD, MCI and HC")
        for label in dataSetLabels:
            print("\n\nNow, plotting for "+label)
            subjectsToPlot = [s for s in classification if classification[s] == label]
            subjectData = loadParms(subjectsToPlot)
            plotParms(subjectData, label, [-4.5, 4.5])

            print("\n*************************************")
            print(f"* Averaged over {len(subjectsToPlot)} subjects with {label}")
            for i in range(subjectData.shape[1]):
                if not printAsJSON:
                    print(f"* {functions_AD.parmLabels[i]}: " +
                          f"mean={np.mean(subjectData[:, i])} " +
                          f"std={np.std(subjectData[:, i])} "
                          )
                else:
                    print(f"'{functions_AD.parmLabels[i]}': " +
                          f"({np.mean(subjectData[:, i])}, " +
                          f"{np.std(subjectData[:, i])}),"
                          )
            print("*************************************\n")

    # -------------------------------------------------------
    # Plot a comparison of the ERROR VALUES across
    # Abeta+Tau, Abeta and Tau
    # -------------------------------------------------------
    if plotErrorComparisonAcrossBurdenTest:
        plotErrorComparisonAcrossBurden()

    # ------------------------------------------------
    # Done !!!
    # ------------------------------------------------
    print("DONE !!!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
