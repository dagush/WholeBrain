# --------------------------------------------------------------------------------------
# Full pipeline for processing AD, MCI and HC subjects
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# By Gustavo Patow
#
# Pre-requisites:
#   Before executing this, be sure to have correctly configured the setup_AD.py file...
#
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio

from setup import *

import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.swFCD as swFCD
import WholeBrain.Observables.phFCD as phFCD
from WholeBrain.Utils.plotSC import plotSC


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def preprocessingPipeline(subject, SCnorm, all_fMRI,  #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                          WEs,  # wStart=0.0, wEnd=6.0, wStep=0.05,
                          plotMaxFrecForAllWe=False):
    fileName = save_folder + '/FICWeights-'+subject+'/BenjiBalancedWeights-{}.mat'

    print("###################################################################\n"*2+"#")
    print(f"## Pre-processing pipeline on {subject}!!!\n#")
    print("###################################################################\n"*2+"\n")
    print(f"# Compute BalanceFIC ({wStart} to {wEnd} with step {wStep})")
    print("###################################################################")
    # BalanceFIC.useDeterministicIntegrator = useDeterministicIntegrator
    BalanceFIC.verbose = True
    balancedParms = BalanceFIC.Balance_AllJ9(SCnorm, WEs=WEs,  # wStart=wStart, wEnd=wEnd, wStep=wStep,
                                             baseName=fileName)
    modelParms = [balancedParms[i] for i in balancedParms]

    # Let's plot it as a verification measure...
    if plotMaxFrecForAllWe:
        import Tests.DecoEtAl2014.fig2 as Fig2
        Fig2.plotMaxFrecForAllWe(SCnorm, wStart=WEs[0], wEnd=WEs[-1], wStep=WEs[1]-WEs[0],
                                 extraTitle='', precompute=False, fileName=fileName)  # We already precomputed everything, right?

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute ParmSweep")
    print("###################################################################\n")
    outFilePath = save_folder+'/'+subject+'-temp'
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, WEs, modelParms, NumSimSubjects=len(all_fMRI),
                                            distanceSettings=distanceSettings,
                                            parmLabel='we',
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["wStart=","wEnd=","wStep="])
    except getopt.GetoptError:
        print('Prepro.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
        sys.exit(2)
    wStart = 0.; wEnd = 5.5; wStep = 0.05
    for opt, arg in opts:
        if opt == '-h':
            print('Prepro.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
            sys.exit()
        elif opt in ("--wStart"):
            wStart = float(arg)
        elif opt in ("--wEnd"):
            wEnd = float(arg)
        elif opt in ("--wStep"):
            wStep = float(arg)
    print(f'Input values are: wStart={wStart}, wEnd={wEnd}, wStep={wStep}')
    return wStart, wEnd, wStep


visualizeAll = True
if __name__ == '__main__':
    import sys
    wStart, wEnd, wStep = processRangeValues(sys.argv[1:])  # Default: wStart = 0.; wEnd = 5.5; wStep = 0.05

    # --------------------------------------------------
    # Compute the average SC for the HC subjects
    # --------------------------------------------------
    avgSCMatrix = dataLoader.computeAvgSC_HC_Matrix(classification, base_folder + "/connectomes")
    dataLoader.analyzeMatrix("AvgHC", avgSCMatrix)
    finalAvgMatrixHC = dataLoader.correctSC(avgSCMatrix)
    sio.savemat(save_folder + '/AvgHC_SC.mat', {'SC':finalAvgMatrixHC})
    dataLoader.analyzeMatrix("AvgHC norm", finalAvgMatrixHC)
    print("# of elements in AVG connectome: {}".format(finalAvgMatrixHC.shape))
    # plotSC.justPlotSC('AVG<HC>', finalMatrix, plotSC.plotSCHistogram)
    # plot_cc_empSC_empFC(HCSubjects)
    neuronalModel.setParms({'SC': finalAvgMatrixHC})
    neuronalModel.couplingOp = Couplings.instantaneousDirectCoupling(finalAvgMatrixHC)

    # --------------------------------------------------
    # load all HC fMRI data
    # --------------------------------------------------
    all_HC_fMRI = dataLoader.load_fullCohort_fMRI(classification, base_folder)

    # Configure and compute Simulation
    # ------------------------------------------------
    subjectName = 'AvgHC'
    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}

    BalanceFIC.use_N_algorithm = False  # To make sure we use Gus' algorithm
    WEs = np.arange(wStart, wEnd+0.01, wStep)
    optimal = preprocessingPipeline(subjectName, finalAvgMatrixHC, all_HC_fMRI,
                                    distanceSettings,
                                    WEs=WEs,  # wStart=wStart, wEnd=wEnd+0.01, wStep=wStep,
                                    plotMaxFrecForAllWe=False)
    print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
