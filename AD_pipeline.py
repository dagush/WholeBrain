# --------------------------------------------------------------------------------------
# Full pipeline from:
# [StefanovskiEtAl2019] Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
#           R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
#           "Linking molecular pathways and large-scale computational modeling to assess candidate
#           disease mechanisms and pharmacodynamics in Alzheimer's disease." bioRxiv: 600205.
# Taken from the code at:
#           https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB/blob/master/Educase_AD_study-LS-Surrogate.ipynb
#
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import os, csv
from pathlib import Path
import matplotlib.pyplot as plt
import time
from functions.Utils.decorators import loadOrCompute

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Utils.plotSC as plotSC
from functions.Models import Abeta_StefanovskiEtAl2019 as Abeta
# from functions.Models import JansenRit as JR
import functions.Models.DynamicMeanField as DMF
neuronalModel = DMF

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

import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.simulateFCD as simulateFCD
simulateFCD.integrator = integrator
simulateFCD.BOLDModel = Stephan2007
from functions import BalanceFIC
BalanceFIC.integrator = integrator

import functions.FC as FC
import functions.swFCD as swFCD
import functions.phFCD as phFCD
import functions.G_optim as G_optim
G_optim.integrator = integrator
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# =====================================================================================
# Methods to input AD data
# =====================================================================================
def getClassifications(subjects):
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # They can be one of:
    # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
    # ============================================================================
    input_classification = csv.reader(open(base_folder+"/subjects.csv", 'r'))
    classification = dict(filter(None,input_classification))
    mistery = []
    for subject in subjects:
        if subject in classification:
            print('Subject {} classified as {}'.format(subject, classification[subject]))
        else:
            print('Subject {} NOT classified'.format(subject))
            mistery.append(subject)
    print("Misisng {} subjects:".format(len(mistery)), mistery)
    print()
    return classification


modality = "Amyloid" # Amyloid or Tau
def loadSubjectData(subject, correctSCMatrix=True):
    sc_folder = base_folder+'/connectomes/'+subject+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
    if correctSCMatrix:
        SCnorm = correctSC(SC)
    else:
        SCnorm = np.log(SC + 1)

    pet_path = base_folder+"/PET_loads/"+subject+"/PET_PVC_MG/" + modality
    RH_pet = np.loadtxt(pet_path+"/"+"L."+modality+"_load_MSMAll.pscalar.txt")
    LH_pet = np.loadtxt(pet_path+"/"+"R."+modality+"_load_MSMAll.pscalar.txt")
    subcort_pet = np.loadtxt(pet_path+"/"+modality+"_load.subcortical.txt")[-19:]
    abeta_burden = np.concatenate((LH_pet,RH_pet,subcort_pet))

    fMRI_path = base_folder+"/fMRI/"+subject+"/MNINonLinear/Results/Restingstate"
    series = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
    subcSeries = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
    fullSeries = np.concatenate((series,subcSeries))

    return SCnorm, abeta_burden, fullSeries


normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std, maxNodeInput, avgNodeInput = FC.characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print("  => impact=Avg*#:{}".format(avg*C.shape[0]), end='')
    print("  => maxNodeInputs:{}".format(maxNodeInput), end='')
    print("  => avgNodeInputs:{}".format(avgNodeInput))


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def preprocessingPipeline(subject, SCnorm, all_fMRI,  #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                          selectedDistance,
                          wStart=0.0, wEnd=6.0, wStep=0.05,
                          precompute=False, plotMaxFrecForAllWe=False):
    fileName = 'Data_Produced/AD/FICWeights-'+subject+'/BenjiBalancedWeights-{}.mat'
    # BalanceFIC.useDeterministicIntegrator = useDeterministicIntegrator
    if precompute:  # What's the point of this?
        BalanceFIC.verbose = True
        BalanceFIC.Balance_AllJ9(SCnorm, wStart=wStart, wEnd=wEnd, wStep=wStep, baseName=fileName)
        # Let's plot it as a verification measure...
    if plotMaxFrecForAllWe:
        import Fig_DecoEtAl2014_Fig2 as Fig2
        Fig2.plotMaxFrecForAllWe(SCnorm, wStart=wStart, wEnd=wEnd, wStep=wStep,
                                 extraTitle='', precompute=False, fileName=fileName)  # We already precomputed everything, right?

    # Now, optimize all we (G) values: determine optimal G to work with
    outFilePath = 'Data_Produced/AD/'+subject+'-temp'
    fitting = G_optim.distanceForAll_G(SCnorm, all_fMRI, NumSimSubjects=len(all_fMRI),
                                       distanceSettings=distanceSettings,
                                       wStart=wStart, wEnd=wEnd, wStep=wStep,
                                       J_fileNames=fileName,
                                       outFilePath=outFilePath)
    G_optim.loadAndPlot(outFilePath, distanceSettings)

    # optimal = distanceSettings[selectedDistance][0].findMinMax(fitting[selectedDistance])
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
        print('AD_pipeline.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
        sys.exit(2)
    wStart = 0.; wEnd = 6.0; wStep = 0.05
    for opt, arg in opts:
        if opt == '-h':
            print('AD_pipeline.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
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
    wStart, wEnd, wStep = processRangeValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 22})

    # ------------------------------------------------
    # Load individual classification
    # ------------------------------------------------
    subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    classification = getClassifications(subjects)
    HCSubjects = [s for s in classification if classification[s] == 'HC']
    ADSubjects = [s for s in classification if classification[s] == 'AD']

    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    AvgHC = sio.loadmat('Data_Produced/AD/AvgHC_SC.mat')['SC']
    analyzeMatrix("AvgHC norm", AvgHC)
    print("# of elements in AVG connectome: {}".format(AvgHC.shape))

    # ------------------------------------------------
    # Load individual Abeta and Tau PET SUVRs
    # ------------------------------------------------
    ADSubject = ADSubjects[0]
    ADSCnorm, ADabeta, ADfullSeries = loadSubjectData(ADSubject)
    analyzeMatrix("AD SC", ADSCnorm)
    print("# of elements in AD SCnorm connectome: {}".format(ADSCnorm.shape))

    # Configure and compute Simulation
    # ------------------------------------------------
    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}
    selectedDistance = 'swFCD'

    #####################################################################################################
    # Results:
    # - Optimal FC = 0.2937951863392448 @ 3.1
    # - Optimal swFCD = 0.1480175847479508 @ 3.55
    # - Optimal phFCD = 0.030982182261673485 @ 3.15
    #####################################################################################################


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
