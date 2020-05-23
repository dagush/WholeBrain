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


def displayResults(gc_range, psp_baseline, psp_peak_freq, eeg_peak_freq):
    import matplotlib
    from matplotlib.gridspec import GridSpec

    # define colormap
    lower = plt.cm.jet(np.linspace(0,1,200))
    colors = np.vstack(([0,0,0,0],lower))
    tmap = matplotlib.colors.LinearSegmentedColormap.from_list('test', colors)

    # plot results
    plt.figure(figsize=(18, 4))
    grid = GridSpec(nrows=1, ncols=3)
    x_coord       = gc_range.repeat(379)
    x_coord_eeg   = gc_range.repeat(64)

    plt.suptitle("Diagnosis : "+DX, fontweight="bold", fontsize="18", y = 1.05)

    # plot psp frequency
    plt.subplot(grid[0,0])
    plt.hist2d(x_coord, psp_peak_freq.flatten(), bins=[len(gc_range),40], cmap=tmap,
              range=[[np.min(gc_range),np.max(gc_range)],[-1,14]] ) #, vmax=100)
    plt.colorbar(label="Number of regions")
    plt.grid()
    plt.ylabel(' Frequency in Hz')
    plt.xlabel(' global coupling ')

    # plot psp baseline
    plt.subplot(grid[0,1])
    plt.hist2d(x_coord, psp_baseline.flatten(), bins=[len(gc_range),40], cmap=tmap,
              range=[[np.min(gc_range),np.max(gc_range)],[-1,40]])#, vmax=100)
    plt.colorbar(label="Number of regions")
    plt.grid()
    plt.ylabel(' PSP in mV')
    plt.xlabel(' global coupling ')

    # plot eeg frequency
    plt.subplot(grid[0,2])
    plt.hist2d(x_coord_eeg, eeg_peak_freq.flatten(), bins=[len(gc_range),40], cmap=tmap,
              range=[[np.min(gc_range),np.max(gc_range)],[-1,14]] )#, vmax=100)
    plt.colorbar(label="Number of regions")
    plt.grid()
    plt.ylabel(' Frequency in Hz')
    plt.xlabel(' global coupling ')

    plt.tight_layout()

    plt.show()


# =====================================================================================
# Methods for visualizing AD parameters
# =====================================================================================
def plotAllAbetaHistograms(subjects):
    for subject in subjects:
        print("plotting subject: {}".format(subject))
        pet_path=base_folder+"/PET_loads/"+subject+"/PET_PVC_MG/" + modality
        RH_pet = np.loadtxt(pet_path+"/"+"L.Amyloid_load_MSMAll.pscalar.txt")
        LH_pet = np.loadtxt(pet_path+"/"+"R.Amyloid_load_MSMAll.pscalar.txt")
        subcort_pet = np.loadtxt(pet_path+"/"+"Amyloid_load.subcortical.txt")[-19:]
        abeta_burden = np.concatenate((LH_pet,RH_pet,subcort_pet))

        # plt.rcParams["figure.figsize"] = (7,5)
        # plt.rcParams["figure.dpi"] = 300
        plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        n, bins, patches = plt.hist(abeta_burden, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Abeta SUVR')
        plt.ylabel('Regions')
        plt.suptitle("Abeta histogram ({})".format(subject), fontweight="bold", fontsize="18")
        # plt.show()
        plt.savefig("./Results/Abeta/"+subject+".png", dpi=200)
        plt.close()


# =====================================================================================
# Methods to input AD data
# =====================================================================================
def computeAvgSC_HC_Matrix(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    print("SC + HC: {} (0)".format(HC[0]))
    sc_folder = baseFolder+'/'+HC[0]+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")

    sumMatrix = SC
    for subject in HC[1:]:
        print("SC + HC: {}".format(subject))
        sc_folder = baseFolder+'/'+subject+"/DWI_processing"
        SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
        sumMatrix += SC
    return sumMatrix / len(HC)  # but we normalize it afterwards, so we probably do not need this...


def load_all_HC_fMRI(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    all_fMRI = {}
    for subject in HC:
        print("fMRI HC: {}".format(subject))
        fMRI_path = base_folder + "/fMRI/" + subject + "/MNINonLinear/Results/Restingstate"
        series = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
        subcSeries = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
        fullSeries = np.concatenate((series,subcSeries))
        all_fMRI[subject] = fullSeries
    return all_fMRI


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


def loadXData(dataset=1):
    x_path = "Data_Raw/from_Xenia/"
    if dataset == 0:
        xfile = 'sc_fromXenia.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(xfile), [k for k in M.keys()])
        mat0 = M['mat_zero']; print('mat_zero.shape={}'.format(mat0.shape))
        SCnorm = correctSC(mat0)
        fc = M['fc']; print('fc.shape={}'.format(fc.shape))
        ts = M['timeseries']; print('timeseries.shape={}'.format(ts.shape))
        return SCnorm, fc, ts
    elif dataset == 1:
        xfile = '002_S_0413-reduced-sc.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(xfile), [k for k in M.keys()])
        xfile_ts = '002_S_0413-reduced-timeseries.mat'
        ts = sio.loadmat(x_path + xfile_ts); print('{} File contents:'.format(xfile_ts), [k for k in ts.keys()])

        mat0 = M['mat_zero']; print('mat_zero.shape={}'.format(mat0.shape))
        SCnorm = correctSC(mat0)
        mat = M['mat']; print('mat.shape={}'.format(mat.shape))
        FC = ts['FC']; print('FC.shape={}'.format(FC.shape))
        timeseries = ts['timeseries']; print('timeseries.shape={}'.format(timeseries.shape))
        return SCnorm, FC, timeseries
    else:
        xfile = 'timeseries.mat'
        ts = sio.loadmat(x_path + xfile); print('{} File contents:'.format(xfile), [k for k in ts.keys()])
        timeseries = ts['timeseries']; print('timeseries.shape={}'.format(timeseries.shape))
        return None, None, timeseries


# def thresholdSCMatrix(SC):
#     SC[SC > 0.05] = 0.05


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


# =====================================================================================
# Methods to check a few properties AD data
# =====================================================================================
def checkSubjectSCDiff(ax, subject, SCnorm, finalAvgMatrix):
    deltaSC = SCnorm - finalAvgMatrix
    plotSC(ax, deltaSC, subject)


def checkSubjectSC(ax, subject):
    SCnorm, abeta, fullSeries = loadSubjectData(subject)
    plotSC(ax, SCnorm, subject)


def compareTwoSC_WRT_Ref(subjectA, subjectB, refSC=None):
    SCnormA, abetaA, fullSeriesA = loadSubjectData(subjectA)
    SCnormB, abetaB, fullSeriesB = loadSubjectData(subjectB)
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 2)
    ax1 = fig.add_subplot(grid[0,0])
    if refSC is not None:
        checkSubjectSCDiff(ax1, subjectA, SCnormA, refSC)
    else:
        checkSubjectSC(ax1, subjectA)
    ax2 = fig.add_subplot(grid[0,1])
    if refSC is not None:
        checkSubjectSCDiff(ax2, subjectB, SCnormB, refSC)
    else:
        checkSubjectSC(ax2, subjectB)
    plt.suptitle("Structural Connectivity diff ({},{})".format(subjectA, subjectB), fontweight="bold", fontsize="18", y=1.05)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    img = ax1.get_images()[0]
    fig.colorbar(img, cax=cbar_ax)

    plt.show()


# def plotFC_for_G(SCnorm, fMRI):
#     # First, load the empirical data
#     # SCnorm, abeta, fMRI = loadSubjectData(subject)
#     empFC = FC.from_fMRI(fMRI)
#     empFCD = FCD.from_fMRI(fMRI)
#
#     # Set the interval of G values to compute
#     wStart = 0
#     wStep = 0.05  # 0.05
#     wEnd = 6 + wStep
#     wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.
#
#     # now set some simulation variables we need to function...
#     # simulateFCD.Tmax = 20; simulateFCD.recomputeTmaxneuronal()
#     integrator.neuronalModel.initJ(SCnorm.shape[0])
#
#     currentValFC = np.inf; currentWeFC = -1
#     ccFCempFCsim = np.zeros(len(wes))
#     currentValFCD = np.inf; currentWeFCD = -1
#     ksFCDempFCDsim = np.zeros(len(wes))
#     for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
#         print("Processing: {} >> ".format(we), end='')
#         DMF.we = we
#         simBDS = simulateFCD.simulateSingleSubject(SCnorm, warmup=True)
#
#         simFC = FC.from_fMRI(simBDS.T)
#         ccFCempFCsim[kk] = FC.FC_Similarity(empFC, simFC)
#         print('                 -> cc[FC_emp, FC_sim] = {}'.format(ccFCempFCsim[kk]), end='')
#         if ccFCempFCsim[kk] < currentValFC:
#             currentValFC = ccFCempFCsim[kk]
#             currentWeFC = we
#             print(" >> new min FC !!!")
#         else:
#             print()
#
#         simFCD = FCD.from_fMRI(simBDS)
#         ksFCDempFCDsim[kk] = FCD.KolmogorovSmirnovStatistic(empFCD, simFCD)
#         print('                 -> KS[FCD_emp, FCD_sim] = {}'.format(ksFCDempFCDsim[kk]), end='')
#         if ksFCDempFCDsim[kk] < currentValFCD:
#             currentValFCD = ccFCempFCsim[kk]
#             currentWeFCD = we
#             print(" >> new min FCD !!!")
#         else:
#             print()
#
#     plt.plot(wes, ccFCempFCsim, label="FC")
#     plt.plot(wes, ksFCDempFCDsim, label="FCD")
#     # for line, color in zip([1.47, 4.45], ['r','b']):
#     plt.axvline(x=currentWeFC, label='min FC at {}'.format(currentWeFC), c='g')
#     plt.axvline(x=currentWeFCD, label='min FCD at {}'.format(currentWeFCD), c='r')
#     plt.title("Large-scale network (DMF)")
#     plt.ylabel("cc[FC(D)emp,FC(D)sim]")
#     plt.xlabel("Global Coupling (G = we)")
#     plt.legend()
#     plt.show()


def plot_cc_empSC_empFC(subjects):
    results = []
    for subject in subjects:
        empSCnorm, abeta, fMRI = loadSubjectData(subject)
        empFC = FC.from_fMRI(fMRI)
        corr_SC_FCemp = FC.pearson_r(empFC, empSCnorm)
        print("{} -> Pearson_r(SCnorm, empFC) = {}".format(subject, corr_SC_FCemp))
        results.append(corr_SC_FCemp)

    plt.figure()
    n, bins, patches = plt.hist(results, bins=6, color='#0504aa', alpha=0.7) #, histtype='step')  #, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('SC weights')
    plt.ylabel('Counts')
    plt.title("SC histogram", fontweight="bold", fontsize="18")
    plt.show()


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def preprocessingPipeline(subject, SCnorm, all_fMRI,  #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
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
        print('AD_prepro.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
        sys.exit(2)
    wStart = 0.; wEnd = 6.0; wStep = 0.05
    for opt, arg in opts:
        if opt == '-h':
            print('AD_prepro.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
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

    # =====================================
    # Print Human 66 info (as reference)
    # =====================================
    CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    analyzeMatrix("Unnormalized Human 66", C)
    C = 0.2 * C / np.max(C)
    analyzeMatrix("        Norm Human 66", C)
    # plotSC.plotSC_and_Histogram("Human 66", C)

    # ################################################################################################
    # Result:
    # Human 66 => Shape:(66, 66), Max:0.19615559289837184, Min:0.0, Avg:0.0035127188987848714, Std:0.01523519221725181
    #          => impact=Avg*#:0.23183944731980152
    #          => maxNodeInputs:0.7135693141327057
    # ################################################################################################

    # --------------------------------------------------
    # Classify subject information into {HC, MCI, AD}
    # --------------------------------------------------
    subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    # plotAllAbetaHistograms(subjects)  # generates a series of histograms of the Abeta burden...
    classification = getClassifications(subjects)
    HCSubjects = [s for s in classification if classification[s] == 'HC']
    ADSubjects = [s for s in classification if classification[s] == 'AD']

    # --------------------------------------------------
    # Compute the average SC for the HC subjects
    # --------------------------------------------------
    avgSCMatrix = computeAvgSC_HC_Matrix(classification, base_folder + "/connectomes")
    analyzeMatrix("AvgHC", avgSCMatrix)
    finalAvgMatrixHC = correctSC(avgSCMatrix)
    sio.savemat('Data_Produced/AD/AvgHC_SC.mat', {'SC':finalAvgMatrixHC})
    analyzeMatrix("AvgHC norm", finalAvgMatrixHC)
    print("# of elements in AVG connectome: {}".format(finalAvgMatrixHC.shape))
    # plotSC.justPlotSC('AVG<HC>', finalMatrix, plotSC.plotSCHistogram)
    # plot_cc_empSC_empFC(HCSubjects)

    # --------------------------------------------------
    # load all HC fMRI data
    # --------------------------------------------------
    all_HC_fMRI = load_all_HC_fMRI(classification, base_folder)
    # print("# of elements in fMRI: {}".format(fMRI_HC.shape))
    # plotSC.justPlotSC(HCSubject, SCnorm_HCSubject, plotSC.plotSC)  # plotSC.plotSCHistogram
    # plotSC.plotSC_and_Histogram(HCSubject, SCnorm_HCSubject)

    # --------------------------------------------------
    # play a little bit with one HC subject...
    # --------------------------------------------------
    # HCSubject = '002_S_0413'  # this is HCSubjects[0]
    # SCnorm_HCSubject, abeta_HCSubject, fMRI_HCSubject = loadSubjectData(HCSubject)
    # analyzeMatrix("SC norm HC (log({}))".format(HCSubject), SCnorm_HCSubject)
    # # plotSC.plotSC_and_Histogram("SC norm HC", SCnorm_HCSubject)
    # empFC = FC.from_fMRI(fMRI_HCSubject)
    # analyzeMatrix("EmpiricalFC", empFC)
    # C norm HC (log(002_S_0413)) => Shape:(379, 379), Max:14.250680446001775, Min:0.0, Avg:3.513063979447963, Std:2.418758388149712
    #                             => impact=Avg*#:1331.451248210778
    #                             => maxNodeInputs:2655.478582698918
    # plotSC.plotSC_and_Histogram("EmpiricalFC", empFC)
    # # sio.savemat(save_folder+'/empFC_{}.mat'.format(HCSubject), {'SC': SCnorm_HCSubject, 'FC': empFC})
    # print("# of elements in {}'s connectome: {}".format(HCSubject, SCnorm_HCSubject.shape))
    # print("# of elements in "+modality+": {}".format(abeta_HCSubject.shape))

    # --------------------------------------------------
    # compute the G vs. MaxFiringRate for the subject
    # --------------------------------------------------
    # Also, compute the FIC params for all G
    # testDeco2014_Fig2(SCnorm_HC, HCSubject)

    # --------------------------------------------------
    # Just a quick check for an AD subject...
    # --------------------------------------------------
    # ADSubject = ADSubjects[0]
    # print("Processing Subjects {} and {}".format(HCSubject, ADSubject))
    # compareTwoSC_WRT_Ref(HCSubject, ADSubject)  # Plot the two SC at the same time, for comparison...

    # Configure and compute Simulation
    # ------------------------------------------------
    # subjectName = 'AvgHC-N'
    subjectName = 'AvgHC'
    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}

    # ----------- Plot whatever results we have collected ------------
    # quite useful to peep at intermediate results
    # G_optim.loadAndPlot(outFilePath='Data_Produced/AD/'+subjectName+'-temp', distanceSettings=distanceSettings)

    import functions.BalanceFIC as FIC; FIC.use_N_algorithm = False  # To make sure we use A algorithm
    optimal = preprocessingPipeline(subjectName, finalAvgMatrixHC, all_HC_fMRI,
                                    distanceSettings,
                                    wStart=wStart, wEnd=wEnd+0.01, wStep=wStep,
                                    precompute=False, plotMaxFrecForAllWe=False)
    print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

    #####################################################################################################
    # Results (in (0.0, 5.4)):
    # - Optimal FC = 0.2937951863392448 @ 3.1
    # - Optimal swFCD = 0.1480175847479508 @ 3.55
    # - Optimal phFCD = 0.030982182261673485 @ 3.15
    #####################################################################################################


    # ================================================
    # ================================================
    # ================================================
    # Xenia's test block
    # Code to perform tests 4 Xenia
    # ================================================
    # ================================================
    # sc_folder = base_folder+'/connectomes/'+HCSubject+"/DWI_processing"
    # SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
    # logMatrix = np.log(SC+1)
    # SCnorm = normalizationFactor * logMatrix / logMatrix.max()  # Normalization
    #
    # import functions.BOLDFilters as BOLDFilters
    # notUsedSC, abeta_HC, fMRI_HC = loadSubjectData(HCSubject)
    # signal_filt = BOLDFilters.BandPassFilter(fMRI_HC)
    # sfiltT = signal_filt.T
    # FC = np.corrcoef(sfiltT, rowvar=False)  # Pearson correlation coefficients
    #
    # corr0 = np.corrcoef(SC.flatten(), FC.flatten())
    # corr1 = np.corrcoef(SCnorm.flatten(), FC.flatten())
    # corr2 = FC.pearson_r(SCnorm, FC)
    # print("corrcoef(SC,FC)={}".format(corr0))
    # print("corrcoef(SCnorm,FC)={}".format(corr1))
    # print("FC.pearson_r(SCnorm, FC)={}".format(corr2))

    # analyzeMatrix(subject_X, SCnorm_X)
    # empFC = FC.from_fMRI(fMRI_X.T)
    # print("KS(empFC, FC(fMRI))=", FCD.KolmogorovSmirnovStatistic(FC_X, empFC))
    # =================================== do DMF FIC pre-processing
    # subject_X = '002_S_0413-reduced'
    # import DecoEtAl2014_Fig2 as DecoEtAl2014
    # SCnorm_X, FC_X, fMRI_X = loadXData(dataset=1)
    # DecoEtAl2014.filePath = 'Data_Produced/X/FICWeights/BenjiBalancedWeights-'+subject_X+'-{}.mat'
    # DMF.initJ(SCnorm_X.shape[0])
    # DecoEtAl2014.plotMaxFrecForAllWe(SCnorm_X)
    # =================================== test the phFCD code...
    # import functions.phaseFCD as phFCD
    # subject_X = 'timeseries'
    # SCnorm_X, FC_X, fMRI_X = loadXData(dataset=2)
    # print("Starting phFCD calculus")
    # # start_time = time.clock()
    # resFCD = FCD.from_fMRI(fMRI_X)
    # # print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))
    #
    # print("FCD sahpe={}".format(resFCD.shape))
    # # start_time = time.clock()
    # resu = phFCD.from_fMRI(fMRI_X)
    # # print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))
    # print("phFCD shape={}".format(resu.shape))
    # print(resu)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
