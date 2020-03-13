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

import functions.simulateFCD as simulateFCD
import functions.FCD as FCD
from functions import BalanceFIC
BalanceFIC.integrator = integrator


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
def computeAvgSCMatrix(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    print("HC: {} (0)".format(HC[0]))
    sc_folder = baseFolder+'/'+HC[0]+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
    SCnorm = SC / SC.max()
    sumMatrix = SCnorm
    for subject in HC[1:]:
        print("HC: {}".format(subject))
        sc_folder = baseFolder+'/'+subject+"/DWI_processing"
        SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
        sumMatrix += SC
    return sumMatrix / len(HC)  # but we normalize it so we probably do not need this...


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
def loadSubjectData(subject):
    sc_folder = base_folder+'/connectomes/'+subject+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
    SCnorm = correctSC(SC)

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


# def thresholdSCMatrix(SC):
#     SC[SC > 0.05] = 0.05


normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
def correctSC(SC):
    logMatrix = np.log(SC+1)
    areasSC = logMatrix.shape[0]
    avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()
    finalMatrix = logMatrix * avgHuman66/avgSC  #* areasHuman66/areasSC
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std = FCD.characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print(" => impact=Avg*#:{}".format(avg*C.shape[0]))


# =====================================================================================
# Methods to check a few properties AD data
# =====================================================================================
def checkSubjectSCDiff(ax, subject, SCnorm, finalAvgMatrix):
    deltaSC = SCnorm - finalAvgMatrix
    plotSC(ax, deltaSC, subject)


def checkSubjectSC(ax, subject):
    SCnorm, abeta, fullSeries = loadSubjectData(subject)
    plotSC(ax, SCnorm, subject)


def comparteTwoSC_WRT_Ref(subjectA, subjectB, refSC=None):
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


def plotFC_for_G(subject):
    # First, load the empirical data
    SCnorm, abeta, fMRI = loadSubjectData(subject)
    empFC = FCD.FC_from_fMRI(fMRI)
    empFCD = FCD.FCD(fMRI)

    # Set the interval of G values to compute
    wStart = 0
    wStep = 0.05  # 0.05
    wEnd = 6 + wStep
    wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.

    # now set some simulation variables we need to function...
    # simulateFCD.Tmax = 20; simulateFCD.recomputeTmaxneuronal()
    integrator.neuronalModel.initJ(SCnorm.shape[0])

    currentValFC = np.inf; currentWeFC = -1
    ccFCempFCsim = np.zeros(len(wes))
    currentValFCD = np.inf; currentWeFCD = -1
    ksFCDempFCDsim = np.zeros(len(wes))
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {} >> ".format(we), end='')
        DMF.we = we
        simBDS = simulateFCD.simulateSingleSubject(SCnorm, warmup=True)

        simFC = FCD.FC_from_fMRI(simBDS.T)
        ccFCempFCsim[kk] = FCD.FC_Similarity(empFC, simFC)
        print('                 -> cc[FC_emp, FC_sim] = {}'.format(ccFCempFCsim[kk]), end='')
        if ccFCempFCsim[kk] < currentValFC:
            currentValFC = ccFCempFCsim[kk]
            currentWeFC = we
            print(" >> new min FC !!!")
        else:
            print()

        simFCD = FCD.FCD(simBDS)
        ksFCDempFCDsim[kk] = FCD.KolmogorovSmirnovStatistic(empFCD, simFCD)
        print('                 -> KS[FCD_emp, FCD_sim] = {}'.format(ksFCDempFCDsim[kk]), end='')
        if ksFCDempFCDsim[kk] < currentValFCD:
            currentValFCD = ccFCempFCsim[kk]
            currentWeFCD = we
            print(" >> new min FCD !!!")
        else:
            print()

    plt.plot(wes, ccFCempFCsim, label="FC")
    plt.plot(wes, ksFCDempFCDsim, label="FCD")
    # for line, color in zip([1.47, 4.45], ['r','b']):
    plt.axvline(x=currentWeFC, label='min FC at {}'.format(currentWeFC), c='g')
    plt.axvline(x=currentWeFCD, label='min FCD at {}'.format(currentWeFCD), c='r')
    plt.title("Large-scale network (DMF)")
    plt.ylabel("cc[FC(D)emp,FC(D)sim]")
    plt.xlabel("Global Coupling (G = we)")
    plt.legend()
    plt.show()


def plot_cc_empSC_empFC(subjects):
    results = []
    for subject in subjects:
        empSCnorm, abeta, fMRI = loadSubjectData(subject)
        empFC = FCD.FC_from_fMRI(fMRI)
        corr_SC_FCemp = FCD.pearson_r(empFC, empSCnorm)
        print("{} -> Pearson_r(SCnorm, empFC) = {}".format(subject, corr_SC_FCemp))
        results.append(corr_SC_FCemp)

    plt.figure()
    n, bins, patches = plt.hist(results, bins=6, color='#0504aa', alpha=0.7) #, histtype='step')  #, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('SC weights')
    plt.ylabel('Counts')
    plt.title("SC histogram", fontweight="bold", fontsize="18")
    plt.show()


# =====================================================================================
# Test the FIC computation from Deco et al. 2014
# =====================================================================================
def testDeco2014_Fig2(subject):
    import numpy as np
    # import scipy.io as sio
    # import matplotlib.pyplot as plt
    import functions.Models.DynamicMeanField as DMF
    import functions.Integrator_EulerMaruyama as integrator
    integrator.neuronalModel = DMF
    import DecoEtAl2014_Fig2 as DecoEtAl2014

    integrator.verbose = False

    print("=============================================")
    print("= Testing with Deco et al 2014, figure 2 !!!")
    plt.rcParams.update({'font.size': 15})

    # Load connectome:
    # --------------------------------
    # Original code
    # CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
    # C = np.log(CFile['C'] + 1)
    # ================================ Directly loading
    # sc_folder = base_folder+'/connectomes/'+subject+"/DWI_processing"
    # SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
    # SClog = np.log(SC+1)  # SCnorm = normalizationFactor * SCnorm / np.max(SCnorm)
    # areasSC = SClog.shape[0]
    # avgSC = np.average(SClog)
    # # === Normalization ===
    # # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()
    # finalMatrix = SClog * avgHuman66/avgSC * areasHuman66/areasSC
    SCnorm, abeta, fMRI = loadSubjectData(subject)
    DecoEtAl2014.filePath = 'Data_Produced/AD/FICWeights/BenjiBalancedWeights-'+subject+'-{}.mat'
    DMF.initJ(SCnorm.shape[0])

    # BalanceFIC.veryVerbose = True
    DecoEtAl2014.plotMaxFrecForAllWe(SCnorm)

# =====================================================================================
# Methods to simulate and fit AD data
# =====================================================================================
def preComputeJ_Balance(subject, SC):
    # fileName = "Data_Produced/AD/"+subject+"-"+str(neuronalModel.we)+"_JBalance.mat"
    fileName = 'Data_Produced/AD/FICWeights/BenjiBalancedWeights-'+subject+'-{}.mat'.format(neuronalModel.we)
    if not Path(fileName).is_file():
        print("Computing " + fileName + " !!!")
        BalanceFIC.verbose = True
        J=BalanceFIC.JOptim(SC).flatten()  # This is the Feedback Inhibitory Control
        sio.savemat(fileName, {'J': neuronalModel.J})  # save J_Balance J
    else:
        print("Loading "+fileName+" !!!")
        # ==== J can be calculated only once and then load J_Balance J
        J = sio.loadmat(fileName)['J'].flatten()
    return J


def singleSubjectPipeline(subject):
    SCnorm, abeta, fMRI = loadSubjectData(subject)
    empCC = FCD.FCD(fMRI)  # avgFC to get the average FC
    neuronalModel.we = 0.1  # 2.1  # right now, the standard magical value...

    print("Pre-computing J (FIC): Subject {} @ G={}".format(subject, neuronalModel.we))
    neuronalModel.J = preComputeJ_Balance(subject, SCnorm)

    simBDS = simulateFCD.simulateSingleSubject(SCnorm, warmup=True)
    simCC = FCD.FCD(simBDS.T)  # avgFC to get the average FC
    ccFCempFCsim = FCD.KolmogorovSmirnovStatistic(empCC, simCC)  # FC_Similarity for FC comparison
    print("cc[FCemp,FCsim]", ccFCempFCsim)


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
visualizeAll = True
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22})
    # ------------------------------------------------
    # Load individual Abeta PET SUVRs
    # ------------------------------------------------
    subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    # plotAllAbetaHistograms(subjects)
    classification = getClassifications(subjects)
    HCSubjects = [s for s in classification if classification[s] == 'HC']
    ADSubjects = [s for s in classification if classification[s] == 'AD']

    # avgSCMatrix = computeAvgSCMatrix(classification, base_folder + "/connectomes")
    # finalMatrix = correctSC(avgSCMatrix)
    # print("# of elements in AVG connectome: {}".format(finalMatrix.shape))
    # plotSCHistogram(finalMatrix)
    # plot_cc_empSC_empFC(HCSubjects)

    HCSubject = HCSubjects[0]
    SCnorm_HC, abeta_HC, fMRI_HC = loadSubjectData(HCSubject)
    analyzeMatrix("SC norm HC", SCnorm_HC)
    # plotSC.plotSC_and_Histogram("SC norm HC", SCnorm_HC)
    empFC = FCD.FC_from_fMRI(fMRI_HC)
    analyzeMatrix("EmpiricalFC", empFC)
    # plotSC.plotSC_and_Histogram("EmpiricalFC", empFC)
    # # sio.savemat(save_folder+'/empFC_{}.mat'.format(HCSubject), {'SC': SCnorm_HC, 'FC': empFC})

    corr_SC_FCemp = FCD.pearson_r(empFC, SCnorm_HC)
    print("Pearson_r(SCnorm, empFC) = {}".format(corr_SC_FCemp))

    # =====================================
    # Print Human 66 info as reference
    # =====================================
    CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    analyzeMatrix("Human 66", C)
    # plotSC.plotSC_and_Histogram("Human 66", C)

    # ADSubject = ADSubjects[0]
    # print("Processing Subjects {} and {}".format(HCSubject, ADSubject))
    # checkSubjectVsAvgSC(HCSubject)
    # comparteTwoSC_WRT_Ref(HCSubject, ADSubject)
    # print("# of elements in {}'s connectome: {}".format(HCSubject, SCnorm_HC.shape))
    # print("# of elements in "+modality+": {}".format(abeta_HC.shape))
    # print("# of elements in fMRI: {}".format(fMRI_HC.shape))
    # plotSC.justPlotSC(HCSubject, SCnorm_HC, plotSC.plotSC)  # plotSC.plotSCHistogram
    # plotSC.plotSC_and_Histogram(HCSubject, SCnorm_HC)

    # Compute the FIC params for all G
    testDeco2014_Fig2(HCSubject)

    # Determine optimal G to work with
    # plotFC_for_G(HCSubject)

    # Configure and compute Simulation
    # ------------------------------------------------
    # singleSubjectPipeline(HCSubject)


    # ================================================
    # ================================================
    # ================================================
    # Xenia's test block
    # Code to perform tests 4 Xenia
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
    # corr2 = FCD.pearson_r(SCnorm, FC)
    # print("corrcoef(SC,FC)={}".format(corr0))
    # print("corrcoef(SCnorm,FC)={}".format(corr1))
    # print("FCD.pearson_r(SCnorm, FC)={}".format(corr2))

# -- eof
