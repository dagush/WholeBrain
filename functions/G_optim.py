# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Inspired from the code (fgain_Neuro.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from numba import jit
from functions.Utils.decorators import loadOrCompute
import time

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
# import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
# import functions.Integrator_EulerMaruyama as integrator
# integrator.neuronalModel = neuronalModel
# integrator.verbose = False
integrator = None
# import functions.BOLDHemModel_Stephan2007 as Stephan2007

import functions.FC as FC
import functions.swFCD as FCD

import functions.BalanceFIC as BalanceFIC
# BalanceFIC.integrator = integrator
# BalanceFIC.baseName = "Data_Produced/SC90/J_Balance_we{}.mat"

import functions.simulateFCD as simulateFCD

# set BOLD filter settings
# import functions.BOLDFilters as filters
# filters.k = 2                             # 2nd order butterworth filter
# filters.flp = .01                         # lowpass frequency of filter
# filters.fhi = .1                          # highpass
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# def recompileSignatures():
#     # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
#     # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
#     # print("\n\nRecompiling signatures!!!")
#     # serotonin2A.recompileSignatures()
#     integrator.recompileSignatures()


# def LR_version_symm(TC):
#     # returns a symmetrical LR version of the input matrix
#     N = TC.shape[0]  # 90 for AAL 90x90
#     odd = np.arange(0,N,2)
#     even = np.arange(1,N,2)[::-1]  # sort 'descend'
#     symLR = np.zeros((N,TC.shape[1]))
#     symLR[0:int(N/2.),:] = TC[odd,:]
#     symLR[int(N/2.):N,:] = TC[even,:]
#     return symLR


@loadOrCompute
def processEmpiricalSubjects(tc) :#, empiricalSubjectsFile=None):
    # if not Path(empiricalSubjectsFile).is_file():
    NumSubjects = len(tc)
    N = tc[next(iter(tc))].shape[0]  # get the first key to retrieve the value of N = number of areas
    FCemp = np.zeros((NumSubjects, N, N))
    cotsampling = np.array([], dtype=np.float64)
    # Loop over subjects
    for pos, s in enumerate(tc):
        print('   {}/{} Subject: {} ({}x{})'.format(pos, NumSubjects, s, tc[s].shape[0], tc[s].shape[1]), end='', flush=True)
        signal = tc[s]  # LR_version_symm(tc[s])
        start_time = time.clock()
        FCemp[pos] = FC.from_fMRI(signal, applyFilters=False)
        cotsampling = np.concatenate((cotsampling, FCD.from_fMRI(signal)))
        print(" -> computed in {} seconds".format(time.clock() - start_time))

    return {'FCemp': np.squeeze(np.mean(FCemp, axis=0)), 'cotsampling': cotsampling}


# ==========================================================================
# ==========================================================================
# ==========================================================================
# ---- convenience method, with the idea of parallelizing the code
@loadOrCompute
def distanceForOne_G(we, C, N, NumSimSubjects, J_fileNames):
    integrator.neuronalModel.J = BalanceFIC.Balance_J9(we, C, False, J_fileNames.format(np.round(we, decimals=3)))['J'].flatten()  # Computes (and sets) the optimized J for Feedback Inhibition Control [DecoEtAl2014]
    integrator.recompileSignatures()

    FCs = np.zeros((NumSimSubjects, N, N))
    cotsamplingsim = np.array([], dtype=np.float64)

    print("--- BEGIN TIME @ we={} ---".format(we))
    start_time = time.clock()
    for nsub in range(NumSimSubjects):  # trials. Originally it was 20.
        print("   we={} -> SIM subject {}/{}!!!".format(we, nsub, NumSimSubjects))
        bds = simulateFCD.simulateSingleSubject(C, warmup=False).T
        FCs[nsub] = FC.from_fMRI(bds, applyFilters=False)
        cotsamplingsim = np.concatenate((cotsamplingsim, FCD.from_fMRI(bds)))  # Compute the FCD correlations
    print("--- TOTAL TIME: {} seconds ---".format(time.clock() - start_time))
    FC_simul = np.squeeze(np.mean(FCs, axis=0))
    return {'FC_simul': FC_simul, 'cotsampling_sim': cotsamplingsim}


def distanceForAll_G(C, tc, NumSimSubjects,
                     wStart=0, wEnd=6.0, wStep=0.05,
                     J_fileNames=None,
                     outFilePath=None):
    # simulateFCD.BOLDModel = Stephan2007

    NumSubjects = len(tc)
    N = tc[next(iter(tc))].shape[0]  # get the first key to retrieve the value of N = number of areas
    print('tc({} subjects): each entry has N={} regions'.format(NumSubjects, N))

    processed = processEmpiricalSubjects(tc, outFilePath+'/fNeuro_emp.mat')
    FC_emp = processed['FCemp']
    cotsampling_emp = processed['cotsampling'].flatten()

    # %%%%%%%%%%%%%%% Set General Model Parameters
    # dtt   = 1e-3   # Sampling rate of simulated neuronal activity (seconds)
    # dt    = 0.1
    # DMF.J     = np.ones(N,1)
    # Tmaxneuronal = (Tmax+10)*2000;
    WEs = np.arange(wStart+wStep, wEnd, wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
    numWEs = len(WEs)

    FCDfitt = np.zeros((numWEs))
    FCfitt = np.zeros((numWEs))
    Isubdiag = np.tril_indices(N, k=-1)

    # Model Simulations
    # -----------------
    print(' ====================== Model Simulations ======================')
    for pos, we in enumerate(WEs):  # iteration over the values for G (we in this code)
        # ---- Perform the simulation of NumSimSubjects ----
        FC_simul_cotsampling_sim = distanceForOne_G(we, C, N, NumSimSubjects,
                                                    J_fileNames, outFilePath + '/fitting_{}.mat'.format(np.round(we, decimals=3)))
        FC_simul = FC_simul_cotsampling_sim['FC_simul']
        cotsampling_sim = FC_simul_cotsampling_sim['cotsampling_sim'].flatten()

        # ---- and now compute the final FC and FCD distances for this G (we)!!! ----
        FCDfitt[pos] = FCD.distance(cotsampling_emp, cotsampling_sim)
        # FCfitt[pos] = FC.distance(np.arctanh(FC_emp), np.arctanh(FC_simul))  # as in [Kringelbach2020]
        FCfitt[pos] = FC.distance(FC_emp, FC_simul)  # as in [Deco2018]
        print("{}/{}: FCDfitt = {}; FCfitt = {}\n".format(we, wEnd, FCDfitt[pos], FCfitt[pos]))

    # if outFilePath is not None:
    #     sio.savemat(outFilePath+'/fNeuro.mat',
    #                 {'we': WEs,
    #                  'fitting': FCfitt,
    #                  'FCDfitt': FCDfitt
    #                 })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    maxFC = WEs[np.argmax(FCfitt)]
    minFCD = WEs[np.argmin(FCDfitt)]
    print("\n\n#####################################################################################################")
    print(f"# Max FC({maxFC}) = {np.max(FCfitt)}             ** Min FCD({minFCD}) = {np.min(FCDfitt)} **")
    print("#####################################################################################################\n\n")
    print("DONE!!!")
    return FCfitt, FCDfitt, maxFC, minFCD


def plotFitting(fitting5, FCDfitt5, maxFC, minFCD, wStart=0.05, wEnd=6.0, wStep=0.05):
    WEs = np.arange(wStart+wStep, wEnd, wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
    # fitting2 = fNeuro['fitting2'].flatten()
    # fitting5 = fNeuro['fitting5'].flatten()
    # FCDfitt2 = fNeuro['FCDfitt2'].flatten()
    # FCDfitt5 = fNeuro['FCDfitt5'].flatten()

    # mFCDfitt5   = np.mean(FCDfitt5,2);
    # stdFCDfitt5 = np.std(FCDfitt5,[],2);
    # mfitting5   = np.mean(fitting5,2);
    # stdfitting5 = np.std(fitting5,[],2);

    plt.rcParams.update({'font.size': 22})
    plotFCDpla, = plt.plot(WEs, FCDfitt5, 'b')
    plt.axvline(x=minFCD, ls='--', c='b')
    plotFCDpla.set_label("FCD")
    plotFCpla, = plt.plot(WEs, fitting5, 'r')
    plt.axvline(x=maxFC, ls='--', c='r')
    plotFCpla.set_label("FC")
    plt.title("Whole-brain fitting")
    plt.ylabel("Fitting")
    plt.xlabel("Global Coupling (G)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load Structural Connectivity Matrix
    print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
    sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
    C = sc90/np.max(sc90[:])*0.2  # Normalization...

    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset
    print("Simulating {} subjects!".format(NumSubjects))
    Conditions = [4, 1]  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

    #load fMRI data
    print("Loading Data_Raw/LSDnew.mat")
    LSDnew = sio.loadmat('Data_Raw/LSDnew.mat')  #load LSDnew.mat tc_aal
    tc_aal = LSDnew['tc_aal']

    # distanceForAll_G(C, tc_aal, 'Data_Produced/error_{}.mat')
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
