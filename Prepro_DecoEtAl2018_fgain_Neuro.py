# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Taken from the code (FCD_LSD_empirical.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
# from pathlib import Path
# from numba import jit
import time

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.simulateFCD as simulateFCD
simulateFCD.integrator = integrator
simulateFCD.BOLDModel = Stephan2007

import functions.FC as FC
import functions.FCD as FCD

import functions.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
BalanceFIC.baseName = "Data_Produced/SC90/J_Balance_we{}.mat"

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    # serotonin2A.recompileSignatures()
    integrator.recompileSignatures()


# @jit(nopython=True)
# def initRandom():
#     np.random.seed(3)  # originally set to 13


def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,90,2)
    even = np.arange(1,90,2)[::-1]  # sort 'descend'
    symLR = np.zeros((90,TC.shape[1]))
    symLR[0:45,:] = TC[odd,:]
    symLR[45:90,:] = TC[even,:]
    return symLR


def processEmpiricalSubjects(tc_aal, task, NumSubjects, N, Conditions):
    # Loop over subjects for a given task
    FCemp = np.zeros((NumSubjects, N, N))
    cotsampling = np.array([], dtype=np.float64)
    cond = Conditions[task]
    print("Task:", task, "(", cond, ")")
    for s in range(NumSubjects):
        print('   Subject: ', s)
        signal = LR_version_symm(tc_aal[s, cond])
        FCemp[s] = FC.from_fMRI(signal, applyFilters=False)
        cotsampling = np.concatenate((cotsampling, FCD.from_fMRI(signal)))
    return np.squeeze(np.mean(FCemp, axis=0)), cotsampling


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3A.
# Then, later on, we developed the module G_optim using this code as basis. Now, we could refactor it
# using G_optim, but here we compute two fittings in parallel (PLACEBO and LCD), so it would mean either
# duplicating the loops, by making two calls in a row; or generalizing G_optim, to be able to process
# several fittings simultaneously. By now, the second option is not needed and I see no reason for
# implementing the first one, with the resulting waste of computations (all the simulations would be
# repeated). By now, we stick with two different codes. Future improvements on G_optim may render
# this decision different.
def prepro_Fig3():
    # Load Structural Connectivity Matrix
    print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
    sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
    C = sc90/np.max(sc90[:])*0.2  # Normalization...

    # # Load Regional Drug Receptor Map
    # print('Loading Data_Raw/mean5HT2A_bindingaal.mat')
    # mean5HT2A_aalsymm = sio.loadmat('Data_Raw/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
    # # serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()

    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset
    print("Simulating {} subjects!".format(NumSubjects))
    Conditions = [4, 1]  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

    #load fMRI data
    print("Loading Data_Raw/LSDnew.mat")
    LSDnew = sio.loadmat('Data_Raw/LSDnew.mat')  #load LSDnew.mat tc_aal
    tc_aal = LSDnew['tc_aal']
    (N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
    print('tc_aal is {} and each entry has N={} regions and Tmax={}'.format(tc_aal.shape, N, Tmax))

    # TCs = np.zeros((len(Conditions), NumSubjects, N, Tmax))
    # N_windows = int(np.ceil((Tmax-FCD.windowSize) / 3))  # len(range(0,Tmax-30,3))

    FCemp5, cotsampling5 = processEmpiricalSubjects(tc_aal, 0, NumSubjects, N, Conditions)  # PLACEBO
    FCemp2, cotsampling2 = processEmpiricalSubjects(tc_aal, 1, NumSubjects, N, Conditions)  # LSD

    # %%%%%%%%%%%%%%% Set General Model Parameters
    # dtt   = 1e-3   # Sampling rate of simulated neuronal activity (seconds)
    # dt    = 0.1
    # DMF.J     = np.ones(N,1)
    # Tmaxneuronal = (Tmax+10)*2000;
    step = 0.025
    WEs = np.arange(0, 2.5+step, step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)
    numWEs = len(WEs)

    FCDfitt5 = np.zeros((numWEs))
    FCDfitt2 = np.zeros((numWEs))
    fitting5 = np.zeros((numWEs))
    fitting2 = np.zeros((numWEs))
    # Isubdiag = np.tril_indices(N, k=-1)

    # Model Simulations
    # -----------------
    # for we in WEs:  # Pre-processing, to accelerate latter on calculations.
    #     BalanceFIC.Balance_J9(we, C, warmUp=False)  # Computes (and sets) the optimized J for Feedback Inhibition Control [DecoEtAl2014]
    for pos, we in enumerate(WEs):  # iteration over values for G (we in this code)
        # neuronalModel.we = we
        baseName = "Data_Produced/SC90/J_Balance_we{}.mat".format(np.round(we, decimals=3))
        neuronalModel.J = BalanceFIC.Balance_J9(we, C, baseName)['J'].flatten()  # Computes (and sets) the optimized J for Feedback Inhibition Control [DecoEtAl2014]
        integrator.recompileSignatures()
        FCs = np.zeros((NumSubjects, N, N))
        cotsamplingsim = np.array([], dtype=np.float64)
        # cotsamplingsim = simulateFCD.simulate(NumSubjects, C)
        start_time = time.clock()
        for nsub in range(NumSubjects):  # trials. Originally it was 20.
            print("we={} -> SIM subject {}!!!".format(we, nsub))
            bds = simulateFCD.simulateSingleSubject(C, warmup=False).T
            FCs[nsub] = FC.from_fMRI(bds, applyFilters=False)
            cotsamplingsim = np.concatenate((cotsamplingsim, FCD.from_fMRI(bds)))  # Compute the FCD correlations
        print("\n\n--- TOTAL TIME: {} seconds ---\n\n".format(time.clock() - start_time))

        FCDfitt5[pos] = FCD.KolmogorovSmirnovStatistic(cotsampling5, cotsamplingsim)
        FCDfitt2[pos] = FCD.KolmogorovSmirnovStatistic(cotsampling2, cotsamplingsim)

        FCsimul = np.squeeze(np.mean(FCs, axis=0))
        fitting5[pos] = FC.FC_Similarity(FCemp5, FCsimul)
        fitting2[pos] = FC.FC_Similarity(FCemp2, FCsimul)

    filePath = 'Data_Produced/DecoEtAl2018_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'we': WEs,
                 'fitting2': fitting2,
                 'fitting5': fitting5,
                 'FCDfitt2': FCDfitt2,
                 'FCDfitt5': FCDfitt5
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    print("DONE!!!")

if __name__ == '__main__':
    prepro_Fig3()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
