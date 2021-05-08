# ==========================================================================
# ==========================================================================
#  Computes, as a pre-process, the optimization for G
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN (slurm.sbatch_genes_balanced_G_optimization.m)
#
#  Taken from the code (slurm.sbatch_genes_balanced_G_optimization.m) from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito, "Dynamical consequences of regional heterogeneity
#  in the brain’s transcriptional landscape", 2021, biorXiv
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007

import functions.Observables.FC as FC
import functions.Observables.GBC as GBC
import functions.Observables.swFCD as swFCD

import functions.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

import functions.Optimizers.Optim1D as G_optim
G_optim.simulateBOLD = simulateBOLD
G_optim.integrator = integrator

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2       # 2nd order butterworth filter
filters.flp = .008  # lowpass frequency of filter
filters.fhi = .08   # highpass
filters.TR = 0.754  # sampling interval
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


baseInPath = 'Data_Raw/DecoEtAl2020'
baseOutPath = 'Data_Produced/DecoEtAl2020'

N = 68
NSUB = 389
NumTrials = 15  # should be NSUB...
simulateBOLD.Tmax = 616
simulateBOLD.TR = 0.754  # sampling interval
simulateBOLD.dtt = 1e-3
simulateBOLD.Toffset = 14.
simulateBOLD.recomputeTmaxneuronal()


def transformEmpiricalSubjects(tc_aal, tcrange, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = tc_aal[:, tcrange, s].T
    return transformed


def prepro():
    # load genetic info
    print("Loading DKcortex_selectedGenes.mat")
    DKGenes = sio.loadmat(baseInPath+'/DKcortex_selectedGenes.mat')
    expMeasures = DKGenes['expMeasures']
    print(f'raw expMeasures from DKGenes shape is {expMeasures.shape}')

    coefei = np.sum(expMeasures[:,17:25],1) / np.sum(expMeasures[:,1:6],1)  # ampa+nmda/gaba
    ratioEI = np.zeros(N)
    ratioEI[:coefei.size] = coefei/(max(coefei)-min(coefei))
    ratioEI = ratioEI-max(ratioEI)+1
    ratioEI[34:68] = ratioEI[0:34]
    print(f'ratioEI shape is {ratioEI.shape}')

    # Read data..SC FC and time series of BOLD
    # =============================================================
    print('loading SC_GenCog_PROB_30.mat')
    GrCV = sio.loadmat(baseInPath+'/SC_GenCog_PROB_30.mat')['GrCV']
    print(f'Raw GrCV shape is {GrCV.shape}')
    tcrange = np.union1d(np.arange(0,34), np.arange(41,75))  # [1:34 42:75]
    C = GrCV[:, tcrange][tcrange, ]
    C = C/np.max(C)*0.2
    print(f'C shape is {C.shape}')
    # print(np.sum(C))
    # indexsub=1:NSUB;

    print('loading DKatlas_noGSR_timeseries.mat')
    ts = sio.loadmat(baseInPath+'/DKatlas_noGSR_timeseries.mat')['ts']
    print(f'ts shape is {ts.shape}')

    # tsdata = np.zeros((Tmax,N,NSUB))
    # FCdata = np.zeros((NSUB,N,N))
    # for nsub in np.arange(NSUB):
    #     # tsdata[:,:,nsub] = ts[:,range,nsub]
    #     FCdata[nsub,:,:] = FC.from_fMRI(ts[:,tcrange,nsub].T, applyFilters=False)
    #
    # FC_emp = np.squeeze(np.mean(FCdata,axis=0));
    # FCemp2 = FC_emp - np.multiply(FC_emp, np.eye(N))
    # GBC_emp = np.mean(FCemp2,1)

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'GBC': (GBC, False)}  #   'phFCD': (phFCD, True)
    swFCD.windowSize = 80
    swFCD.windowStep = 18

    tc_transf = transformEmpiricalSubjects(ts, tcrange, NSUB)
    print(f'tc_transf shape is {tc_transf[0].shape} for {NSUB} subjects')
    # FCemp_cotsampling = G_optim.processEmpiricalSubjects(tc_transf, distanceSettings, baseOutPath+"fNeuro_emp.mat")
    # FCemp = FCemp_cotsampling['FC']; cotsampling = FCemp_cotsampling['swFCD'].flatten(); GBCemp = FCemp_cotsampling['GBC'];

    J_fileNames = baseOutPath+"/J_Balance_we{}.mat"
    # baseGOptimNames = baseOutPath+"/fitting_we{}.mat"

    step = 0.001
    # WEs = np.arange(0, 3.+step, step)  # Range used in the original code
    WEs = np.arange(0, 3.+step, 0.05)  # reduced range for DEBUG only!!!

    # Model Simulations
    # ------------------------------------------
    BalanceFIC.verbose = True
    balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    fitting = G_optim.distanceForAll_Parms(C, tc_transf, balancedParms, NumSimSubjects=NumTrials,
                                           distanceSettings=distanceSettings,
                                           Parms=WEs,
                                           parmLabel='we',
                                           outFilePath=baseOutPath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    # ------------------------------------------
    # ------------------------------------------
    # numWEs = WEs.size
    # FCfitt = np.zeros((numWEs))
    # swFCDfitt = np.zeros((numWEs))
    # GBCfitt = np.zeros((numWEs))
    # for pos, we in enumerate(WEs):
    #     balancedParms = BalanceFIC.Balance_J9(we, C, J_fileNames.format(we))
    #     # Now we need to fix a "mysterious" problem: for some reason I do not know, an (X,) array (with X an
    #     # integer number) is saved and recovered as an (1,X) array. Why? I do not know...
    #     balancedParms['J'] = balancedParms['J'].flatten()
    #     balancedParms['we'] = balancedParms['we'].flatten()
    #     FCsimul_cotsamplingsim = G_optim.distanceForOne_G(we, C, balancedParms,
    #                                                       N, NumTrials,
    #                                                       distanceSettings,
    #                                                       baseName.format(np.round(we, decimals=3)))  # Jbal=Balance_J(we,C)
    #     FC_sim = FCsimul_cotsamplingsim['FC']
    #     swFCD_sim = FCsimul_cotsamplingsim['swFCD'].flatten()
    #     GBC_sim = FCsimul_cotsamplingsim['GBC']
    #
    #     swFCDfitt[pos] = swFCD.distance(cotsampling, swFCD_sim)
    #     FCfitt[pos] = FC.distance(FCemp, FC_sim)
    #     GBCfitt[pos] = GBC.distance(GBCemp, GBC_sim)
    #     print("{}/{}: swFCDfitt = {}; FCfitt = {}; GBCfitt = {}\n".format(we, wEnd, swFCDfitt[pos], FCfitt[pos], GBCfitt[pos]))
    # ------------------------------------------
    # ------------------------------------------

    filePath = baseOutPath+'/DecoEtAl2020_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'we': WEs,
                 'swFCDfitt': fitting['swFCD'],  # swFCDfitt,
                 'FCfitt': fitting['FC'],  # FCfitt,
                 'GBCfitt': fitting['GBC'],  # GBCfitt
                })
    print(f"DONE!!! (file: {filePath})")


if __name__ == '__main__':
    prepro()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
