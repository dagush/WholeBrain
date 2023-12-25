# --------------------------------------------------------------------------------------
# Full pipeline for AD subject processing
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# Pre-requisites:
#   Before executing this, run Prepro.py to get the 3.1 value for we (G in the paper)
#
# Note: I should refactor this to use AD_setup, which was created for this!
#
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import time
import dataLoader

# ==========================================================================
# Important config options: filenames
# ==========================================================================
base_folder = "../../Data_Raw/from_ADNI"
save_folder = "../../Data_Produced/AD_ABeta_and_Tau"

# ----------------------- Optimization methods...
optMethod = 'gp_minimize-cheat'
def getSaveFileName(subjectName, targetFunc):  # .mat files to save the final results to...
    if targetFunc == 'full_Abeta+Tau':  # This is done to avoid changing all the files generated so far... ;-)
        fileName = f'Data_Produced/AD/AD_{subjectName}_fittingResult-{optMethod}.mat'
    else:  # for newer optimization targets (Tau or Abeta only)
        fileName = f'Data_Produced/AD/AD_{subjectName}_fittingResult-{optMethod}-{targetFunc}.mat'
    print(f"saving to {fileName}")
    return fileName


def get_pkl_fileName(subjectName):  # .pkl filename needed for the gp_minimize method...
    checkpointPath = f"{save_folder}/temp/AD_{subjectName}-{targetFunc}-{optMethod}.pkl"
    print(f'saving to {checkpointPath}...')
    return checkpointPath


# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import DMF_AD as adDMF
neuronalModel = adDMF
import Models.Couplings as Couplings

import Integrators.EulerMaruyama as scheme
scheme.neuronalModel = neuronalModel
import Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = neuronalModel
integrator.verbose = False

# import WholeBrain.BOLDHemModel_Stephan2007 as Stephan2007
import Utils.BOLD.BOLDHemModel_Stephan2008 as Stephan2008
import Utils.simulate_SimAndBOLD as simulateBOLD
# simulateBOLD.TR = 3.
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2008
from Utils.FIC import BalanceFIC
BalanceFIC.integrator = integrator
import Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
BalanceFIC.balancingMechanism = Deco2014Mechanism  # default behaviour for this project

import WholeBrain.Observables.phFCD as phFCD

import WholeBrain.Optimizers.ParmSweep as G_optim
G_optim.simulateBOLD = simulateBOLD
G_optim.integrator = integrator

import functions_AD
functions_AD.neuronalModel = neuronalModel
functions_AD.integrator = integrator
functions_AD.simulateBOLD = simulateBOLD

from Utils.preprocessSignal import processEmpiricalSubjects
verbose = True
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------



# ==========================================================================================
#  WholeBrain to fit the model parms (!)
# ==========================================================================================
# First, select the target function
def selectFunction(targetFunc):
    if targetFunc == 'full_Abeta+Tau':
        return functions_AD.func6D
    elif targetFunc == 'full_tau':
        return functions_AD.func_full_Tau
    elif targetFunc == 'full_ABeta':
        return functions_AD.func_full_ABeta
    return None


# ========================================================================
# Simple function to evaluate at the default (or other) parms
# ========================================================================
plotObservableUsed = False  # True: compute the observable matrix for the default (without Amyloid or Tau)/selected state
                            # False: skip these parts of the code and do not plot anything!
computeObservableAtDefault = True  # True: use default, i.e., [0,...,0] parameters
                                    # False: load the params with the getSaveFileName function...
def evaluateAtDefault(subjectName):
    orig_trials = functions_AD.trials
    print("\n\n##################################################################")
    print(f"#  Evaluating {subjectName} at default parms!!!")
    print(f"#  Trials: {orig_trials}")
    print("##################################################################\n\n")
    parms = np.array([0, 0, 0, 0, 0, 0])
    if plotObservableUsed:
        import WholeBrain.Observables.phFCD as observable
        # ================ Change mode at observable to save the resulting matrix
        observable.saveMatrix = True
        functions_AD.trials = 1
    if not computeObservableAtDefault:
        fileName = getSaveFileName(subjectName, targetFunc)
        print(f'processing {subjectName} (from {fileName}) at non-default values')
        parms = sio.loadmat(fileName)['parms'].flatten()
    # ================ evaluate !!!
    # if not desperateContinuationForMCISubjects5and8:
    valueDefault = functions_AD.func6D(parms)  # test
    # ================ end evaluation!
    if plotObservableUsed:
        save_file = save_folder + "/"+observable.name+".mat"
        phIntMatr = sio.loadmat(save_file)
        import Utils.Plotting.plotSC as plotSC
        plotSC.plotFancyMatrix(phIntMatr[observable.name], axisName="Time", matrixName="FC dynamics (FCD)", showAxis='off')
        # ============= Restore state at observable, do not keep saving anything!
        observable.saveMatrix = False
        functions_AD.trials = orig_trials
    print(f"Result Default: {valueDefault}")
    return valueDefault  # To prevent needing to compute it again!


# now, the actual AD fitting!!!
# @loadOrCompute
def fit_AD_Parms(subjectToTest, # SCMatrix,
                 targetBOLDSeries, BOLD_length, distanceSetting,
                 Abeta, tau,
                 method, targetFunc):
    subjectName = subjectToTest
    measureName = list(distanceSetting.keys())[0]
    measure = distanceSetting[measureName][0]
    functions_AD.measure = measure
    functions_AD.applyFilters = distanceSetting[measureName][1]

    # Now we are going to preprocess the data for ONE subject, the one which
    # we want to fit the AD data to...
    print("Measuring empirical data from_fMRI...")
    outEmpFileName = save_folder + '/fNeuro_emp_AllHC.mat'
    functions_AD.processedEmp = processEmpiricalSubjects({subjectToTest: targetBOLDSeries},
                                                         distanceSettings,
                                                         outEmpFileName)[measureName]

    print("Starting optimization...")
    # Setting up simulation parms

    print("\n\n##################################################################")
    print("#  Modality: Direct Abeta and Tau influence ")  # + detailStr)
    print("##################################################################\n\n")
    adDMF.Abeta = Abeta
    adDMF.Tau = tau
    (N, Tmax) = targetBOLDSeries.shape
    functions_AD.N = N

    simulateBOLD.Tmax = BOLD_length
    simulateBOLD.recomputeTmaxneuronal()

    valueDefault = evaluateAtDefault(subjectName)

    print("\n\n##################################################################")
    print(f"#  Fitting {subjectName} with {method}!!!")
    print(f"#  Target function: {targetFunc}")
    print(f"#  Trials: {functions_AD.trials}")
    print("##################################################################\n\n")
    funcToUse = selectFunction(targetFunc)
    start_time = time.perf_counter()
    if method in ['gp_minimize', 'gp_minimize-cheat']:
        from skopt import gp_minimize
        import WholeBrain.Utils.decorators as decorators
        decorators.verbose = True
        # =============================================== Do the optimization!!!
        res = gp_minimize(funcToUse, dimensions=functions_AD.parmBounds,
                          n_calls=100, verbose=True,
                          )
        res.success = True  # Not really needed, but used later on...
        print("x*=%.2f f(x*)=%.2f" % (res.x[0], res.fun))


    print("\n\n --- TIME: {} seconds ---".format(time.perf_counter() - start_time), flush=True)
    obj = res
    print("success!!!") if obj.success else print("failure")
    print(f"\n\nFound: f={obj.fun} at ", obj.x)
    if verbose:
        print(f'min[f(x)])={min(obj.x)}  max[f(x)]={max(obj.x)}')
        if method in ['gp_minimize', 'gp_minimize-cheat']:
            print("\n\nModels:", res.models, "\n\n")
            print("\n\nx_iters:", res.x_iters, "\n\n")
            print("\n\nfunc_vals:", res.func_vals, "\n\n")
            print("\n\nspace:", res.space, "\n\n")
            print("\n\nspecs:", res.specs, "\n\n")

    return {'subject': subjectToTest, 'parms': obj.x, 'value': obj.fun, 'default': valueDefault}


# ====================================================================================
# Main method of this file: Computes the fitting pipeline for the AD disease subjects
# ====================================================================================
def AD_pipeline(subjectName,
                distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                AvgHC,
                targetFunc):
    N = AvgHC.shape[0]

    # ------------------------------------------------
    # Load individual Abeta and Tau PET SUVRs
    # ------------------------------------------------
    AD_SCnorm, AD_Abeta, AD_tau, AD_fullSeries = dataLoader.loadSubjectData(subjectName)
    AD_fullSeries = dataLoader.cutTimeSeriesIfNeeded(AD_fullSeries)
    dataLoader.analyzeMatrix("AD SC norm", AD_SCnorm)
    print("   # of elements in AD SCnorm connectome: {}".format(AD_SCnorm.shape))

    # ------------------------------------------------
    # Configure simulation
    # ------------------------------------------------
    we = 3.1  # Result from previous preprocessing using phFCD... (see Prepro.py)
    J_fileName = save_folder + f'/FICWeights-AvgHC/BenjiBalancedWeights-{we}.mat'
    neuronalModel.setParms({'SC': AD_SCnorm, 'we': we})  # neuronalModel.we = we
    neuronalModel.couplingOp = Couplings.instantaneousDirectCoupling(AD_SCnorm)
    neuronalModel.setParms({'J': sio.loadmat(J_fileName)['J'].flatten()})  # Loads the optimized J for Feedback Inhibition Control [DecoEtAl2014]
    neuronalModel.setParms({'M_e': np.ones(N)})
    neuronalModel.setParms({'M_i': np.ones(N)})
    integrator.recompileSignatures()


    # Fit params!!!
    # --------------------------------------------------------------
    # ----------------------- Error measures...
    result = fit_AD_Parms(subjectName,
                          AD_fullSeries, AD_fullSeries.shape[1],
                          distanceSettings,
                          AD_Abeta, AD_tau,
                          optMethod, targetFunc)
    return result

# This code was written when we had several running problems, related with Windows updates and such. Should NOT be used in normal functioning!!!!
# Buy the way: Why can't we disable automatic "urgent" Windows updates that restart the computer when needed?
emergencyPipeline = False
def AD_EmergencyPipeline():
    data = {'full_ABeta': [
                    # --target full_ABeta --group HC
                {'subject': '002_S_0413', 'parms': [-0.0504510501782609, 2.0814031290007446, -0.9996861754826295, -0.011873121748156468, -0.5265575144074548, -1.8419233203697725], 'value': 0.0831038961038961, 'default': 0.23348051948051946},
                {'subject': '002_S_1280', 'parms': [-0.5517283246650502, 0.0, 0.8894025562678678, -3.575226965265267, -0.8995134787734773, 0.0], 'value': 0.3259675324675324, 'default': 0.48542857142857143},
                {'subject': '002_S_4213', 'parms': [-0.4140448060822882, 1.9942002048016771, -0.9999631049491673, -0.08413029243886605, -0.9004884424560451, -3.6413446210212257], 'value': 0.09337012987012983, 'default': 0.6392272727272728},
                {'subject': '002_S_4799', 'parms': [0.2732728889690279, 2.741263405505976, 0.4563224416937244, -1.8433163600791977, -0.8611641602406015, -3.6007636808887535], 'value': 0.20283766233766232, 'default': 0.4978961038961039},
                {'subject': '002_S_5178', 'parms': [-0.479729527050363, 2.1364086128622586, -0.7803042426762834, -0.34601448812160385, 0.028459775160636802, -2.994475447287905], 'value': 0.0933051948051948, 'default': 0.19469480519480517},
                {'subject': '002_S_6009', 'parms': [0.6389732717240886, 3.870878808794171, -0.17514061840587392, -1.1596563614267468, -0.8488542254519003, -0.4177771318547654], 'value': 0.15642207792207785, 'default': 0.588590909090909},
                {'subject': '002_S_6030', 'parms': [1.0, 0.8138957460531946, -0.9806362598833864, 0.0, 0.8219301460094695, -0.7798375558889106], 'value': 0.07478571428571434, 'default': 0.333409090909091},
                {'subject': '002_S_6053', 'parms': [-0.847506916539725, 1.5080149947883223, 0.9887418639392973, -1.2178688887804991, 0.6893372318108, 0.0], 'value': 0.11785714285714288, 'default': 0.714935064935065},
                {'subject': '002_S_6103', 'parms': [-0.3173286717292717, 4.0, -0.9437579434400519, -0.031163471101380846, -0.8021294929031247, -1.1441940326946751], 'value': 0.12485714285714289, 'default': 0.7255909090909091},
                {'subject': '003_S_4288', 'parms': [-0.7001802770556695, 1.6762627281526203, 0.8436820543473555, -2.7055804624727235, -0.7071436138000311, -2.5523638905843224], 'value': 0.24542207792207793, 'default': 0.7286298701298701},
                {'subject': '003_S_4644', 'parms': [0.7181478787098441, 1.1075424687747155, -0.10457156654103095, -1.4528441684508735, -0.7865837154347471, -2.375447801569497], 'value': 0.3356038961038961, 'default': 0.7023051948051948},
                {'subject': '003_S_6067', 'parms': [0.450740259559282, 2.9669444509156597, -0.7076333253416889, -0.44127886511886993, -0.36086799214255705, -1.6754250596229951], 'value': 0.08787012987012988, 'default': 0.8989675324675325},
                {'subject': '007_S_4387', 'parms': [-0.09919249776648298, 3.37682576120494, -0.834859686753543, -0.2577545391642615, 0.9797109342339336, -0.873986933144121], 'value': 0.3838205467246331, 'default': 0.8443714166694519},
                {'subject': '007_S_4488', 'parms': [-0.20897325022998814, 3.132455162257536, -0.9711279691988888, -0.011698992978048217, -0.9118024071703539, -0.7216007231471977], 'value': 0.10022036998843231, 'default': 0.8075070780076743},
                {'subject': '007_S_4620', 'parms': [0.05567415084572702, 2.647576974568154, -0.3883063867842482, -0.7486904178671003, 0.6309736989250092, -3.3017665275004906], 'value': 0.16526778632202022, 'default': 0.8741011032154087},
                {'subject': '002_S_6007', 'parms': [-1.0, 1.402855423906969, 0.12394926592062028, 0.0, -1.0, -4.0], 'value': 0.18177272727272725, 'default': 0.5604025974025975},
                {'subject': '003_S_6014', 'parms': [0.7368745526471727, 0.342138471573239, 0.4692469951171274, -2.6462641754173912, -1.0, -0.8189961403108419], 'value': 0.2250844155844156, 'default': 0.7279155844155843},
                    # --target full_ABeta --group MCI
                {'subject': '002_S_1155', 'parms': [0.19862230399159708, 2.932936163469557, -0.38524560218228976, -0.8833539361691582, 0.11552391625790581, -1.7867966887827667], 'value': 0.2431168831168831, 'default': 0.1641363636363636},
                {'subject': '002_S_1261', 'parms': [-0.6328598873233878, 1.3039340419526795, 0.8720449082791653, -2.673600815020123, -0.2267952708555987, -3.9832922278061016], 'value': 0.25775324675324673, 'default': 0.3979350649350649},
                {'subject': '002_S_4229', 'parms': [-0.9701758077029716, 2.3587071868162024, -0.9239449355846533, -0.20202802650727758, 0.7519360978233018, -3.6059309644589743], 'value': 0.1526168831168831, 'default': 0.6143181818181819},
                {'subject': '002_S_4654', 'parms': [-0.9607399103888417, 1.6339517483925687, 0.8909807031157315, -1.2003534167617969, -0.7359399259845305, -3.593400630785322], 'value': 0.18088311688311687, 'default': 0.48797402597402595},
                {'subject': '003_S_1122', 'parms': [-0.8860829607267977, 0.0, 0.5925406516071989, -3.475692470790266, 0.2144211191304628, -0.7232045389432633], 'value': 0.3156558441558442, 'default': 0.5529155844155844},
                {'subject': '007_S_4272', 'parms': [-0.46540566051432675, 1.7979189948646292, 0.8673903595827923, -0.8685017799206753, 0.10637933866159677, -3.9445261311703006], 'value': 0.1437710801532909, 'default': 0.8622523838599415},
                {'subject': '012_S_6073', 'parms': [-1.0, 3.8409052413896547, -0.9586980037735287, 0.0, -0.17472580461330567, 0.0], 'value': 0.0728896103896104, 'default': 0.7563376623376623},
                {'subject': '022_S_5004', 'parms': [1.0, 0.6821770462235712, -0.46927832469653996, -0.8862607361520687, -0.1776375065940059, 0.0], 'value': 0.11162337662337662, 'default': 0.6127857142857143},
                {'subject': '023_S_4115', 'parms': [1.0, 4.0, 0.7554310396669599, -2.6124007430970324, -0.29853436214515383, 0.0], 'value': 0.2127036032826629, 'default': 0.8496277045407672},
                ],
            'full_tau': [
                    # --target full_tau --group HC
                {'subject': '002_S_0413', 'parms': [-0.31308842375915436, 0.4612904762318125, 1.0, 0.0, 1.0, 0.0], 'value': 0.17116883116883116, 'default': 0.2598701298701299},
                {'subject': '002_S_1280', 'parms': [-0.07638841842893651, 0.09806916463122588, -1.0, -4.0, -1.0, -4.0], 'value': 0.43613636363636366, 'default': 0.487461038961039},
                {'subject': '002_S_4213', 'parms': [-0.05662103536691088, 0.0, -1.0, 0.0, 0.7949483710780516, -4.0], 'value': 0.5918571428571429, 'default': 0.6544740259740259},
                {'subject': '002_S_4799', 'parms': [-0.08606282489505979, 0.053485425841701316, 0.29459222811812436, -4.0, 0.5958731041689453, -2.4371155661472894], 'value': 0.4305519480519481, 'default': 0.5237597402597403},
                {'subject': '002_S_5178', 'parms': [-0.014951021926867925, 0.0, 1.0, 0.0, -1.0, -3.373990754493446], 'value': 0.1379090909090909, 'default': 0.15757142857142858},
                {'subject': '002_S_6009', 'parms': [-0.5125896586043979, 0.6616381759762298, 0.5373920242003851, -0.7606384298947879, -0.587373066940935, -3.0194145074659877], 'value': 0.5097467532467532, 'default': 0.5861623376623376},
                {'subject': '002_S_6030', 'parms': [-0.0666016455969014, 0.09862880896980859, -0.9762853240923608, -2.501551071275351, -1.0, -4.0], 'value': 0.24577272727272728, 'default': 0.34172727272727277},
                {'subject': '002_S_6053', 'parms': [-1.0, 1.2741692852451838, -0.02421025808317867, 0.0, 1.0, -4.0], 'value': 0.6561363636363636, 'default': 0.7197077922077922},
                {'subject': '003_S_4288', 'parms': [-0.6167471693129847, 0.6969621281842724, -1.0, -1.7702224481438917, 1.0, -4.0], 'value': 0.6445454545454545, 'default': 0.7315974025974026},
                {'subject': '002_S_6103', 'parms': [-0.13744977837629868, 0.0, 1.0, -0.7029058275169962, 0.723596562515544, -2.8958111039649648], 'value': 0.7083246753246754, 'default': 0.7367532467532467},
                {'subject': '003_S_4644', 'parms': [-0.06573378675813846, 0.0, 0.6785413746154709, -4.0, 1.0, 0.0], 'value': 0.6703246753246753, 'default': 0.7348116883116884},
                {'subject': '003_S_6067', 'parms': [-1.0, 1.5166130503103976, -1.0, 0.0, 0.8570682414014428, 0.0], 'value': 0.136987012987013, 'default': 0.8607922077922079},
                {'subject': '007_S_4387', 'parms': [-0.796011746963555, 0.9252437064727224, -1.0, -2.150576811688341, -1.0, -0.7048022052632925], 'value': 0.7966518924550435, 'default': 0.8109720205724851},
                {'subject': '007_S_4488', 'parms': [-0.4892566547310744, 0.5488101088659639, -0.06542212747708209, -2.844390018947502, 1.0, 0.0], 'value': 0.7737105023530859, 'default': 0.8390672926345515},
                {'subject': '007_S_4620', 'parms': [-0.18733604837931894, 0.0, -1.0, -4.0, 1.0, 0.0], 'value': 0.8588610419772933, 'default': 0.8746723165730133},
                {'subject': '002_S_6007', 'parms': [-0.013973207684512934, 0.0, -0.7956758319191065, -4.0, 0.9302258539019355, 0.0], 'value': 0.49250649350649345, 'default': 0.5963506493506493},
                {'subject': '003_S_6014', 'parms': [-0.7447913782659041, 0.9681315925228252, 1.0, -4.0, -1.0, -3.884716810977994], 'value': 0.613525974025974, 'default': 0.7210779220779221},
                    # --target full_tau --group MCI
                {'subject': '002_S_1155', 'parms': [-0.03666120298427267, 0.0, 0.3389622316222132, -4.0, -1.0, 0.0], 'value': 0.5737467532467533, 'default': 0.6673766233766234},
                {'subject': '002_S_1261', 'parms': [-0.41540380973141233, 0.6496455146075011, 1.0, -4.0, 1.0, -4.0], 'value': 0.33966233766233767, 'default': 0.4052402597402597},
                {'subject': '002_S_4229', 'parms': [-1.0, 1.486312724189601, -1.0, 0.0, 0.03192598693818316, -0.041581853450993744], 'value': 0.5796363636363637, 'default': 0.6126948051948051},
                {'subject': '002_S_4654', 'parms': [-0.16424448916653467, 0.21686283386427174, 1.0, 0.0, 0.24117469209033082, -3.359169957237488], 'value': 0.40574025974025973, 'default': 0.4923961038961039},
                {'subject': '003_S_1122', 'parms': [-0.2537397240610091, 0.2668024530207156, -0.44210514593789674, -3.148642291106637, 0.2429120564820726, -3.6718118308046814], 'value': 0.43127272727272725, 'default': 0.49670129870129864},
                {'subject': '007_S_4272', 'parms': [-0.2726567799945365, 0.16045908626468158, 0.6827165788811298, -3.3681407745241927, 0.8105874503063111, -3.4967985072164103], 'value': 0.8660562881973968, 'default': 0.8740896292208199},
                {'subject': '012_S_6073', 'parms': [-0.08579800080467692, 0.0, 0.24569932505464176, -4.0, 1.0, -3.159647403240162], 'value': 0.7330194805194805, 'default': 0.7594870129870129},
                {'subject': '022_S_5004', 'parms': [-0.009326657547063633, 0.0, -1.0, 0.0, -1.0, -3.7518738315088154], 'value': 0.5606298701298702, 'default': 0.6106298701298701},
                {'subject': '023_S_4115', 'parms': [-0.8275351340172721, 0.8861362695198751, 0.10220693445210416, -3.4368107654762072, -0.3455328745448033, -3.5194869195052236], 'value': 0.8192596848258765, 'default': 0.8462357064189968},
                ]
        }
    for target in data:
        print(f"{target}")
        for subject in data[target]:
            subjName = subject["subject"]
            fileName = getSaveFileName(subjName, target)
            print(f'    going to repair: {subjName} on file {fileName}')
            sio.savemat(fileName, subject)
    print('    Reparations DONE!!!')


def saveResults(results, target):
    subjName = results["subject"]
    fileName = getSaveFileName(subjName, target)
    sio.savemat(fileName, results)


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def printHelp():
    print('pipeline2.py', '--aStart <aStartValue> --aEnd <aEndValue> --aStep <aStepValue>',
                             '--bStart <bStartValue> --bEnd <bEndValue> --bStep <bStepValue>',
                             '--aVar <aVarValue> --bVar <bVarValue>',
                             '--group <groupID>',
                             '--subject <subjectID>',
                             '--target <targetFunc>')


def processParmValues(argv):
    global aVar, bVar
    import getopt
    # =============== Let's make sure everything is in order...
    try:
        opts, args = getopt.getopt(argv,'',["aStart=","aEnd=","aStep=",
                                            "bStart=","bEnd=","bStep=",
                                            "aVar=", "bVar=",
                                            "subject=", "group=", "target="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)

    # ================ Default Values
    targetFunc = 'full_Abeta+Tau'
    group = 'AD'
    # Some example subjects...
    # nice FCD plot: 003_S_6067 / largest span AD: 036_S_4430 / 114_S_6039 / 168_S_6142 / 114_S_6039 / 011_S_4827 / ADSubjects[0]
    ADSubject = '003_S_6067'
    aStart = functions_AD.parmBounds[4][0]; aEnd = functions_AD.parmBounds[4][1]; aStep = 0.05
    bStart = functions_AD.parmBounds[5][0]; bEnd = functions_AD.parmBounds[5][1]; bStep = 0.05

    # =============== Let's process the input
    for opt, arg in opts:
        if opt in ['-h', '--h', '-H', '--H']:
            printHelp()
            sys.exit()
        elif opt in ("--aStart"):
            aStart = float(arg)
        elif opt in ("--aEnd"):
            aEnd = float(arg)
        elif opt in ("--aStep"):
            aStep = float(arg)
        elif opt in ("--bStart"):
            bStart = float(arg)
        elif opt in ("--bEnd"):
            bEnd = float(arg)
        elif opt in ("--bStep"):
            bStep = float(arg)
        elif opt in ("--aVar"):
            aVar = int(arg)
        elif opt in ("--bVar"):
            bVar = int(arg)
        elif opt in ("--subject"):
            try:  # OK, ugly type conversion here...
                ADSubject = int(arg)
            except:
                ADSubject = arg
        elif opt in ("--group"):
            group = arg
        elif opt in ("--target"):
            targetFunc = arg

    print(f'Input values are: aStart={aStart}, aEnd={aEnd}, aStep={aStep}')
    print(f'                  bStart={bStart}, bEnd={bEnd}, bStep={bStep}')
    # print(f'                  aVar={AD_functions.aVar}, bVar={AD_functions.bVar}')
    if isinstance(ADSubject, int):
        print(f'                  subject={ADSubject} (group={group})')
    else:
        print(f'                  subject={ADSubject} !!!')
    print(f'                  targetFunc={targetFunc}')
    return (aStart, aEnd, aStep), (bStart, bEnd, bStep), (ADSubject, group, targetFunc)


visualizeAll = True
if __name__ == '__main__':
    import sys
    aParmRange, bParmRange, studyParms = processParmValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 22})

    # ------------------------------------------------
    # Load individual classification
    # ------------------------------------------------
    subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    classification = dataLoader.checkClassifications(subjects)
    HCSubjects = [s for s in classification if classification[s] == 'HC']
    MCISubjects = [s for s in classification if classification[s] == 'MCI']
    ADSubjects = [s for s in classification if classification[s] == 'AD']
    print(f"We have {len(HCSubjects)} HC, {len(MCISubjects)} MCI and {len(ADSubjects)} AD \n")


    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    AvgHC = sio.loadmat(save_folder + '/AvgHC_SC.mat')['SC']
    dataLoader.analyzeMatrix("AvgHC norm", AvgHC)
    print("# of elements in AVG connectome: {}".format(AvgHC.shape))

    # ------------------------------------------------
    # Simulation settings
    # ------------------------------------------------
    distanceSettings = {'phFCD': (phFCD, True)}

    # ------------------------------------------------
    # Run pipeline for subject ADSubject
    # ------------------------------------------------
    if isinstance(studyParms[0], int):  # OK, ugly type check here...
        if studyParms[1] == 'AD':
            useGroup = ADSubjects
        elif studyParms[1] == 'MCI':
            useGroup = MCISubjects
        else:
            useGroup = HCSubjects
        finalSubject = useGroup[studyParms[0]]
    else:
        finalSubject = studyParms[0]
    targetFunc = studyParms[2]

    print('\n\n#######################################################')
    print(f"# Selected subject {finalSubject} from group {classification[finalSubject]}")
    print(f"# Selected target {targetFunc}")
    print('#######################################################\n\n')

    if emergencyPipeline:
        AD_EmergencyPipeline()
    result = AD_pipeline(finalSubject, distanceSettings, AvgHC, targetFunc)
    saveResults(result, targetFunc)

    print('\n\n#######################################################')
    print(f'# Final Result for {finalSubject}: {result}')
    print('#######################################################\n\n')

    # ------------------------------------------------
    # Done !!!
    # ------------------------------------------------
    print("DONE !!!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
