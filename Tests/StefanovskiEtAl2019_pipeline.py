# --------------------------------------------------------------------------------------
# Full pipeline from:
# [StefanovskiEtAl2019] Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
#           R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
#           "Linking molecular pathways and large-scale computational modeling to assess candidate
#           disease mechanisms and pharmacodynamics in Alzheimer's disease."
#           Front. Comput. Neurosci., 13 August 2019 | https://doi.org/10.3389/fncom.2019.00054
# Taken from the code at:
#           https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB/blob/master/Educase_AD_study-LS-Surrogate.ipynb
#
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from WholeBrain.Models import Abeta_StefanovskiEtAl2019 as Abeta
from WholeBrain.Models import JansenRit as JR

integrator = None
# Integration parms...
dt = 5e-5
tmax = 20.
ds = 1e-4
Tmaxneuronal = int((tmax+dt))


def recompileSignatures():
    integrator.recompileSignatures()  # just in case...
    JR.recompileSignatures()


def configSim(abeta_burden):
    global integrator, runSim
    import WholeBrain.Integrator_Euler
    integrator = WholeBrain.Integrator_Euler
    integrator.neuronalModel = JR
    integrator.clamping = False
    integrator.ds = ds

    # b = 0.07  # default Jansen-Rit inhibitory membrane constant
    JR.setParms({'b': Abeta.transform_abeta_exp(abeta_burden) * 1000})  # I use the original JR values for b...

    # Use a constant stimuli of 108.5/s.
    import WholeBrain.Stimuli.constant as stimuli
    # Do not set N, as it is constant...
    stimuli.onset = 0.
    stimuli.amp = 108.5  # [s^-1]
    integrator.stimuli = stimuli

    JR.setParms({'ds': ds})

    # In the original code we had:
    # mu = 0.1085
    # jrm = models.JansenRit(v0=6., mu=mu, p_max=mu, p_min=mu,
    #                        b = b,
    #                        variables_of_interest=['y1 - y2'])

    # init = np.random.rand(4000,6,SCnorm.shape[0],1);

    # omitting any time delay between regions -> not needed in my implementation...
    # white_matter = connectivity.Connectivity(weights=SCnorm, tract_lengths=np.zeros(SCnorm.shape))

    # set up the simulator -> Already done in my implementation...
    # adjust the simulation_length to your needs/ available computation time
    # in the paper a simulation_length of 120000 was used
    # but only the 2nd minute was used in the analysis to cut out possible transients
    # sim = simulator.Simulator(  connectivity=white_matter,
    #                             model=jrm,
    #                             coupling=coupling.SigmoidalJansenRit(a=0),
    #                             integrator=integrators.HeunDeterministic(dt=0.5),
    #                             conduction_speed=100,
    #                             monitors=monitors.SubSample(period=5),
    #                             initial_conditions = init,
    #                             simulation_length=4000.0)
    # sim.configure();


def run_sim(SCnorm, lf_mat):
    import scipy.signal as sig

    N = SCnorm.shape[0]
    # JR.initBookkeeping(N, tmax)
    JR.setParms({'SC': SCnorm})
    recompileSignatures()  # just in case...
    v = integrator.simulate(dt, Tmaxneuronal)
    # v = integrator.returnBookkeeping()
    PSP = v[400:,0,:]

    # --------------------------------------------------------------
    ##### Analyze PSP
    # analyze signal, get baseline and frequency
    psp_baseline = PSP.mean(axis=0)
    psp_f, psp_pxx = sig.periodogram(PSP-psp_baseline, axis=0) # nfft=1024, fs=200)
    psp_f *= 10./(dt*tmax)  # needed because of...
    psp_peak_freq = psp_f[np.argmax(psp_pxx, axis=0)]
    p = np.max(psp_pxx, axis=0)

    # --------------------------------------------------------------
    # ##### Analyze EEG
    # generate EEG by multiplication of PSP with lf_mat
    EEG = lf_mat.dot(PSP.T) # EEG shape [n_samples x n_eeg_channels]

    # reference is mean signal, tranposing because trailing dimension of arrays must agree
    EEG = (EEG.T - EEG.mean(axis=1).T).T

    # analyze signal, get baseline and frequency
    # eeg_baseline = EEG.mean(axis=0)
    EEG = EEG - EEG.mean(axis=0)  # center EEG

    eeg_f, eeg_pxx = sig.periodogram(EEG, axis=1)  # fs=200, nfft=1024,
    # eeg_Pxx        = eeg_pxx.T
    eeg_f *= 10./(dt*tmax)
    eeg_peak_freq  = eeg_f[np.argmax(eeg_pxx.T, axis=0)]  # eeg_pxx.T

    # --------------------------------------------------------------
    # if save_timeseries_to_file :
    #     # save results
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #     file_name=save_path+"/"+DX+"_gc_"+str(global_coupling)+".mat"
    #     sio.savemat(file_name,mdict={"PSP":PSP})

    return psp_baseline, psp_peak_freq, eeg_peak_freq


def simAllGC(gc_range, SCnorm, lf_mat):
    def simWe(we):
        JR.setParms({'we': we})
        return run_sim(SCnorm, lf_mat)

    print('starting sim: we={}'.format(gc_range[0]))
    psp_baseline, psp_peak_freq, eeg_peak_freq = simWe(gc_range[0])
    for we in gc_range[1:]:
        print('starting sim: we={}'.format(we))
        psp_b, psp_p, eeg_p = simWe(we)
        # concatenate the output
        psp_baseline = np.vstack((psp_baseline, psp_b))
        psp_peak_freq = np.vstack((psp_peak_freq, psp_p))
        eeg_peak_freq = np.vstack((eeg_peak_freq, eeg_p))
    return psp_baseline, psp_peak_freq, eeg_peak_freq


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
    x_coord     = gc_range.repeat(379)
    x_coord_eeg = gc_range.repeat(64)

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


visualizeAll = True
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})
    # for the visualization of the ABeta curve, look the file StefanovskiEtAl2019_Fig3.py
    # ------------------------------------------------
    # Load individual Abeta PET SUVRs
    # ------------------------------------------------
    # select the subject you want to simulate
    base_folder = "../Data_Raw/surrogate_AD"
    DX = "MCI" # one of AD, MCI or HC
    modality = "Amyloid"
    pet_path = base_folder+"/_"+DX
    RH_pet = np.loadtxt(pet_path+"/"+DX+"_RH.txt")
    LH_pet = np.loadtxt(pet_path+"/"+DX+"_LH.txt")
    subcort_pet = np.loadtxt(pet_path+"/"+DX+"_subcortical.txt")
    abeta_burden = np.concatenate((LH_pet,RH_pet,subcort_pet))

    if visualizeAll:
        n, bins, patches = plt.hist(abeta_burden, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Abeta SUVR')
        plt.ylabel('Regions')
        plt.suptitle("Abeta histogram", fontweight="bold", fontsize="18", y = 1.05)
        plt.show()

    # Configure Simulator
    # ------------------------------------------------
    # load SC
    sc_folder = "../Data_Raw/surrogate_AD"
    SCnorm = np.loadtxt(sc_folder+"/avg_healthy_normSC_mod_379.txt")

    if visualizeAll:
        # Plot Figure 4A in [StefanovskiEtAl2019]
        plt.imshow(np.asarray(SCnorm))
        plt.colorbar()
        plt.xlabel("Regions")
        plt.ylabel("Regions")
        plt.suptitle("Structural Connectivity", fontweight="bold", fontsize="18", y = 1.05)
        plt.show()

    # load leadfield matrix
    lf_mat = sio.loadmat(base_folder+"/_"+DX+"/leadfield.mat")["lf_sum"]

    # Simulate!
    # --------------------------------------------------------------
    # define global coupling range to explore in simulation
    # in the original study a range from 0 to 600 with steps of 3 was explored
    # NOTE: Too many steps will take very long time when running the script on a local computer
    gc_range = np.arange(0, 600, 30)  # 30

    configSim(abeta_burden)

    psp_baseline, psp_peak_freq, eeg_peak_freq = simAllGC(gc_range, SCnorm, lf_mat)

    displayResults(gc_range, psp_baseline, psp_peak_freq, eeg_peak_freq)

# =========================================================
# =========================================================
# =========================================================EOF
