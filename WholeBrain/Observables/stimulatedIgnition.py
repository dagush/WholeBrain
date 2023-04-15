# --------------------------------------------------------------------------------------
# Full pipeline for Stimulated Ignition computation
#
#  Taken from the code from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito,
#  "Dynamical consequences of regional heterogeneity in the brain’s transcriptional landscape",
#  SCIENCE ADVANCES, 14 Jul 2021, Vol 7, Issue 29, DOI: 10.1126/sciadv.abf4752
#
#  Based on:
#  [ChaudhuriEtAl_2015] A Large-Scale Circuit Mechanism for Hierarchical Dynamical Processing in the Primate Cortex
#  Rishidev Chaudhuri, Kenneth Knoblauch, Marie-Alice Gariel, Henry Kennedy, Xiao-Jing Wang,
#  Neuron, Volume 88, Issue 2, 21 October 2015, Pages 419-431, DOI: 10.1016/j.neuron.2015.09.008
#
# Adapted to python by Gustavo Patow
# --------------------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from WholeBrain.Utils.decorators import loadOrCompute

print("Going to use Stimulated Ignition...")

name = 'stimulatedIgnition'

# 	Fitting routines for Stimulated Ignition
sigfunc = lambda x,A0,A1,A2,A3 : A0 / (1 + np.exp(-A1*(x-A2))) + A3  # sigfunc = @(A, x)(A(1) ./ (1 + exp(-A(2)*(x-A(3)))) + A(4))
plotting = False


# From [DecoEtAl_2021]:
# We measure the evoked responses at the level of population firing rates rather than simulated
# BOLD signal changes to have direct access to the millisecond timescale. To quantify the
# effect of occipital stimulation on activity in each of the other 66 brain regions, we plot, for
# each non-stimulated region, how its population firing rate changes as a function of occipital
# stimulation intensity. Two quantities are relevant here: (1) the maximum firing rate achieved
# at the highest stimulation intensity, r_{max}; and (2) the speed with which the firing rate
# increases beyond a given intensity threshold, c_{max}, which we quantify as the concavity of the
# regional response function (i.e., the maximal second derivative). We then summarize the
# ignition capacity of each non-stimulated region i as I(i) = c_{max}(i) x r_{max}(i), and estimate
# the global ignition capacity of the brain as the mean across all regions, I = 1/(N-1) \sum I(i).
def stimulatedIgnition(PERTURB, neuro_act_all, SEED):
    # NSUB = 389
    # Ntrials = 10
    # Tmax = 616
    # indexsub = np.arange(0,NSUB)

    # ============================================================================
    # Parameters that were used in the simulation.
    nPerturbs, T, N = neuro_act_all.shape  # N = 68
    # SEED=10;

    peakrate2 = np.zeros((PERTURB.size, N))
    peakrate3 = np.zeros(N)
    basal = np.zeros(N)
    # First assign the variables to get the max peak rate:
    for ppos, perturb in enumerate(PERTURB):
        neuro_actf = neuro_act_all[ppos, :, :]
        for area in range(N):
            peakrate3[area] = np.mean(neuro_actf[160:200, area], axis=0)
            basal[area] = np.mean(neuro_actf[100:140, area])
            # tscale=0:1/length(decayneuro):1
        # end ss in range(N)
        peakrate2[ppos, :] = peakrate3/basal
    # end perturb in PERTURB

    # Below we have the range that you want the function to eventually sample to:
    PERTURB_sample = np.arange(0,0.2,0.001)


    # 	options=optimset('MaxFunEvals',10000,'MaxIter',1000,'Display','off');

    peakrate1 = np.mean(peakrate2[-10, :])

    ignition1 = np.zeros(N)
    ignition1a = np.zeros(N)
    ignition1b = np.zeros(N)
    # nntarget = 1
    if plotting:
        plt.rcParams.update({'font.size': 8})
        fig = plt.figure()
        fig.subplots_adjust(hspace=1.0, wspace=0.8)
        fig.suptitle(f"Ignition (region {SEED})")
    # figure('color','white');
    for ntarget in range(N):
        a00 = np.abs(np.mean(peakrate2[-10:,ntarget])-np.mean(peakrate2[0:9,ntarget]))
        A0 = np.array([a00,
                       10,
                       0.1,
                       np.mean(peakrate2[0:9,ntarget])])
        # starts at x0 and finds coefficients x to best fit the nonlinear function sigfun(x,PERTURB)
        # to the data peakrate2 (in the least-squares sense). peakrate2 must be the same size as the
        # vector (or matrix) F returned by sigfun.
        Afit = curve_fit(sigfunc,  # f: callable
                         PERTURB, peakrate2[:, ntarget].T,  # xdata (The independent variable), ydata (The dependent data)
                         p0=A0,  # Initial guess for the parameters (length N).
                         bounds=([0, 0, -1, 0],
                                 [100, 100, 1, 10*np.mean(peakrate2[0:9,ntarget])]),  # Lower & Upper bounds
                         method='trf', maxfev=50000)  # lsqcurvefit
        # Just a little part here to sample the fitted regime to a different level (the normalization level is always for above)
        AValues = Afit[0]
        yfit = sigfunc(PERTURB_sample, *AValues)  # AValues[0] / (1 + np.exp(-AValues[1]*(PERTURB_sample-AValues[2])))+AValues[3]

        if plotting:
            if ntarget+1 < 35:
                ax = fig.add_subplot(6,6,ntarget+1)
                points = ax.plot(PERTURB, peakrate2[:,ntarget], '.')[0]
                lines = ax.plot(PERTURB_sample, yfit)[0]
                ax.set_title(f'Region: {ntarget+1}')
                ax.set_xlabel(r'$\rho$')
                ax.set_ylabel(r'$r^E_{max}/ r^E_{rest}$')
                # ax.legend()

        # Calculating ignition and remembering to divide by the sampling rate (it should really just
        # be "grad" to make it easier...)
        ignition1[ntarget] = np.max(np.diff(np.diff(yfit/0.001)/0.001)) * yfit[-1]/1000
        ignition1a[ntarget] = yfit[-1]/1000
        ignition1b[ntarget] = np.max(np.diff(np.diff(yfit/0.001)/0.001))
        # nntarget=nntarget+1;
    # end for ntarget in range(N)

    if plotting:
        # ax = fig.add_subplot(6,6,36)
        fig.legend([points, lines],
                  labels=['data', 'fit'],
                  loc="lower right")
        plt.show()

    # Here we are looking at the SEED 10, and its contralateral component and zeroing it out
    ignition1[SEED] = 0
    ignition1[SEED+int(N/2)] = 0

    Ignition2 = np.mean(ignition1[ignition1 > np.mean(ignition1)+np.std(ignition1)])
    Excitability2 = np.mean(peakrate1)

    print("Done!!!")

    # Now here we want to calculate the decay using a nonlinear fit
    kk = neuro_act_all.shape[2]
    # Using the last point (the point of greatest stimulation)
    sr = 20*1e-3
    time_vector = np.arange(120, 350)
    time = np.arange(0, (time_vector.size-1)*sr, sr)
    whole_time = np.arange(0, (350-1)*sr, sr)

    # Using a nonlinear decay function instead
    expfunc = lambda x, A, B, D: A*(np.exp(-x*D)+B)
    # s = fitoptions('Method','NonlinearLeastSquares','StartPoint',[1 0.5 3])
    # f = fittype('A*(exp(-x*D)+B)','options',s)
    new_time = np.linspace(whole_time[0], whole_time[-1], 1000)
    perturb = 0.2
    stim_vector = perturb * 0.5 * (np.sign(new_time-3) - np.sign(new_time-4))

    recalcuated_decay = np.zeros((N,3))
    neuro_actf = neuro_act_all[:,:,kk]
    tseed = np.arange(1,N)
    # 	ssnum=1;
    for ssnum, ss in enumerate(tseed):
        decayneuro = np.squeeze(neuro_actf[201:, ss])
        tscale = np.arange(0, (decayneuro.size-1)*sr, sr)

        cfit = curve_fit(expfunc,  # f: callable
                         tscale[0:decayneuro.size].T, decayneuro,  # xdata (The independent variable), ydata (The dependent data)
                         p0=np.array([1, 0.5, 3]),  # Initial guess for the parameters (length N).
                         # bounds=([0, 0, -1, 0],
                         #         [100, 100, 1, 10*np.mean(peakrate2[0:9,ntarget])]),  # Lower & Upper bounds
                         method='lm', maxfev=50000)  # [c, gof] = fit(,decayneuro,f)
        # Just a little part here to sample the fitted regime to a different level (the normalization level is always for above)
        AValues = cfit[0]

        recalcuated_decay[ss] = AValues  #c.D
        # ssnum=ssnum+1;
        # % figure;
        # % plot(tscale(1:length(decayneuro)),(decayneuro'))
        # % hold on;
        # % plot(tscale(1:length(decayneuro)),f(c.A,c.B,c.D,tscale(1:length(decayneuro))))
        # % plot(tscale(1:length(decayneuro)),exp(polyval(bdecay,tscale(1:length(decayneuro)))));
        # % legend({'\phi(t)','fitNew','fitOLD'});
    #     end
    # end
    return Ignition2, Excitability2, ignition1, ignition1a, recalcuated_decay


def setupStimulation(seed, perturb):
    # do the simulation setting stuff...
    integrator.stimuli = stim
    stim.N = N
    stim.onset = 3000
    stim.termination = 4000
    stim.seed = [seed, int(N/2+seed)]
    stim.amp = perturb


@loadOrCompute
def simulateTrials(numTrials, N, PERTURB, seed):
    neuro_act = np.zeros((numTrials, Tmaxneuronal+1, N))
    neuro_actf = np.zeros((int(Tmaxneuronal/20), N))
    neuro_act_all = np.zeros((PERTURB.size, int(Tmaxneuronal/20.), N))

    for ppos, pval in enumerate(PERTURB):
        print(f"simulating PERTURB: {pval}/{PERTURB[-1]}")
        setupStimulation(seed, pval)
        for i in range(numTrials):
            print(f"   simulating Trial: {i}")
            currObsVars = integrator.simulate(dt, Tmaxneuronal)
            neuro_act[i] = currObsVars[:, 1, :]  # curr_rn

        neuro_act1 = np.squeeze(np.mean(neuro_act, axis=0))
        for ntwi, twi in enumerate(np.arange(0, Tmaxneuronal-20, 20)):
            neuro_actf[ntwi, :] = np.mean(neuro_act1[twi:twi+20, :], axis=0)

        neuro_act_all[ppos] = neuro_actf

    return {"neuro_act": neuro_act_all}


if __name__ == '__main__':
    inDataPath = "../../Data_Raw/DecoEtAl2020/"
    outDataPath = "../../Data_Produced/DecoEtAl2020/"

    # --------------------------------------------------------------------------
    #  Begin setup...
    # --------------------------------------------------------------------------
    import WholeBrain.Models.DynamicMeanField as neuronalModel
    import Integrators.EulerMaruyama as integrator
    integrator.neuronalModel = neuronalModel
    integrator.verbose = False

    import WholeBrain.BalanceFIC as BalanceFIC
    BalanceFIC.integrator = integrator
    # --------------------------------------------------------------------------
    #  End setup...
    # --------------------------------------------------------------------------

    # ===========================================================================
    # Parameters for the mean field model
    # taon=100  # as in the DMF model definition
    # taog=10   #           ✓
    # gamma=0.641  #        ✓
    # sigma=0.01  # as in the Euler-Maruyama integrator
    # JN=0.15  # as in the DMF model definition
    # I0=0.382  #           ✓
    # Jexte=1.  #           ✓
    # Jexti=0.7  #          ✓
    # w=1.4  #              ✓

    # ===========================================================================
    # Set General Model Parameters
    dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
                # note: 1e-3 is the length of a millisecond, in seconds,
                # so basically this is a milliseconds to seconds conversion factor
                # and 1/dtt is a seconds to milliseconds conversion factor...
    dt  = 0.1

    TR = 1.            # Sampling rate of recorded BOLD simulation (seconds)
    Tmax = 7.        # Number of (useful) time-points in each fMRI session
                       # each time-point is separated by TR seconds => Tmax * TR is the total length, in seconds
    Toffset = 0.
    Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))  # Number of simulated time points (in milliseconds)
    print(f"going to simulate {Tmaxneuronal} timepoints at {dt} samples per millisecond ;-)")

    # ===========================================================================
    # OK, let's roll!
    # ===========================================================================
    # s=1  # s=str2num(getenv('SLURM_ARRAY_TASK_ID'))  % for debug purposes only...

    dataFile = 'SC_GenCog_PROB_30.mat'
    M = sio.loadmat(inDataPath + dataFile); print('{} File contents:'.format(inDataPath + dataFile), [k for k in M.keys()])
    GrCV = M['GrCV']
    tcrange = np.union1d(np.arange(0,34), np.arange(41,75))  # [1:34 42:75]
    C = GrCV[:, tcrange][tcrange, ]
    C=C/np.max(C)*0.2
    print(f'C shape is {C.shape}')
    N = 68
    neuronalModel.SC = C

    # ------------------------------------------------
    # Configure simulation
    # ------------------------------------------------
    we = 2.1  # Result from the pre-processing stage
    J_fileName = outDataPath + f'BenjiBalancedWeights-{we}.mat'
    neuronalModel.we = we
    balancedParms = BalanceFIC.Balance_J9(we, C, J_fileName.format(we))
    integrator.neuronalModel.J = balancedParms['J'].flatten()
    integrator.recompileSignatures()

    # regions of occipital cortex (namely, the cuneus, lateral occipital, lingual gyrus, and pericalcarine
    # regions in the Desikan-Killiany atlas (34)).
    import WholeBrain.Stimuli.singleAreaStimulation as stim
    # SEED = np.array([4, 10, 12, 20])
    SEED = np.array([10])  # only 1 for debug
    PERTURB = np.arange(0., 0.2, 0.001)
    # PERTURB = np.arange(0., 0.2, 0.05)  # just a few, for debug

    numTrials = 20  # 500
    for s in SEED:
        outFileName = outDataPath + f'neuro_act_all-we{we}-seed{s}-trials{numTrials}.mat'
        print(f"computing {outFileName} at seed {s}")
        neuro_act = simulateTrials(numTrials, N, PERTURB, s, outFileName)["neuro_act"]
        stimulatedIgnition(PERTURB, neuro_act, s)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
