# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
# Fit the Jansen-Rit model (JR) parameters so it has a firing rate at a frequency of 3Hz
# See the JR model at:
#
#     .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
#         visual evoked potential generation in a mathematical model of
#         coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.
#
# ================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import functions.Models.JansenRit as JR
import functions.Integrator_Euler as integrator
import functions.Utils.fft as fft

integrator.neuronalModel = JR
integrator.clamping = False

import JR_RunSim as runJR
runJR.JR = JR
runSim = runJR.runSim2


# [DF_2003] used a standard normal distribution...
import functions.Stimuli.randomStdNormal as stimuli
stimuli.onset = 0.
stimuli.mu = 220.
stimuli.sigma = 22.
integrator.stimuli = stimuli

# Integration parms...
dt = 5e-5
runJR.dt = dt
tmax = 10.
runJR.tmax = tmax
JR.ds = 1e-4
Tmaxneuronal = int((tmax+dt))
runJR.Tmaxneuronal = Tmaxneuronal
N = 1
Conn = np.zeros((N,N))

# Take the original values, so we can keep the ratio invariant later on...
H_e_orig = 3.25         # JR.A [mV]
H_i_orig = 22           # JR.B [mV]
tau_e_orig = 10e-3      # 1./JR.a [s]
tau_i_orig = 20e-3      # 1./JR.b [s]


def evalJR(tau_e, tau_i):
    # Simulate for a given combination of tau_e and tau_i
    JR.A = H_e_orig*tau_e_orig/tau_e
    JR.B = H_i_orig*tau_i_orig/tau_i
    JR.a = 1./tau_e
    JR.b = 1./tau_i

    return runSim(Conn)


targetFreq = 3.  # We want the firing rate to be at 3Hz
def distTo3Hz(f):
    return np.abs(f-targetFreq)

def drawFiringRates(tau_es, tau_is):
    import functions.Utils.visualization as viz
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    max_freq = np.zeros((len(tau_is), len(tau_es)))
    freq_pow = np.zeros_like(max_freq)
    min = np.inf
    minTau_e, minTau_i = 0, 0
    minidx_r, minidx_c = 0, 0
    # calculate PSDs
    for idx_c, tau_e in enumerate(tau_es):
        for idx_r, tau_i in enumerate(tau_is):
            print("Computing ({},{}) at ({},{})...".format(tau_e, tau_i, idx_c, idx_r), end=' ')

            f, p, freqs, power, v = evalJR(tau_e, tau_i)
            if p < 140.:
                max_freq[idx_r, idx_c] = f
            else:
                max_freq[idx_r, idx_c] = 0.
            freq_pow[idx_r, idx_c] = p
            print("f={}".format(f))

            # "minimize" to find the freq we're looking for
            if distTo3Hz(f) < min:
                min = distTo3Hz(f)
                minTau_e, minTau_i = tau_e, tau_i
                minidx_r, minidx_c = idx_r, idx_c

    # -------------------------------------------------
    # Plot the results!!!
    plt.rcParams.update({'font.size': 15})
    # create axis ticks
    yticks = (tau_is * 1e3).astype('int')
    yticks[::2] = 0.
    yticks = [str(y) if y > 0. else '' for y in yticks]

    xticks = (tau_es * 1e3).astype('int')
    xticks[::2] = 0.
    xticks = [str(x) if x > 0. else '' for x in xticks]

    # plot the dominant frequencies
    fig, ax = plt.subplots()
    im, cbar = viz.heatmap(max_freq, yticks, xticks, ax=ax,
                           cmap="Oranges", origin='lower')
    ax.set_xlabel(r'$\mathbf{\tau_e}$ in ms', labelpad=15.)
    ax.set_ylabel(r'$\mathbf{\tau_i}$ in ms', labelpad=15.)
    ax.set_title(f'Dominant frequency band', pad=20.)
    texts = viz.annotate_heatmap(im, valfmt="{x:.2f}")
    # cbar = plt.gcf().axes[-1]
    # cbar.set_yticklabels(['hyper signal', r'1. $\delta$ (1-4 Hz.)', r'2. $\theta$ (4-8 Hz.)', r'3. $\alpha$ (8-12) Hz.',
    #                       r'4. $\beta$ (12-30) Hz.', r'5. $\gamma$ (> 30 Hz.)'])

    highlightCoords = (minidx_c-0.5, minidx_r-0.5)  # coords of the highlight rectangle
    rect = plt.Rectangle(highlightCoords, 1, 1, fill=False, edgecolor='blue', lw=3, zorder=2)
    ax.add_patch(rect)

    fig.tight_layout()
    plt.show()
    return minTau_e,minTau_i,min+targetFreq


def fitJansenRitTo3Hz(iniTau_e, iniTau_i):
    def errorFunc(x):
        print("x:",x)
        f, p, freqs, power, v = evalJR(x[0], x[1])
        if p < 140.:
            finalf = f
        else:
            finalf = 0.
        return distTo3Hz(finalf)

    print("Fitting the JansenRit model to 3Hz...")
    # init...
    from scipy.optimize import minimize
    initialValues = np.array([iniTau_e, iniTau_i])

    # Now, fit it !!!
    # Note: Be careful with max_nfev, evaluations include numerical derivatives, so the actual number will be around 4.6 (or more) * this number before stopping!!!
    res = minimize(errorFunc, x0=initialValues, method='CG', options={'gtol': 1e-08})
    final_values = errorFunc(res.x)
    print("Result:", res)
    print("Final Value:", final_values+targetFreq)
    return res, final_values


# ================================================================================================================
# compute the ranges we are going to use for tau_e and tau_i
step = 6e-3
tau_es = np.arange(1e-3, 63e-3, step)         # excitatory synaptic timescales
tau_is = np.arange(1e-3, 63e-3, step)         # inhibitory synaptic timescales
minTau_e,minTau_i,min = drawFiringRates(tau_es, tau_is)
print('Minimum found at: tau_e={}ms, tau_i={}ms with value={}'.format(minTau_e,minTau_i,min))
# Result:
# Minimum found at: tau_e=0.043000000000000003ms, tau_i=0.061ms with value=3.051781095742002

step = 2e-3
tau_es = np.arange(43e-3, 63e-3, step)         # excitatory synaptic timescales
tau_is = np.arange(36e-3, 56e-3, step)         # inhibitory synaptic timescales
minTau_e,minTau_i,min = drawFiringRates(tau_es, tau_is)
print('Minimum found at: tau_e={}ms, tau_i={}ms with value={}'.format(minTau_e,minTau_i,min))
# Result:
# Minimum found at: tau_e=0.051000000000000004ms, tau_i=0.05400000000000001ms with value=3.024513431651548

fitJansenRitTo3Hz(0.051, 0.054)  # use (0.043, 0.061) for the result of the first visual iteration...
# Result:
#      fun: 0.024513431651548068
#      jac: array([0., 0.])
#  message: 'Optimization terminated successfully.'
#     nfev: 112
#      nit: 2
#     njev: 28
#   status: 0
#  success: True
#        x: array([0.051    , 0.0540513])
# final value=3.051781095742002
# As you can see, it didn't move from its initial position. Typical of optimization inside noisy functions...
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
