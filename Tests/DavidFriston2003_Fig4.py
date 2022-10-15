# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
#                       Figure 4 from [DF_2003]
# [DF_2003] Olivier David, Karl J. Friston, “A neural mass model for MEG/EEG:: coupling and neuronal dynamics”, NeuroImage,
#           Volume 20, Issue 3, 2003, Pages 1743-1755, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2003.07.015.
#
# In this example we examine the relationship between changes in the synaptic timescales and the dynamic behavior
# of the Jansen-Rit model. Thereby, we systematically change the excitatory and inhibitory synaptic timescales and
# evaluate dynamic behavior of the Jansen-Rit model for each combination of the two. The dynamic behavior is
# operationalized as the response frequency band with the highest power-spectral density evaluated via the membrane
# potential time traces of the pyramidal cell population.
# Explanation taken from [PyRates]:
# [PyRates] Gast, R., Daniel, R., Moeller, H. E., Weiskopf, N. and Knoesche, T. R. (2019). “PyRates – A Python Framework
#           for rate-based neural Simulations.” bioRxiv (https://www.biorxiv.org/content/10.1101/608067v2).
# Also
# [SpieglerEtAl2013] Spiegler A1, Kiebel SJ, Atay FM, Knösche TR. (2010). "Bifurcation analysis of neural mass models:
#           Impact of extrinsic inputs and dendritic time constants."
#           Neuroimage. Sep;52(3):1041-58. doi: 10.1016/j.neuroimage.2009.12.081. Epub 2010 Jan 4.
# ================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import WholeBrain.Models.JansenRit as JR
import WholeBrain.Integrator_Euler as integrator
import WholeBrain.Utils.fft as fft

np.random.seed(7)

integrator.neuronalModel = JR
integrator.clamping = False

import WholeBrain.Utils.visualization as viz
import bisect

# [DF_2003] used a standard normal distribution...
import WholeBrain.Stimuli.randomStdNormal as stimuli
stimuli.onset = 0.
stimuli.mu = 220.
stimuli.sigma = 22.
integrator.stimuli = stimuli


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # print("\n\nRecompiling signatures!!!")
    integrator.recompileSignatures()
    JR.recompileSignatures()


# Integration parms...
dt = 5e-5
tmax = 10.
integrator.ds = 1e-4
Tmaxneuronal = int((tmax+dt))
N = 1
Conn = np.zeros((N,N))
JR.setParms({'SC': Conn})


# Take the original values, so we can keep the ratio invariant later on...
H_e_orig = 3.25         # JR.A [mV]
H_i_orig = 22           # JR.B [mV]
tau_e_orig = 10e-3      # 1./JR.a [s]
tau_i_orig = 20e-3      # 1./JR.b [s]

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# # Some quick debug code...
# ---------------------------------------------------------------
# tau_e = 60e-3
# tau_i = 60e-3
# JR.A = H_e_orig*tau_e_orig/tau_e #H_e_orig / tau_e_orig * tau_e # 3.25e-3*10e-3/tau_e
# JR.B = H_i_orig*tau_i_orig/tau_i #H_i_orig / tau_i_orig * tau_i # -22e-3*20e-3/tau_i
# JR.a = 1./tau_e
# JR.b = 1./tau_i
# JR.initBookkeeping(N, tmax)
# integrator.simulate(dt, Tmaxneuronal, Conn)
# v = JR.returnBookkeeping()
# plt.plot(v[10000:20000])
# plt.show()
# print("Single freq done...")
# ---------------------------------------------------------------
# ---------------------------------------------------------------

# compute the ranges we are going to use for tau_e and tau_i
tau_es = np.arange(1e-3, 63e-3, 5e-3)         # excitatory synaptic timescales
tau_is = np.arange(1e-3, 63e-3, 5e-3)         # inhibitory synaptic timescales
max_freq = np.zeros((len(tau_is), len(tau_es)))
freq_pow = np.zeros_like(max_freq)
# calculate PSDs
for idx_c, tau_e in enumerate(tau_es):
    for idx_r, tau_i in enumerate(tau_is):
        print("Computing ({},{}) at ({},{})...".format(tau_e, tau_i, idx_c, idx_r), end='')

        # Simulate for a given combination of tau_e and tau_i
        # This uses the definition by [SpieglerEtAl2013] & [PyRates], which are "similar" to the one used in [DF_2003]
        JR.setParms({'A': H_e_orig*tau_e_orig/tau_e,
                     'B': H_i_orig*tau_i_orig/tau_i,
                     'a': 1./tau_e,
                     'b': 1./tau_i})
        integrator.initBookkeeping(N, tmax)
        recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)

        lowCut = int(1./integrator.ds)  # Ignore the first steps for warm-up...
        freqs, power = fft.fft(v[lowCut:]/1e3, integrator.ds)  # we make use of linearity of the fft to avoid too high values...
        p = np.max(power)
        f = freqs[np.argmax(power)]
        if p < 140.:
            max_freq[idx_r, idx_c] = bisect.bisect_left([4., 8., 12., 30.], f) + 1
        else:
            max_freq[idx_r, idx_c] = 0.
        freq_pow[idx_r, idx_c] = p

# -------------------------------------------------
# Plot the results!!!
plt.rcParams.update({'font.size': 12})
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
texts = viz.annotate_heatmap(im, valfmt="{x:.0f}")
cbar = plt.gcf().axes[-1]
cbar.set_yticklabels(['hyper signal', r'1. $\delta$ (1-4 Hz.)', r'2. $\theta$ (4-8 Hz.)', r'3. $\alpha$ (8-12) Hz.',
                      r'4. $\beta$ (12-30) Hz.', r'5. $\gamma$ (> 30 Hz.)'])

fig.tight_layout()
plt.show()
