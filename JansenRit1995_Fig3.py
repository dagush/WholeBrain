# ==========================================================================
#                       Figure 3 from [JR_1995]
#     .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
#         visual evoked potential generation in a mathematical model of
#         coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.
# ==========================================================================
print("====================================")
print("=  Generating fig 3 in [JR_1995]   =")
print("====================================")
import numpy as np
import matplotlib.pyplot as plt
import functions.Models.JansenRit as JR
import functions.Integrator_Euler as integrator

integrator.neuronalModel = JR
integrator.clamping = False

# In the original [JR_1995] paper, the random white noise input p(t) had an amplitude
# varying between 120 and 320 pulses per second.
import functions.Stimuli.randomUniform as stimuli
stimuli.onset = 0.
stimuli.ampLo = 120.
stimuli.ampHi = 320.
integrator.stimuli = stimuli

# Integration parms...
dt = 1e-4
tmax = 3.
JR.ds = 1e-3
Tmaxneuronal = int((tmax+dt))
N = 1
Conn = np.zeros((N,N))

plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(6, sharex=True)
fig.suptitle("Pyramidal cell membrane potential")
for pos, JR.C in enumerate([68., 128., 135., 270., 675., 1350.]):
    print("Computing {} with {}...".format(pos,JR.C))

    # Simulate for a given JR.C
    JR.initBookkeeping(N, tmax)
    integrator.simulate(dt, Tmaxneuronal, Conn)
    v = JR.returnBookkeeping()

    # Plot the results!!!
    time = np.arange(0, Tmaxneuronal, JR.ds)
    lowCut = int(0.8/JR.ds)  # Ignore the first steps for warm-up...
    axs[pos].plot(time[lowCut:], v[lowCut:], 'k', alpha=1.0)
    axs[pos].set_title("C = {}".format(JR.C))
plt.show()


# ==========================================================================
# ==========================================================================
# ==========================================================================
