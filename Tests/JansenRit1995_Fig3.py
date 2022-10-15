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

import WholeBrain.Models.JansenRit as JR
import WholeBrain.Integrator_Euler as integrator

integrator.neuronalModel = JR
integrator.clamping = False

# In the original [JR_1995] paper, the random white noise input p(t) had an amplitude
# varying between 120 and 320 pulses per second.
import WholeBrain.Stimuli.randomUniform as stimuli
stimuli.onset = 0.
stimuli.ampLo = 120.
stimuli.ampHi = 320.
integrator.stimuli = stimuli


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # print("\n\nRecompiling signatures!!!")
    integrator.recompileSignatures()
    JR.recompileSignatures()


# Integration parms...
dt = 1e-4
tmax = 3.
integrator.ds = 1e-3
Tmaxneuronal = int((tmax+dt))
N = 1
Conn = np.zeros((N,N))
JR.setParms({'SC':Conn})


plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(6, sharex=True)
fig.suptitle("Pyramidal cell membrane potential")
for pos, C in enumerate([68., 128., 135., 270., 675., 1350.]):
    print(f"Computing {pos} with {C}...")

    # Simulate for a given JR.C
    JR.setParms({'C':C})
    # JR.initBookkeeping(N, tmax)
    recompileSignatures()
    v = integrator.simulate(dt, Tmaxneuronal)
    # v = JR.returnBookkeeping()

    # Plot the results!!!
    time = np.arange(0, Tmaxneuronal, integrator.ds)
    lowCut = int(0.8/integrator.ds)  # Ignore the first steps for warm-up...
    axs[pos].plot(time[lowCut:], v.reshape(-1)[lowCut:len(v)-1], 'k', alpha=1.0)
    axs[pos].set_title(f"C = {C}")
plt.show()


# ==========================================================================
# ==========================================================================
# ==========================================================================
