#
#
# This prog. optimizes the strengh of the feedback inhibition of the FIC model 
# for varying global couplings (G)
# Saves the steady states and the feedback inhibition (J).
#
#
# For an isolated node, an input to the excitatory pool equal to I_i^E - b_E/a_E = -0.026; 
# i.e., slightly inhibitory dominated, leads to a firing rate equal to 3.0631 Hz. 
# Hence, in the large-scale model of interconnected brain areas, 
# we aim to constraint in each brain area (i) the local feedback inhibitory weight Ji such 
# that I_i^E - b_E/a_E = -0.026 is fulfilled (with a tolerance of +-0.005). 
# To achieve this, we apply following procedure: we simulate during 5000 steps 
# the system of stochastic differential DMF Equations and compute the averaged level of 
# the input to the local excitatory pool of each brain area,
# then we upregulate the corresponding local feedback inhibition J_i = J_i + delta;
# otherwise, we downregulate J_i = J_i - delta. 
# We recursively repeat this procedure until the constraint on the input
# to the local excitatory pool is fulfilled in all N brain areas.
#
# see:
# Deco et al. (2014) J Neurosci.
# http://www.jneurosci.org/content/34/23/7886.long
#
# Adrian Ponce-Alvarez
#--------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
#from randn2 import randn2
import Balance_J9

np.random.seed(42)  # Fix the seed for debug purposes...

# ======================================================================
# ======================================================================
# ======================================================================
# Load connectome:
# --------------------------------
CFile = sio.loadmat('Human_66.mat')  # load Human_66.mat C
C = CFile['C']

# all tested global couplings (G in the paper):
wStart = 0
wEnd = 2 + 0.001  # 2
wStep = 0.05
wes = np.arange(wStart + wStep,
                wEnd,
                wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
numW = wes.size  # length(wes);

# ==========================
# Some monitoring info: initialization
N = C.shape[0]
Balance_J9.Se_init = np.zeros((N, numW))
Balance_J9.Si_init = np.zeros((N, numW))
Balance_J9.JI = np.zeros((N, numW))

for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
    Balance_J9.kk = kk  # Just for monitoring
    J = Balance_J9.JOptim(we, C)

sio.savemat('BenjiBalancedWeights.mat',
            {'wes': wes,
             'JI': Balance_J9.JI,
             'Se_init': Balance_J9.Se_init,
             'Si_init': Balance_J9.Si_init})  # save Benji_Balanced_weights wes JI Se_init Si_init
