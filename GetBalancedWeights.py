#
#
# This prog. optimizes the strengh of the feedback inhibition of the FIC model 
# for varying global couplings (G)
# Saves the steady states and the feedback inhibition (J).
#
# see:
# Deco et al. (2014) J Neurosci.
# http://www.jneurosci.org/content/34/23/7886.long
#
# Adrian Ponce-Alvarez. Refactoring by Gustavo Patow
#--------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
from functions import Balance_J9
from functions import DynamicMeanField as DMF

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
JI=np.zeros((N,numW))
Se_init = np.zeros((N,numW))
Si_init = np.zeros((N,numW))
for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
    J = Balance_J9.JOptim(we, C)
    Se_init[:, kk] = DMF.sn[:, 0]  # Store steady states S^E (after many iterations/simulations)
    Si_init[:, kk] = DMF.sg[:, 0]  # Store steady states S^I
    JI[:,kk]=J[:,0]

sio.savemat('BenjiBalancedWeights-test.mat', #{'JI': JI})
            {'wes': wes,
             'JI': JI,
             'Se_init': Se_init,
             'Si_init': Si_init})  # save Benji_Balanced_weights wes JI Se_init Si_init
