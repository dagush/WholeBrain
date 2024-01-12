# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This prog. optimizes the strengh of the feedback inhibition of the FIC model
# for a given global coupling (G)
# Returns the feedback inhibition (J) (and the steady states if wanted).
#
# Requires the definition of an optimization method
#
# To the best of my knowledge, first proposed in:
# [DecoEtAl2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#     How local excitation-inhibition ratio impacts the whole brain dynamics
#     J. Neurosci., 34 (2014), pp. 7886-7898
#     http://www.jneurosci.org/content/34/23/7886.long
#
# by Gustavo Patow
# --------------------------------------------------------------------------
import numpy as np

import WholeBrain.Utils.decorators as decorators
integrator = None  # in the original paper, Integrator_EulerMaruyama

veryVerbose = False
verbose = True

print("Going to use the Balanced J9 (FIC) mechanism...")


import Utils.FIC.Balance_DecoEtAl2014 as Balance_DecoEtAl2014
balancingMechanism = Balance_DecoEtAl2014


# =====================================
# =====================================
# Auxiliary WholeBrain to simplify work: if it was computed, load it. If not, compute (and save) it!
@decorators.loadOrCompute
def Balance_J9(we, N, warmUp=False): # Computes (and sets) the optimized J for Feedback Inhibition Control [DecoEtAl2014]
    print(f"Balancing for we={we}")
    integrator.neuronalModel.setParms({'we': we})
    balancingMechanism.integrator = integrator
    bestJ, nodeCount = balancingMechanism.JOptim(N, warmUp=warmUp)  # This is the Feedback Inhibitory Control
    integrator.neuronalModel.setParms({'J': bestJ.flatten()})
    return {'we': we, 'J': bestJ.flatten()}


def Balance_AllJ9(C, WEs,
                  baseName=None,
                  parallel=False):
    # all tested global couplings (G in the paper):
    # integrator.neuronalModel.setParms({'SC': C})
    N = C.shape[0]
    result = {}
    # if not parallel:
    for we in WEs:  # iterate over the weight range (G in the paper, we here)
        balance = Balance_J9(we, N, baseName.format(np.round(we, decimals=2)))['J'].flatten()
        result[we] = {'we': we, 'J': balance}
    return result

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
