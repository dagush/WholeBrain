# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#
#  Computes simulations with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#
#  - the optimal coupling (we=2.1) for fitting the placebo condition
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#
# Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
# Gustavo Deco, Josephine Cruzat, Joana Cabral, Gitte M. Knudsen, Robin L. Carhart-Harris, Peter C. Whybrow, Nikos K.
# Logothetis, and Morten L. Kringelbach, Current Biology 28, 3065–3074, October 8, 2018
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from numba import jit

print("Going to use the serotonin 2A receptor (5-HT_{2A}R) transfer functions!")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    phie.recompile(); phii.recompile()


# Regional Drug Receptor Modulation (RDRM) constants for their transfer functions:
# serotonin 2A receptor (5-HT_{2A}R): the neuronal gain function of the model is modulated
# by the 5-HT_{2A}R density
# --------------------------------------------------------------------------
Receptor = 0
wgaini = 0.
wgaine = 0.


# transfer functions:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae = 310.
be = 125.
de = 0.16
@jit(nopython=True)
def phie(x):
    y = (ae*x-be)*(1+Receptor*wgaine)  # for LSD
    # if (y != 0):
    return y/(1-np.exp(-de*y))
    # else:
    #     return 0


# transfer function: inhibitory
ai = 615
bi = 177
di = 0.087
@jit(nopython=True)
def phii(x):
    y = (ai*x-bi)*(1+Receptor*wgaini)  # for LSD
    # if (y != 0):
    return y/(1-np.exp(-di*y))
    # else:
    #     return 0

