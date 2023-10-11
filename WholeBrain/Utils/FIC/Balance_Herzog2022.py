# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This prog. optimizes the strengh of the feedback inhibition of the FIC model
# for a given global coupling (G)
# Returns the feedback inhibition (J) (and the steady states if wanted).
#
#
# see:
# [HerzogEtAl2022] Neural mass modelling for the masses: Democratising access to
# whole-brain biophysical modelling with FastDMF, Rubén Herzog, Pedro A.M. Mediano,
# Fernando E. Rosas, Andrea I. Luppi, Yonatan Sanz Perl, Enzo Tagliazucchi, Morten
# Kringelbach, Rodrigo Cofré, Gustavo Deco, bioRxiv
# doi: https://doi.org/10.1101/2022.04.11.487903
#
# by Gustavo Patow
# --------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

integrator = None  # in the original paper, Integrator_EulerMaruyama

veryVerbose = False
verbose = True

print("Going to use the Balanced J (FIC) mechanism in HerzogEtAl2022...")


# From [HerzogEtAl2022]:
# We used least squares to find the optimal value of alpha (0.725). However,
# this value gives lower goodness of fit for higher G values, which are
# usually the values where the model better fits empirical data.
# Accordingly, we used weighted least squares, giving 10 times more
# weight to G values larger than 2.1 (close to bifurcation with
# alpha = 0.725), finding an optimal value for alpha = 0.75. This approach,
# as expected, better matches the slope values for high G values and
# also extends the range where stability can be attained by the linear
# approximation.
alpha = 0.75
def computeFirstOrderHeuristic():  # heuristic values
    G = integrator.neuronalModel.getParm({'G'})
    SC = integrator.neuronalModel.getParm({'SC'})
    J = alpha * G * np.sum(SC, axis=0) + 1
    integrator.neuronalModel.setParms({'J': J})
    integrator.recompileSignatures()
    return J


# =====================================
# =====================================
# Computes the optimum of the J_i for a given structural connectivity matrix C and
# a coupling coefficient G, which should be set externally directly at the neuronal model.
def JOptim(N, warmUp = False):
    bestJ = computeFirstOrderHeuristic()
    return bestJ, N


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF