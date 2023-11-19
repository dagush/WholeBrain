# ==========================================================================
# ==========================================================================
# ==========================================================================
# Trivial code to test the simplified Hopf normal form
#
# Original code by Christoffer Alexandersen
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# refactored by Gustavo Patow
#
# ==========================================================================
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import csv

# load integrator and neuronal model
# ============================================================================
import simplifiedHopfNormalForm as neuronalModel
import Models.Couplings as Couplings
import Integrators.Heun as scheme
scheme.neuronalModel = neuronalModel
import Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import Utils.simulate_SimOnly as simulator
simulator.integrator = integrator

from setup import *


def readSC(path):
    # Read Budapest coupling matrix
    with open(path, 'r') as f:
        W = list(csv.reader(f, delimiter=','))
    W = np.array(W, dtype=float)
    W = np.maximum(W, W.transpose())
    N = np.shape(W)[0]  # number of nodes
    return W, N


def setupSimulator(TR, dt):
    simulator.dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
    # note: 1e-3 is the length of a millisecond, in seconds,
    # so basically this is a milliseconds to seconds conversion factor
    simulator.dt = dt
    simulator.TR = TR  # Sampling rate of saved simulated BOLD (seconds)
    simulator.t_min = TR / dt  # Skip t_min first samples
    simulator.Tmax = 220.  # Number of (useful) time-points in each fMRI session
    # each time-point is separated by TR seconds => Tmax * TR is the total length, in seconds
    simulator.Toffset = 10.  # Number of initial time-points to skip
    # We simulate Tmax+Toffset time-points with the idea of extracting Tmax useful time-points.
    simulator.recomputeTmaxneuronal()  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal


# --------------------------------------------------
# --------------------------- main -----------------
# --------------------------------------------------
if __name__ == '__main__':
    # Set General Model Parameters
    # ============================================================================
    TR = 2.
    dt = 0.1
    trials = 10

    setupSimulator(TR, dt)
    W_0, N = readSC(loadDataPath + 'CouplingMatrixInTime-000.csv')
    neuronalModel.couplingOp = Couplings.instantaneousDirectCoupling(W_0)

    # randomize initial values
    # ============================================================================
    # dyn_x0 = np.zeros((trials, N))
    # dyn_y0 = np.zeros((trials, N))
    # theta0 = np.random.uniform(0, 2*3.14, (trials, N))
    # R0 = np.random.uniform(0, 1, (trials, N))
    # dyn_x0[:,:] = R0 * np.cos(theta0)
    # dyn_y0[:,:] = R0 * np.sin(theta0)
    dataSavePath = '../../Data_Produced/Progression/'
    time = 0; trial = 0
    # ---------- load pre- data
    path = dataSavePath + f"DynSim/pre_{time}_{trial}.npz"
    with open(path, 'rb') as f:
        npzfile = np.load(f, allow_pickle=True)
        W = npzfile['W']
        y0 = npzfile['y0']
        parameterss = npzfile['parameters'].flatten()
    a = parameterss[0:N].flatten()
    b = parameterss[N:2*N].flatten()
    w = parameterss[2*N:].flatten()

    # Set model parms
    # ============================================================================
    neuronalModel.setParms({'SC': W,
                            'y0': y0,
                            'a': a,
                            'b': b,
                            'omega': w})

    # Simulate!
    # ============================================================================
    simBOLD = simulator.simulateSingleSubject()

    # Plot!
    # ============================================================================
    plt.plot(simBOLD)

    # Check!
    # ============================================================================
    # ---------- load post- data
    path = dataSavePath + f"DynSim/post_{time}_{trial}.npz"
    with open(path, 'rb') as f:
        npzfile = np.load(f, allow_pickle=True)
        sol = np.atleast_1d(npzfile['sol'])[0]
    x = sol['x']
    y = sol['y']
    t = sol['t']

    print('tests done!')

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF