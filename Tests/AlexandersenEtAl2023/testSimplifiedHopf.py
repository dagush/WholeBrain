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

from setup import *

import loadDelays
import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.measures as measures
dist = measures.pearsonDissimilarity()


def readSC(path):
    # Read Budapest coupling matrix
    with open(path, 'r') as f:
        W = list(csv.reader(f, delimiter=','))
    W = np.array(W, dtype=float)
    W = np.maximum(W, W.transpose())
    N = np.shape(W)[0]  # number of nodes
    return W, N


dt = None
def setupSimulator():
    global dt
    TR = 1.
    dt = 1/1250.  # integration time step, in milliseconds
    simulator.dtt = TR  # Sampling rate of simulated neuronal activity (seconds)
    simulator.dt = dt
    integrator.ds = dt  # recording interval
    simulator.TR = TR  # Sampling rate of saved simulated BOLD (seconds)
    # simulator.t_min = TR / dt  # Skip t_min first samples
    simulator.Tmax = 10.  # Number of (useful) time-points in each fMRI session
    # each time-point is separated by TR seconds => Tmax * TR is the total length, in seconds
    simulator.Toffset = 1.  # Number of initial time-points to skip
    # We simulate Tmax+Toffset time-points with the idea of extracting Tmax useful time-points.
    simulator.recomputeTmaxneuronal()  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal
    print('simulator ready')


def testSingleFrame(time, trial, plot=True):
    # randomize initial values
    # ============================================================================
    # dyn_x0 = np.zeros((trials, N))
    # dyn_y0 = np.zeros((trials, N))
    # theta0 = np.random.uniform(0, 2*3.14, (trials, N))
    # R0 = np.random.uniform(0, 1, (trials, N))
    # dyn_x0[:,:] = R0 * np.cos(theta0)
    # dyn_y0[:,:] = R0 * np.sin(theta0)
    # dataSavePath = '../../Data_Produced/Progression/DynSim-Delays/'
    dataSavePath = '../../Data_Produced/Progression/DynSim-NoDelays/'
    # ---------- load pre- data
    path = dataSavePath + f"pre_{time:.1f}_{trial}.npz"
    with open(path, 'rb') as f:
        npzfile = np.load(f, allow_pickle=True)
        SC = npzfile['W']
        y0 = npzfile['y0']
        parameterss = npzfile['parameters'].flatten()
    a = parameterss[0:N].flatten()
    b = parameterss[N:2*N].flatten()
    w = parameterss[2*N:].flatten()

    # Set model parms
    # ============================================================================
    # Debug:
    # N = 5
    # SC = np.array([[0,1,2,3,4],[1,0,2,3,4],[1,2,0,3,4],[1,2,3,0,4],[1,2,3,4,0]])
    # y0 = np.zeros(N)
    # a = 1. * np.ones(N)
    # b = 1. * np.ones(N)
    # w = np.array([[50,50,50,50,50]])
    # ============================================================================
    neuronalModel.setParms({'SC': SC,
                            'y0': y0,
                            'a': a,
                            'b': b,
                            'omega': w})
    neuronalModel.couplingOp = Couplings.instantaneousDirectCoupling(SC)
    # neuronalModel.couplingOp = Couplings.delayedDirectCoupling(SC, delays, dt)
    # neuronalModel.couplingOp.initConstantPast(y0[0::2])

    # Simulate!
    # ============================================================================
    simBOLD = simulator.simulateSingleSubject().T

    # Plot solution!
    # ============================================================================
    if plot:
        fig, axs = plt.subplots(1, 2, layout='constrained')
        signal = simBOLD[:, ::10].T
        t = np.linspace(0,10,signal.shape[0])
        axs[0].plot(t, signal)
        axs[0].set_title(f'WholeBrain (y:{time},t:{trial})')

    # Check!
    # ============================================================================
    # ---------- load post- data
    path = dataSavePath + f"post_{time:.1f}_{trial}.npz"
    with open(path, 'rb') as f:
        npzfile = np.load(f, allow_pickle=True)
        sol = np.atleast_1d(npzfile['sol'])[0]
    x = sol['x']
    y = sol['y']
    t = sol['t']

    # Plot Reference!
    # ============================================================================
    if plot:
        signal = x[:, ::10].T
        t = np.linspace(0,10,signal.shape[0])
        axs[1].plot(t, signal)
        axs[1].set_title(f'Alexandersen sim (y:{time},t:{trial})')

        plt.show()

    FC_sim = FC.from_fMRI(simBOLD, applyFilters=False)
    FC_read = FC.from_fMRI(x, applyFilters=False)
    res = dist.dist(FC_sim, FC_read)
    return res


# --------------------------------------------------
# --------------------------- main -----------------
# --------------------------------------------------
if __name__ == '__main__':
    # Set General Model Parameters
    # ============================================================================
    setupSimulator()

    trials = 10

    # --------- SC weights
    W_0, N = readSC(loadDataPath + f'CouplingMatrixInTime-000.csv')
    # --------- delays
    transmission_speed = 130.0
    delay_dim = 40  # discretization dimension of delay matrix
    distances = loadDelays.loadDistances(loadDataPath + 'LengthFibers33.csv')
    delays = loadDelays.build_delay_matrix(distances, transmission_speed, N, discretize=delay_dim)
    # couplingDelays = np.rint(delays / dt).astype(np.int32)
    # couplingHorizon = couplingDelays.max() + 1
    # ----------------------------------
    # DEBUG CODE
    delays = np.zeros(delays.shape)
    # ----------------------------------
    neuronalModel.setParms({'delays': delays})
    # neuronalModel.couplingOp = Couplings.delayedDirectCoupling
    # -------- Done! Setup sim

    years = np.arange(0, 35.00001, 3.5)
    res = np.zeros((11, 10))
    for ypos, year in enumerate(years):  # the .00001 is to force the last one...
        for trial in range(10):
            test = testSingleFrame(year, trial, plot=False)
            res[ypos, trial] = test
            print(f'Simulating year {year}, trial {trial} = {test}')
    avg = np.average(res, axis=1)
    # std = np.std(res, axis=1)
    plt.fill_between(years, np.min(res, axis=1), np.max(res, axis=1), facecolor='lightblue')
    plt.plot(years, avg, 'b')
    plt.xticks(years, labels=years)
    plt.title(r'$1-corr(FC^{emp},FC^{sim})$')
    plt.show()

    # Plot one particular "frame"...
    year, trial = np.unravel_index(np.argmax(res), res.shape)
    test = testSingleFrame(years[year], trial, plot=True)
    print(f'Independent simulation of year {year}, trial {trial} = {test}')


    print('tests done!')

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF