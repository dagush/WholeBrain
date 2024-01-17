# --------------------------------------------------------------------------------------
# Simulation of Alzheimer's disease progression
#
# By Christoffer Alexandersen
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# This is the first file in the sequence for the propagation simulation
#
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
import symengine as sym
from math import pi
import pickle
import csv

import AlexandersenEtAl2023.orig.ADfunc as AD_func
import loadDelays
from setup import *


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Here we simulate Alzheimer's' disease Progression
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# --------------------------------------------------
# --------------------------- main -----------------
# --------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------------------
    # dynamical (oscillator) settings
    # --------------------------------------------------------------------------------------
    dyn_step = 1/1250; dyn_atol = 10**-6; dyn_rtol = 10**-4
    dyn_tspan = (0,11)
    t_stamps = np.linspace(0,35,11)
    dyn_cutoff = 1  # time to cutoff for analysis
    trials = 10  # number of repetitions
    delay_dim = 40  # discretization dimension of delay matrix


    # --------------------------------------------------------------------------------------
    # spread (heterodimer) settings
    # --------------------------------------------------------------------------------------
    tau_nodes = [26, 67]  # left and right entorhinal cortex, toxic initialization
    beta_nodes = [0, 41, 3, 44, 13, 54, 14, 55, 19, 60, 33, 74]  # Mattson et. al (2019) stage I, toxic initialization
    seed_amount = 0.01; spread_atol = 10**-6; spread_rtol = 10**-4
    spread_tspan = (0,35); spread_y0 = False  # False gives default setting
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------


    # --------------------------------------------------------------------------------------
    # Read Budapest coupling matrix
    # --------------------------------------------------------------------------------------
    with open(loadDataPath + 'CouplingMatrixInTime-000.csv', 'r') as f:
        W = list(csv.reader(f, delimiter=','))
    W = np.array(W, dtype=float)
    W = np.maximum(W, W.transpose())
    N = np.shape(W)[0]  # number of nodes

    # **************************************************************************************
    # DEBUG code
    # N = 5
    # trials = 1
    # **************************************************************************************

    # --------------------------------------------------------------------------------------
    # HOPF PARAMETERS
    w = [sym.var(f'wf_{n}') for n in range(N)]  # natural frequency
    a = [sym.var(f'a_{n}') for n in range(N)]  # excitatory strength
    b = [sym.var(f'b_{n}') for n in range(N)]  # inhibitory strengh
    decay = -0.01; kappa = 1; h = 5; transmission_speed = 130.0
    control_pars = [*a, *b, *w]  # parameters to be overwriteable in C++ compilation

    freqss = np.random.normal(10,1, size=(N,trials))  # samples of frequencies
    freqss *= 2*pi

    # --------------------------------------------------------------------------------------
    # SPREADING PARAMETERS
    rho = 1 * 10**(-3)
    # AB
    a0 = 1 * 2; ai = 1 * 2; aii = 1 * 2; api = 0.75 * 2
    # tau
    b0 = 1 * 2; bi = 1 * 2; bii = 1 * 2; biii = 6 * 2; bpi = 1.33 * 2
        # concentration-to-damage
    k1 = 1; k2 = 1; k3 = 0; gamma = 0.0
    # damage-to-NNM
    c1 = 0.8 ; c2 = 1.8; c3 = 0.4
    # NNM variable parameters
    a_init = 1; b_init = 1; a_min = 0.05; a_max = 1.95; b_min = 0.05
    delta = False

    # --------------------------------------------------------------------------------------
    #  Load distances and compute delays...
    # --------------------------------------------------------------------------------------
    distances = loadDelays.loadDistances(loadDataPath + 'LengthFibers33.csv')
    delays = loadDelays.build_delay_matrix(distances, transmission_speed, N, discretize=delay_dim)
    # **************************************************************************************
    # DEBUG code
    # W = np.array([[0,1,2,3,4],[1,0,2,3,4],[1,2,0,3,4],[1,2,3,0,4],[1,2,3,4,0]])
    # W = np.identity(W.shape[0])
    # y0 = np.zeros(N)
    # a = 1. * np.ones(N)
    # b = 1. * np.ones(N)
    # w = np.array([[50,50,50,50,50]])
    # delays = 0.01 * (np.ones(delays.shape) - np.identity(delays.shape[0]))
    # **************************************************************************************

    # --------------------------------------------------------------------------------------
    # Compile hopf model
    # --------------------------------------------------------------------------------------
    print('\nCompiling...')
    NeuroModelDE = AD_func.compile_hopf(N, a=a, b=b, delays=delays, t_span=dyn_tspan,
                                         kappa=kappa, w=w, decay=decay, random_init=True,
                                         h=h, control_pars=control_pars)
    print('Done.')

    # --------------------------------------------------------------------------------------
    # randomize initial values
    # dyn_y0 is trials (10) * 2*N (2*83)
    dyn_y0 = np.zeros((trials, 2*N))
    theta0 = np.random.uniform(0, 2*3.14, (trials, N))
    R0 = np.random.uniform(0, 1, (trials, N))
    dyn_y0[:, ::2] = R0 * np.cos(theta0)  # dyn_y0 values are interleaved...
    dyn_y0[:, 1::2] = R0 * np.sin(theta0)

    # ----------------- just some silly debug code
    # plt.imshow(dyn_y0)
    # plt.colorbar()
    # plt.show()
    # **************************************************************************************
    dyn_y0 = np.zeros((trials, 2*N))
    # **************************************************************************************

    # --------------------------------------------------------------------------------------
    # SOLVE
    # simulate alzheimer's progression
    # --------------------------------------------------------------------------------------
    print('\nSolving alzheimer model...')
    spread_sol, dyn_sols = AD_func.alzheimer(
        W, NeuroModelDE, dyn_y0,
        # tau_seed=tau_nodes, beta_seed=beta_nodes,
        seed_amount=seed_amount, trials=trials,
        t_spread=t_stamps, spread_tspan=spread_tspan,
        spread_y0=spread_y0, a0=a0, ai=ai, api=api, aii=aii, b0=b0, bi=bi,
        bii=bii, biii=biii, gamma=gamma,
        delta=delta, bpi=bpi, c1=c1, c2=c2, c3=c3, k1=k1, k2=k2, k3=k3,
        rho=rho, a_min=a_min, a_max=a_max, b_min=b_min, a_init=a_init, b_init=b_init,
        freqss=freqss, method='RK45',
        spread_atol=spread_atol, spread_rtol=spread_rtol, dyn_atol=dyn_atol, dyn_rtol=dyn_rtol,
        dyn_step=dyn_step, dyn_tspan=dyn_tspan, dyn_cutoff=dyn_cutoff, display=True
        )
    print('\nSaving solutions...')
    pickle.dump(dyn_sols, open(dyn_save_path.format(f'gamme={gamma}'), "wb"))
    pickle.dump(spread_sol, open(spread_save_path.format(f'gamme={gamma}'), "wb"))
    print('Done saving.')

    print('Done solving.')

    # --------------------------------------------------------------------------------------
    # SAVE SOLUTIONS
    # dump

    # WE'RE DONE

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
