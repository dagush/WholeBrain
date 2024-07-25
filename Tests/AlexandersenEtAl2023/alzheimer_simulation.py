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
# This is the first file in the sequence for the propagation simulation, AFTER preparing setup_Chen_Campbell.py...
#
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
from math import pi
import pickle
import csv

import Progression.progression_functions as AD_func
# import loadDelays  # no delays in our implementation
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
    # Setup modified Hopf model
    # --------------------------------------------------------------------------------------
    AD_func.setupSimulator(neuronalModel, integrator, simulator, Couplings.instantaneousDirectCoupling)

    # --------------------------------------------------------------------------------------
    # Overall simulation settings
    # --------------------------------------------------------------------------------------
    t_years = np.linspace(0,35,11)
    trials = 10  # number of repetitions

    # --------------------------------------------------------------------------------------
    # spread (heterodimer) settings
    # --------------------------------------------------------------------------------------
    tau_nodes = [26, 67]  # left and right entorhinal cortex, toxic initialization
    beta_nodes = [0, 41, 3, 44, 13, 54, 14, 55, 19, 60, 33, 74]  # Mattson et. al (2019) stage I, toxic initialization
    spreadingInitialParms = {
        'tau_seed': tau_nodes,
        'beta_seed': beta_nodes,
        'seed_amount': 0.01,
        'spread_tspan': (0, 35),  # years
        'spread_atol': 10**-6,
        'spread_rtol': 10**-4,
        'spread_max_step': 0.125,
        'spread_y0': False,  # Propagation model initial parms. False gives default setting -> initialized inside AD_func.alzheimer
    }
    # # --------------------------------------------------------------------------------------
    # SPREADING PARAMETERS
    gamma = 0.0
    spreadingParms = {  # SPREADING PARAMETERS
        'rho': 1 * 10**(-3),
        # Amyloid-beta
        'a0': 1 * 2, 'ai': 1 * 2, 'api': 0.75 * 2, 'aii': 1 * 2,
        # tau
        'b0': 1 * 2, 'bi': 1 * 2, 'bii': 1 * 2, 'biii': 6 * 2, 'bpi': 1.33 * 2,
        # concentration-to-damage
        'k1': 1, 'k2': 1, 'k3': 0, 'gamma': gamma,
        # damage-to-NNM
        'c1': 0.8, 'c2': 1.8, 'c3': 0.4,
        # NNM variable parameters
        'a_init': 1, 'b_init': 1, 'a_min': 0.05, 'a_max': 1.95, 'b_min': 0.05,
        'delta': False,
    }

    # --------------------------------------------------------------------------------------
    # Read Budapest coupling matrix
    # --------------------------------------------------------------------------------------
    with open(loadDataPath + 'CouplingMatrixInTime-000.csv', 'r') as f:
        W = list(csv.reader(f, delimiter=','))
    W = np.array(W, dtype=float)
    W = np.maximum(W, W.transpose())
    N = np.shape(W)[0]  # number of nodes

    # --------------------------------------------------------------------------------------
    #  Load distances and compute delays... NOT USED IN THIS IMPLEMENTATION!!!
    # --------------------------------------------------------------------------------------
    # transmission_speed = 130.0
    # delay_dim = 40  # discretization dimension of delay matrix
    # distances = loadDelays.loadDistances(loadDataPath + 'LengthFibers33.csv')
    # delays = loadDelays.build_delay_matrix(distances, transmission_speed, N, discretize=delay_dim)

    # --------------------------------------------------------------------------------------
    # HOPF PARAMETERS
    # --------------------------------------------------------------------------------------
    # _baseDecay = -0.01; _baseH = 5; _kappa = 1;  # kappa is the coupling constant, now called G
    _freqss = np.random.normal(10,1, size=(N,trials))  # samples of frequencies
    _freqss *= 2*pi
    # --------------------------------------------------------------------------------------
    # randomize initial values
    # dyn_y0 is trials (10) * 2*N (2*83)
    _dyn_y0 = np.zeros((trials, 2*N))
    _theta0 = np.random.uniform(0, 2*3.14, (trials, N))
    _R0 = np.random.uniform(0, 1, (trials, N))
    _dyn_y0[:, ::2] = _R0 * np.cos(_theta0)  # dyn_y0 values are interleaved...
    _dyn_y0[:, 1::2] = _R0 * np.sin(_theta0)
    massModelParms = {
        'freqs': _freqss,
        'dyn_y0': _dyn_y0,
    }

    # --------------------------------------------------------------------------------------
    # SOLVE
    # simulate alzheimer's progression
    # --------------------------------------------------------------------------------------
    print('\nSolving alzheimer model...')
    spread_sol, dyn_sols = AD_func.alzheimer(
        # ---- Structural Connectivity
        W,
        # ---- Dynamic parms
        massModelParms=massModelParms,
        # ---- Spread initialization and parms
        spreadingInitialParms=spreadingInitialParms,  # ------------- The propagation model initial parms!
        modelParms=spreadingParms,  # ------------- The propagation model parms!!!
        # ---- more configuration setup
        trials=trials,
        t_spread=t_years,
        method='RK45',
        display=True
        )

    print('\nSaving solutions...')
    dynSaveFile = dyn_save_path.format(f'gamme={gamma}')
    pickle.dump(dyn_sols, open(dynSaveFile, "wb"))
    spreadSaveFile = spread_save_path.format(f'gamme={gamma}')
    pickle.dump(spread_sol, open(spreadSaveFile, "wb"))
    print(f"Done saving:\n   {dynSaveFile}\n   {spreadSaveFile}).")
    print('Done solving.')

    # --------------------------------------------------------------------------------------
    # SAVE SOLUTIONS
    # dump

    # WE'RE DONE

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
