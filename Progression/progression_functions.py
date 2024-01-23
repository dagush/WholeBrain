# --------------------------------------------------------------------------------------
# Simulation of Alzheimer's disease progression
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# Code by Christoffer Alexandersen
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
from scipy.integrate import solve_ivp
import numpy as np
# import symengine as sym
import time as timer

import Progression.ProgrModels.Alexandersen2023 as progrModel

# -----------------------------------------
# helper functions to simulate Alzheimers!
# ----------------------------------------

dt = None
neuronalModel=None
simulator = None
couplingClass = None
def setupSimulator(nM, integrator, sim, coup):
    global dt, neuronalModel, simulator  #, couplingClass
    neuronalModel = nM
    simulator = sim
    # couplingClass = coup
    dt = 1/1250.  # integration time step, in milliseconds
    TR = 1.

    simulator.dtt = TR  # Sampling rate of simulated neuronal activity (seconds)
    simulator.dt = dt
    integrator.ds = dt  # recording interval
    simulator.TR = TR  # Sampling rate of saved simulated BOLD (seconds)
    # simulator.t_min = TR / dt  # Skip t_min first samples
    simulator.Tmax = 10.  # Number of (useful) time-points in each fMRI session
    # each time-point is separated by TR seconds => Tmax * TR is the total length, in seconds
    simulator.Toffset = 1.  # Number of initial time-points to skip
    # We simulate Tmax+Toffset time-points with the idea of extracting Tmax useful time-points.
    simulator.observationVarsMask = [0,1]
    simulator.recomputeTmaxneuronal()  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal
    print('simulator ready')


def testSingleTimePoint(SC, a, b, w, y0):
    # First, set model parms
    # ============================================================================
    neuronalModel.setParms({'SC': SC,
                            'y0': y0,
                            'a': a,
                            'b': b,
                            'omega': w})
    neuronalModel.couplingOp.setParms(SC)
    # Now, simulate!
    # ============================================================================
    simBOLD = simulator.simulateSingleSubject().T
    return simBOLD


# ----------------------------------------------------------------
# computeMassModel: solve DDE
# INPUT:
#     DDE - a jitcdde object
#     y0 - numpy array (initial conditions)
#     parameterss -  numpy array shape: (#runs, #parameters)
#     -> each row is a parameter setting with a parameter in each
#     column
# OUTPUT:
#   sols: (#runs) array with solutions stored as dictionaries
# ----------------------------------------------------------------
def computeMassModel(#DDE,  # Compiled DDE
                     y0,  # Initial values, of size N*2 because we have 2 integration vars
                     W,  # Coupling Matrix (N*N)
                     parameterss=False,  # Here we should receive the model params
                     t_span=(0,10), step=10**-4, atol=10**-6, rtol=10**-4,
                     display=False, discard_y=False, cutoff=0,
                     runID=False):
    # input to the computeMassModel functions, for each trial and for each t, are:
    #   DDE the compiled Delayed Equation
    #   y0: a 2*N arrays with the initial values for the equation (x,y)
    #       remember that values are interleaved
    #   W: the coupling matrix at time t of size N*N
    #   parameterss: [a, b, w] each with N values
    #   t_span: the simulation interval (0, 11) seconds?
    #   step: the integration step: 0.0008 seconds?
    #   atol and rtol: the tolerances (1e-6 and 0.0001)
    #   display and discard_y are not used at the moment...
    #      display is for displaying the elapsed time for each call
    #      discard_y is to replace y's results by [] (empty set)
    #   cutoff: 1
    #   Added by GusP: runID = (time, trial)
    # ---------------------------------------------------------------- DEBUG
    # # dataSavePath = '../../Data_Produced/Progression/DynSim-NoDelays/'
    # # dataSavePath = '../../Data_Produced/Progression/DynSim-Delays/'
    # dataSavePath = '../../Data_Produced/Progression/DynSim-Debug/'
    # # ------------ save pre- info
    # path = dataSavePath + f"pre_{runID[0]:.1f}_{runID[1]}.npz"
    # with open(path, 'wb') as f:
    #     np.savez(f, y0=y0, W=W, parameters=parameterss, allow_pickle=True)
    # ---------------------------------------------------------------- END DEBUG

    # ----------------------------------------------------------------
    # check if parameter array given
    if parameterss is False:
        parameterss = np.array([])
        # parN, num_par = (1, 0)
    else:
        parameterss = np.array(parameterss)[0]  # 1*249 parameters
        # parN, num_par = parameterss.shape
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # initialize
    # sols = np.empty((parN), dtype='object')  # Here, parN == 1, so there will be a single empty cell

    # ----------------------------------------------------------------
    # set number of nodes
    N = W.shape[0]

    # start clock
    if display:
        start = timer.time()

    a = parameterss[0:N].flatten()
    b = parameterss[N:2*N].flatten()
    w = parameterss[2*N:].flatten()

    sol = testSingleTimePoint(W, a, b, w, y0)

    # ----------------------------------------------------------------
    # display simulation time
    if display:
        end = timer.time()
        print(f'\nElapsed time for all DDE simulations: {end-start} seconds\nElapsed time per DDE simulation: {round((end-start),4)}')

    # ----------------------------------------------------------------
    # we're done
    # output is a (1) cell with a dictionary {'x':..., 'y':..., 't':...}
    #    'x' and 'y' are arrays N*timepoints
    #    't' is simply the timepoints
    t = np.arange(dt, simulator.Tmax, dt)
    sols = {'x': sol[:,0,:],
            'y': sol[:,1,:],
            't': t}
    # ---------------------------------------------------------------- DEBUG
    # ------------ save post- info
    # path = dataSavePath + f"post_{runID[0]:.1f}_{runID[1]}.npz"
    # with open(path, 'wb') as f:
    #     np.savez(f, allow_pickle=True, sol=sols[0])
    # ---------------------------------------------------------------- END DEBUG

    return sols


def oneTimePointSpread(spread_sol, spread_y0, N, M):
    spread_sol['t'] = np.concatenate((spread_sol['t'], [0]))
    spread_sol['u'] = np.concatenate((spread_sol['u'], np.reshape(spread_y0[0:N], (N, 1))), axis=1)
    spread_sol['up'] = np.concatenate((spread_sol['up'], np.reshape(spread_y0[N:2 * N], (N, 1))), axis=1)
    spread_sol['v'] = np.concatenate((spread_sol['v'], np.reshape(spread_y0[2 * N:3 * N], (N, 1))), axis=1)
    spread_sol['vp'] = np.concatenate((spread_sol['vp'], np.reshape(spread_y0[3 * N:4 * N], (N, 1))), axis=1)
    spread_sol['qu'] = np.concatenate((spread_sol['qu'], np.reshape(spread_y0[4 * N:5 * N], (N, 1))), axis=1)
    spread_sol['qv'] = np.concatenate((spread_sol['qv'], np.reshape(spread_y0[5 * N:6 * N], (N, 1))), axis=1)
    spread_sol['a'] = np.concatenate((spread_sol['a'], np.reshape(spread_y0[6 * N:7 * N], (N, 1))), axis=1)
    spread_sol['b'] = np.concatenate((spread_sol['b'], np.reshape(spread_y0[7 * N:8 * N], (N, 1))), axis=1)
    spread_sol['c'] = np.concatenate((spread_sol['c'], np.reshape(spread_y0[8 * N:9 * N], (N, 1))), axis=1)
    spread_sol['w'] = np.concatenate((spread_sol['w'], np.reshape(spread_y0[9 * N:9 * N + M], (M, 1))), axis=1)


def concatenateSol(spread_sol, sol, N, M, t):
    spread_sol['t'] = np.concatenate((spread_sol['t'], sol.t))
    spread_sol['u'] = np.concatenate((spread_sol['u'], sol.y[0:N, :]), axis=1)
    spread_sol['up'] = np.concatenate((spread_sol['up'], sol.y[N:2 * N, :]), axis=1)
    spread_sol['v'] = np.concatenate((spread_sol['v'], sol.y[2 * N:3 * N, :]), axis=1)
    spread_sol['vp'] = np.concatenate((spread_sol['vp'], sol.y[3 * N:4 * N, :]), axis=1)
    spread_sol['qu'] = np.concatenate((spread_sol['qu'], sol.y[4 * N:5 * N, :]), axis=1)
    spread_sol['qv'] = np.concatenate((spread_sol['qv'], sol.y[5 * N:6 * N, :]), axis=1)
    spread_sol['a'] = np.concatenate((spread_sol['a'], sol.y[6 * N:7 * N, :]), axis=1)
    spread_sol['b'] = np.concatenate((spread_sol['b'], sol.y[7 * N:8 * N, :]), axis=1)
    spread_sol['c'] = np.concatenate((spread_sol['c'], sol.y[8 * N:9 * N, :]), axis=1)
    spread_sol['w'] = np.concatenate((spread_sol['w'], sol.y[9 * N:9 * N + M, :]), axis=1)
    spread_sol['disc_t'].append(t)
    return spread_sol


def constructSpreadInitialValues(N, w0, modelParms):
    u = modelParms['a0'] / modelParms['ai'] * np.ones(N)  # healthy ABeta
    up = np.zeros(N)                                      # toxic ABeta
    v = modelParms['b0'] / modelParms['bi'] * np.ones(N)  # healthy tau
    vp = np.zeros(N)                                      # toxic tau
    qu = np.zeros(N)                                      # damage ABeta
    qv = np.zeros(N)                                      # damage tau
    a = modelParms['a_init'] * np.ones(N)
    b = modelParms['b_init'] * np.ones(N)
    c = modelParms['c_init'] * np.ones(N)
    spread_y0 = [*u, *up, *v, *vp, *qu, *qv, *a, *b, *c, *w0]
    return spread_y0, a , b


# ----------------------------------------------------------------
# solveDynModel: solve the dynamical model for each trial
# INPUT:
#     ...
# OUTPUT:
#     ...
# ----------------------------------------------------------------
def solveDynModel(trials, dyn_y0, a, b, W_t, freqss,
                  dyn_tspan, dyn_step, dyn_atol, dyn_rtol, dyn_cutoff,
                  t0, t,
                  ):
    # SOLVE DYNAMICAL MODEL AT T0
    # initialize storage for trial simulations
    dyn_x = []
    dyn_y = []
    N = len(dyn_y0)

    # ------------------------------------------------------------------------------
    # solve dynamical model for each trial
    # ------------------------------------------------------------------------------
    for l in range(trials):  # ------- loop over trials
        # set initial values
        dyn_y0_l = dyn_y0[l, :]

        # update dynamical parameters
        if freqss.size > 0:
            freqs = freqss[:, l]
            dyn_pars = [[*a, *b, *freqs]]
        else:
            dyn_pars = [[*a, *b]]

        # ------------------------------------------------------------------------------
        # solve dynamical model at time 0
        # ------------------------------------------------------------------------------
        print(f'\tSolving dynamical model at year {t0} (trial {l + 1} of {trials}) ...')
        # input to the computeMassModel functions, for each trial and for each t, are:
        #    DE the compiled Delayed Equation
        #    dyn_y0_l: a 2*N arrays with the initial values for the equation (x,y)
        #             remember that values are interleaved
        #    W_t: the coupling matrix at time t of size N*N
        #    dyn_pars: [a, b, w] each with N values
        #    dyn_tspan: the simulation interval (0, 11) seconds?
        #    dyn_step: the integration step: 0.0008 seconds?
        #    dyn_atol and dyn_rtol: the tolerances (1e-6 and 0.0001)
        #    dyn_cutoff: 1
        #    Added by Gus: runID which is (time, trial)
        dyn_sol = computeMassModel(dyn_y0_l, W_t,
                                   parameterss=dyn_pars,
                                   t_span=dyn_tspan, step=dyn_step,
                                   atol=dyn_atol, rtol=dyn_rtol,
                                   cutoff=dyn_cutoff,
                                   runID=(t, l))
        # output is a (1) cell with a dictionary {'x':..., 'y':..., 't':...}
        #   'x' and 'y' are arrays N*timepoints
        #   't' is simply the timepoints
        print('\tDone')
        # ------------------------------------------------------------------------------

        # store each trial
        dyn_x_l = dyn_sol['x']
        dyn_y_l = dyn_sol['y']
        dyn_x.append(dyn_x_l)
        dyn_y.append(dyn_y_l)
        # ----------- End trials loop --------------------------------------------------
        # ------------------------------------------------------------------------------

    # store all trials in a tuple and add to dyn_sols
    dyn_t = dyn_sol['t']  # This has length of timepoints (e.g., 12499)
    dyn_x = np.array(dyn_x)  # trials * N * timepoints
    dyn_y = np.array(dyn_y)  # trials * N * timepoints
    dyn_sol_tup = (dyn_t, dyn_x, dyn_y)
    return dyn_sol_tup, dyn_x_l, dyn_y_l


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# simulate multi-timescale alzheimer's model 
# INPUT:
#   W0 - numpy array (N,N), initial adjacency matrix
#   DE - a JiTC*DE object, has to be compiled with
#           2*N implicit parameters
#   dyn_y0 - numpy array (#trials, #variables), initial values for DE
#   optional arguments are spreading parameters and
#       integration parameters
# OUTPUT:
#   spread_sol: dictionary, solutions of spreading model
#   dyn_sols: array of dictionaries, solutions of dynamical model
#               at different time points
# ----------------------------------------------------------------
# ----------------------------------------------------------------
def alzheimer(W0,
              dyn_y0,
              tau_seed=False, beta_seed=False, seed_amount=0.1,
              t_spread=False,
              spread_tspan=False,
              spread_y0=False,  # Propagation model initial values
              modelParms=None,  # Propagation model parameters
              freqss=np.empty([1,1]), method='RK45', spread_max_step=0.125, as_dict=True,
              spread_atol=10**-6, spread_rtol=10**-3, dyn_atol=10**-6, dyn_rtol=10**-4,
              dyn_step=1/100, dyn_tspan=(0,10), display=False, trials=1,  # SDE=False,
              dyn_cutoff=0, kf=1, # bii_max=2, feedback=False,
              adaptive=False):
    # ------------- first, let's instantiate a progression model (i.e., the RHS of a diff eq)
    model = progrModel.Alexandersen2023()
    # ------------- some needed parms
    a0 = modelParms['a0']; ai = modelParms['ai']; b0 = modelParms['b0']; bi = modelParms['bi']
    a_init = modelParms['a_init']; b_init = modelParms['b_init']; c_init = modelParms['c_init']
    a_min = modelParms['a_min']; a_max = modelParms['a_max']; b_min = modelParms['b_min']
    delta = modelParms['delta']

    # set t_spread if not provided, and add end points if not inluded by user
    if t_spread.size == 0:
        t_spread = [0,spread_tspan[-1]]
    else:
        if 0 not in t_spread:
            t_spread = [0] + t_spread
    Ts_final = t_spread[-1]  # The final simulation time (in years). Here, 35

    # initialize dynamical solutions
    dyn_sols = []

    # if only one initial condition given, repeat it for all trials
    if len(dyn_y0.shape) == 1:
        n_vars = dyn_y0.shape[0]
        new_dyn_y0 = np.empty((trials,n_vars))
        for l in range(trials):
            new_dyn_y0[l,:] = dyn_y0
        dyn_y0 = new_dyn_y0

    # construct laplacian, list of edges, and list of neighours
    N = W0.shape[0]  
    M = 0
    edges = []
    neighbours = [[] for _ in range(N)]
    w0 = []
    for i in range(N):
        for j in range(i+1, N):
            if W0[i,j] != 0:
                M += 1
                edges.append((i,j))
                neighbours[i].append(j)
                neighbours[j].append(i)
                w0.append(W0[i,j])

    # construct spreading initial values, spread_y0
    if not spread_y0:
        spread_y0, a, b = constructSpreadInitialValues(N, w0, modelParms)

    # seed tau and beta
    if beta_seed:
        for index in beta_seed:
            beta_index = N+index
            if seed_amount:
                spread_y0[beta_index] = seed_amount
            else:
                spread_y0[beta_index] = (10**(-2)/len(beta_seed))*a0/ai
    if tau_seed:
        for index in tau_seed:
            tau_index = 3*N+index 
            if seed_amount:
                spread_y0[tau_index] = seed_amount
            else:
                spread_y0[tau_index] = (10**(-2)/len(tau_seed))*b0/bi

    # define a and b limits
    if delta:
        a_max = 1 + delta
        a_min = 1 - delta
        b_min = 1 - delta
    elif a_max is not False and a_min is not False and b_min is not False:
        pass
    else:
        print("\nError: You have to either provide a delta or a_min, a_max, and b_min\n")

    # make pf a list (necessary, in case of feedback)
    pf = np.ones((N))

    # initialize spreading solution
    t0 = t_spread[0]
    empty_array = np.array([[] for _ in range(N)])
    empty_arraym = np.array([[] for _ in range(M)])
    spread_sol = {'t': np.array([]), 'u':empty_array, 'up':empty_array, 'v':empty_array,
                  'vp':empty_array, 'qu':empty_array, 'qv':empty_array, 'a':empty_array,
                  'b':empty_array, 'c':empty_array, 'w':empty_arraym, 'w_map': edges,
                  'rhythms':[(w0, [1 for _ in range(N)], [1 for _ in range(N)], t0)],
                  'pf':np.transpose(np.array([pf])), 'disc_t':[0]}

    # ------------------------------------------------------------------------------
    # measure computational time
    # ------------------------------------------------------------------------------
    if display:
        start = timer.time()

    # set initial dynamical model parameters
    W_t = W0

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # SOLVE MULTI-SCALE MODEL FOR TIME>0
    # ------------------------------------------------------------------------------
    t = 0;  i = 0 
    while t < Ts_final + 1:
        dyn_sol_tup, dyn_x_l, dyn_y_l = solveDynModel(trials, dyn_y0, a, b, W_t, freqss,
                                                      dyn_tspan, dyn_step, dyn_atol, dyn_rtol, dyn_cutoff,
                                                      t0, t)
        dyn_sols.append(dyn_sol_tup)

        # ------------------------------------------------------------------------------
        # SPREADING MODEL FROM T0 to T
        # if only one time-point, return the spreading initial conditions
        # ------------------------------------------------------------------------------
        if len(t_spread) == 1:
            print('\tOnly one time point in spreading simulation')
            oneTimePointSpread(spread_sol, spread_y0, N, M)
        # end simulation at last time point
        if t >= Ts_final:
            break

        # ------------------------------------------------------------------------------
        # set time interval to solve (if adaptive, analyze dynamics here)
        # ------------------------------------------------------------------------------
        if not adaptive:
            t = t_spread[i+1]
        spread_tspan = (t0, t)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # solve spreading from time t_(i-1) to t_(i)
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        print(f'\n\tSolving spread model for {spread_tspan} ...')
        # ------------ set last minute models
        modelParms['N'] = N; modelParms['M'] = M
        modelParms['a_min'] = a_min; modelParms['a_max'] = a_max; modelParms['b_min'] = b_min
        modelParms['edges'] = edges; modelParms['neighbours'] = neighbours; modelParms['pf'] = pf
        # ------------ set model parms!
        model.serParms(modelParms)
        # ------------ solve the initial value problem!!!
        sol = solve_ivp(model.dfun, spread_tspan, spread_y0, method=method,
                         max_step=spread_max_step, atol=spread_atol, rtol=spread_rtol)
        print('\tDone.')

        # ------------------------------------------------------------------------------
        # append spreading solution
        # ------------------------------------------------------------------------------
        spread_sol = concatenateSol(spread_sol, sol, N, M, t)

        # extract the parameters for the dynamic model
        a = sol.y[6*N:7*N,-1]
        b = sol.y[7*N:8*N,-1]
        w = sol.y[9*N:9*N+M,-1]

        # construct adjacency matrix at time t
        W_t = np.zeros((N,N))
        for j in range(M):
            n, m = edges[j]
            weight = w[j]
            W_t[n,m] = weight
            W_t[m,n] = weight

        # append dynamic model parameters to rhythms list
        rhythms_i = (W_t, a, b, t)
        spread_sol['rhythms'].append(rhythms_i)
        
        # update spreading initial values, spread_y0, and start of simulation, t0
        spread_y0 = sol.y[:,-1]
        t0 = t
        i += 1

    # display computational time
    if display:
        end = timer.time()
        print(f'\nElapsed time for alzheimer simulations: {end-start} seconds\nElapsed time per time step: {round((end-start)/len(t_spread),4)}')

    # done
    return spread_sol, dyn_sols


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF