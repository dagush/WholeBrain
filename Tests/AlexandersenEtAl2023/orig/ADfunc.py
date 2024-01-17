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
import symengine as sym
import time as timer

# -----------------------------------------
# helper functions to simulate Alzheimers!
# ----------------------------------------

# -----------------------------------------
# compile hopf normal form model into
# C++ wrapper
# INPUT:
#   Hopf normal form parameters (parameters to be changed must be symengine variables)
#   control_pars - list of symengine variables (parameters that can be changed)
# OUTPUT:
#   DDE - JiTCDDE object
#   y0 - numpy array (initial conditions)
# -----------------------------------------
useDelays = True
def compile_hopf(N, a=False, b=False, delays=False, t_span=(0,10),
             kappa=10, h=1, w=False, decay=-0.01, inter_idx=[], inter_c=1,
             random_init=True, delay_c=1, max_delay=None,  #decay0=0, decay1=1,
             only_a=False, control_pars=()):
    # import must be within function (or else t will not be caught)
    from jitcdde import jitcdde, y, t


    # set default parameter values
    if delays is False:
        delays = np.zeros((N,N))
    if not a:
        a = 1
    if not b:
        b = 1

    # construct adjacency matrix of symbols
    W = [[sym.var(f'W_{i}_{j}') for j in range(N)] for i in range(N)]

    # interhemispheric coupling matrix (scales interhemispheric coupling by inter_c)
    inter_mat = [ [1 for _ in range(N)] for _ in range(N) ]
    for e1, e2 in inter_idx:  # in the current implementation, inter_idx=[], so inter_mat=np.ones((N,N))
        inter_mat[e1][e2] = inter_c

    # if a or b not list then make list (list necessary for symengine variables)
    if not isinstance(a,list):
        a_val = a
        a = [a_val for _ in range(N)]
    if not isinstance(b,list):
        b_val = b
        b = [b_val for _ in range(N)]
    if not isinstance(decay,list):
        decay_val = decay
        decay = [decay_val for _ in range(N)]
    if not isinstance(h,list):
        h_val = h
        h = [h_val for _ in range(N)]

    # TEST DISCARDING B SEMIAXIS
    if only_a:
        b = a

    # define generator of rhs
    def neural_mass():
        for k in range(N):
            if useDelays:
                # define input to node: With delays
                afferent_input = kappa * sum(inter_mat[j][k] * W[j][k] * y(2*j+0, t-delay_c*delays[j,k]) for j in range(N))
            else:
                # define input to node: Without delays
                afferent_input = kappa * sum(inter_mat[j][k] * W[j][k] * y(2 * j + 0) for j in range(N))

            # transform decays
            # decay[k] = decay1*(decay[k]-decay0)

            # dynamics of node k
            # Remember values are interleaved: x is at even positions
            yield decay[k]*y(2*k+0) - w[k]*(a[k]/b[k])*y(2*k+1) \
                     - y(2*k+0)*(y(2*k+0)**2/a[k]**2 + y(2*k+1)**2/b[k]**2) \
                         + h[k] * sym.tanh(afferent_input)
            # y is at odd positions
            yield decay[k]*y(2*k+1) + w[k]*(b[k]/a[k])*y(2*k+0)  \
                     - y(2*k+1)*(y(2*k)**2/a[k]**2 + y(2*k+1)**2/b[k]**2)

    # set up initial conditions
    if random_init:
        theta0 = np.random.uniform(0, 2*3.14, N)
        R0 = np.random.uniform(0,1,N)
    else:
        R0 = np.full((N),1)
        theta0 = np.full((N),0)
    y0 = np.zeros((2*N)) 
    y0[::2] = R0 * np.cos(theta0)
    y0[1::2] = R0 * np.sin(theta0)
    
    # flatten symbolic adjacency matrix as list
    flat_W = list(np.array(W).flatten())

    # include symbolic adjacency matrix as implicit parameters
    # So, the final parms will be [flat_W, a, b, w] because the last three come from main...
    control_pars = [*flat_W, *control_pars]

    # compile DDE, set integration parameters, and store number of nodes
    DDE = jitcdde(neural_mass, n=2*N, control_pars=control_pars, max_delay=max_delay)  
    DDE.compile_C(do_cse=True, chunk_size=int(N*2))  # after vacation this is suddenly slow

    # add number of nodes and initial conditions to DDE object
    DDE.N = N
    DDE.y0 = y0

    return DDE


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
tempDataSavePath = None
def computeMassModel(DDE,  # Compiled DDE
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
    # ------------ save pre- info
    path = tempDataSavePath + f"pre_{runID[0]:.1f}_{runID[1]}.npz"
    with open(path, 'wb') as f:
        np.savez(f, y0=y0, W=W, parameters=parameterss, allow_pickle=True)
    # import must be within function (or else t will not be caught)
    # from jitcdde import jitcdde, y, t

    # ----------------------------------------------------------------
    # check if parameter array given
    if parameterss is False:
        parameterss = np.array([[]])
        parN, num_par = (1, 0)
    else:
        parameterss = np.array(parameterss)  # 1*249 parameters
        parN, num_par = parameterss.shape
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # initialize
    sols = np.empty((parN), dtype='object')  # Here, parN == 1, so there will be a single empty cell

    # ----------------------------------------------------------------
    # set number of nodes and flatten values of adjacency matrix
    N = DDE.N
    flat_num_W = list(W.flatten())  # flattened SC matrix (W), N*N

    # set integration parameters
    DDE.set_integration_parameters(rtol=rtol, atol=atol)
    #DDE.set_integration_parameters(rtol=1e12,atol=1e12, first_step=10**-4, max_step=10**-4, min_step=10**-4)  # test fixed step size

    # start clock
    if display:
        start = timer.time()

    # loop over parameter sets (here, only 1)
    for i in range(parN):
        # set past history as a fixed value (y0)
        DDE.constant_past(y0, time=0.0)

        # ----------------------------------------------------------------
        # set model parameters
        # The final parms will be [flat_W, a, b, w] because the last three come packed from main (3*N parms)...
        # In total, this will be N^2+3N
        # If we have N = 83 => 83*83 + 3*83 = 6889 + 249 = 7138 parms!
        parameters = [*flat_num_W, *parameterss[i,:]]  # add numeric adj. matrix and add model parameters
        try:
            DDE.set_parameters(parameters)
        except:
            print(f'\nThe number of implicit parameters is {num_par}. Make sure that this is reflected in the JiTCDDE compilation.\n')
            return None, None
        # ----------------------------------------------------------------

        # handle initial discontinuities
        DDE.adjust_diff()

        # solve
        data = []
        t = []
        for time in np.arange(DDE.t, DDE.t+t_span[1],  step):  # t_span[1]/step = 11/0.0008 = 13750 timesteps
            data.append(DDE.integrate(time))
            t.append(time)

        # organize data
        data = np.array(data)  # data is a matrix of shape (timesteps, N*2)
        data = np.transpose(data)  # (N*2, timesteps)
        t = np.array(t)  # timesteps (13750)

        # store solution as dictionary, potentially discard y and cut off transients
        sol = {}
        sol['x'] = data[0:2*N:2, t>cutoff]  # this is N * (t >= 1 @ 1250) -> 12500-1 timepoints
        if discard_y:
            sol['y'] = []
        else:
            sol['y'] = data[1:2*N:2, t>cutoff]  # same size: N * (t >= 1 @ 1250) -> 12500-1 timepoints
        sol['t'] = t[t>cutoff]  # same size: (t >= 1 @ 1250) -> 12500-1 timepoints

        # purge past history
        DDE.purge_past()

        # store solution in grid array
        sols[i] = sol

    # ----------------------------------------------------------------
    # display simulation time
    if display:
        end = timer.time()
        print(f'\nElapsed time for all DDE simulations: {end-start} seconds\nElapsed time per DDE simulation: {round((end-start)/parN,4)}')

    # ----------------------------------------------------------------
    # we're done
    # output is a (1) cell with a dictionary {'x':..., 'y':..., 't':...}
    #    'x' and 'y' are arrays N*timepoints
    #    't' is simply the timepoints
    # ------------ save post- info
    path = tempDataSavePath + f"post_{runID[0]:.1f}_{runID[1]}.npz"
    with open(path, 'wb') as f:
        np.savez(f, allow_pickle=True, sol=sols[0])
    return sols


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
def alzheimer(W0, DE, dyn_y0, tau_seed=False, beta_seed=False, seed_amount=0.1, t_spread=False,
              spread_tspan=False,
              spread_y0=False, a0=0.75, ai=1, api=1, aii=1, b0=1, bi=1, bii=1, biii=1, gamma=0, delta=0.95,
              bpi=1, c1=1, c2=1, c3=1, k1=1, k2=1, k3=1, c_init=0, c_min=0,
              rho=10**(-3), a_min=False, a_max=False, b_min=False, a_init=1, b_init=1,
              freqss=np.empty([1,1]), method='RK45', spread_max_step=0.125, as_dict=True,
              spread_atol=10**-6, spread_rtol=10**-3, dyn_atol=10**-6, dyn_rtol=10**-4,
              dyn_step=1/100, dyn_tspan=(0,10), display=False, trials=1,  # SDE=False,
              normalize_row=False, dyn_cutoff=0, feedback=False, kf=1, bii_max=2, adaptive=False):
    # imports
    # from math import e

    # set t_spread if not provided, and add end points if not inluded by user
    if t_spread.size == 0:
        t_spread = [0,spread_tspan[-1]]
    else:
        if 0 not in t_spread:
            t_spread = [0] + t_spread
    Ts_final = t_spread[-1]  # The final simulation time (in years). Here, 35

    # initialize dynamical solutions
    #dyn_sols = np.empty((len(t_spread)), dtype='object')
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
        u = np.array([a0/ai for _ in range(N)])
        up = np.array([0 for _ in range(N)])
        v = np.array([b0/bi for _ in range(N)])
        vp = np.array([0 for _ in range(N)])
        qu = np.array([0 for _ in range(N)])
        qv = np.array([0 for _ in range(N)])
        a = np.array([a_init for _ in range(N)])
        b = np.array([b_init for _ in range(N)])
        c = np.array([c_init for _ in range(N)])
        spread_y0 = [*u, *up, *v, *vp, *qu, *qv, *a, *b, *c, *w0]

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

    # -------------------------------------------------------------------------------
    # spreading dynamics
    # -------------------------------------------------------------------------------
    def rhs(t, y):
        # set up variables as lists indexed by node k
        u = np.array([y[i] for i in range(N)])
        up = np.array([y[i] for i in range(N, 2*N)])
        v = np.array([y[i] for i in range(2*N, 3*N)])
        vp = np.array([y[i] for i in range(3*N, 4*N)])
        qu = np.array([y[i] for i in range(4*N, 5*N)])
        qv = np.array([y[i] for i in range(5*N, 6*N)])
        a = np.array([y[i] for i in range(6*N, 7*N)])
        b = np.array([y[i] for i in range(7*N, 8*N)])
        c = np.array([y[i] for i in range(8*N, 9*N)])

        # update laplacian from m weights
        w = np.array([y[i] for i in range(9*N, 9*N+M)])
        L = np.zeros((N,N))
        for i in range(M):
            n, m = edges[i]
            # set (n,m) in l
            L[n,m] = -w[i]
            L[m,n] = L[n,m]
            # update (n,n) and (m,m) in l
            L[n,n] += w[i]
            L[m,m] += w[i]

        # check if l is defined correctly
        for i in range(N):
            if abs(sum(L[i,:])) > 10**-10:
                print('L is ill-defined')
                print(sum(L[i,:]))
    
        # scale Laplacian by diffusion constant
        L = rho*L
        
        # nodal dynamics
        du, dup, dv, dvp, dqu, dqv, da, db, dc = [[] for _ in range(9)]
        for k in range(N):
            # index list of node k and its neighbours
            neighbours_k = neighbours[k] + [k]

            # heterodimer dynamics
            duk = sum([-L[k,l]*u[l] for l in neighbours_k]) + a0 - ai*u[k] - aii*u[k]*up[k]
            dupk = sum([-L[k,l]*up[l] for l in neighbours_k]) - api*up[k] + aii*u[k]*up[k]
            dvk = pf[k]*sum([-L[k,l]*v[l] for l in neighbours_k]) + b0 - bi*v[k] \
                     - bii*v[k]*vp[k] - biii*up[k]*v[k]*vp[k]
            dvpk = pf[k]*sum([-L[k,l]*vp[l] for l in neighbours_k]) - bpi*vp[k] \
                     + bii*v[k]*vp[k] + biii*up[k]*v[k]*vp[k]
            ## append
            du.append(duk)
            dup.append(dupk)
            dv.append(dvk)
            dvp.append(dvpk)

            # damage dynamics
            dquk = k1*up[k]*(1-qu[k])
            dqvk = k2*vp[k]*(1-qv[k]) + k3*up[k]*vp[k]
            ## append
            dqu.append(dquk)
            dqv.append(dqvk)

            # excitatory-inhibitory dynamics
            dak = c1*qu[k]*(a_max-a[k])*(a[k]-a_min) - c2*qv[k]*(a[k]-a_min)
            dbk = -c3*qu[k]*(b[k]-b_min)
            dck = -c3*qu[k]*(c[k]-c_min)
            ## append
            da.append(dak)
            db.append(dbk)
            dc.append(dck)

        # connecctivity dynamics
        dw = []
        for i in range(M):
            # extract edge
            n, m = edges[i]
            
            # axonopathy dynamcs
            dwi = -gamma * w[i] * (qv[n] + qv[m])
            ## append
            dw.append(dwi)

        # pack right-hand side
        rhs = [*du, *dup, *dv, *dvp, *dqu, *dqv, *da, *db, *dc, *dw]

        return rhs
    # -------------------------------------------------------------------------------

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
        # SOLVE DYNAMICAL MODEL AT T0
        # initialize storage for trial simulations
        dyn_x = []
        dyn_y = []

        # ------------------------------------------------------------------------------
        # solve dynamical model for each trial
        # ------------------------------------------------------------------------------
        for l in range(trials):
            # set initial values
            dyn_y0_l = dyn_y0[l,:]
    
            # update dynamical parameters
            if freqss.size > 0:
                freqs = freqss[:,l] 
                dyn_pars = [[*a, *b, *freqs]]
            else:
                dyn_pars = [[*a, *b]]

            # if told, normalize adj. matrix
            if normalize_row:
                for n in range(N):
                    W_t[n,:] = W_t[n,:] / np.sum(W_t[n,:])

            # ------------------------------------------------------------------------------
            # solve dynamical model at time 0
            # ------------------------------------------------------------------------------
            print(f'\tSolving dynamical model at year {t0} (trial {l+1} of {trials}) ...')
            # input to the computeMassModel functions, for each trial and for each t, are:
            #  DE the compiled Delayed Equation
            #  dyn_y0_l: a 2*N arrays with the initial values for the equation (x,y)
            #            remember that values are interleaved
            #  W_t: the coupling matrix at time t of size N*N
            #  dyn_pars: [a, b, w] each with N values
            #  dyn_tspan: the simulation interval (0, 11) seconds?
            #  dyn_step: the integration step: 0.0008 seconds?
            #  dyn_atol and dyn_rtol: the tolerances (1e-6 and 0.0001)
            #  dyn_cutoff: 1
            #  Added by Gus: runID which is (time, trial)
            dyn_sol = computeMassModel(DE, dyn_y0_l, W_t,
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
            dyn_x_l = dyn_sol[0]['x'] 
            dyn_y_l = dyn_sol[0]['y'] 
            dyn_x.append(dyn_x_l)
            dyn_y.append(dyn_y_l)
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------

        # store all trials in tuple and add to dyn_sols
        dyn_t = dyn_sol[0]['t']  # This has length of timepoints (e.g., 12499)
        dyn_x = np.array(dyn_x)  # trials * N * timepoints
        dyn_y = np.array(dyn_y)  # trials * N * timepoints
        dyn_sol_tup = (dyn_t, dyn_x, dyn_y)
        dyn_sols.append(dyn_sol_tup)
        #dyn_sols[i] = dyn_sol_tup  

        # ------------------------------------------------------------------------------
        # SPREADING MODEL FROM T0 to T
        # if only one time-point, return the spreading initial conditions
        # ------------------------------------------------------------------------------
        if len(t_spread) == 1:
            print('\tOnly one time point in spreading simulation')
            spread_sol['t'] = np.concatenate((spread_sol['t'], [0]))
            spread_sol['u'] = np.concatenate((spread_sol['u'], np.reshape(spread_y0[0:N], (N,1))), \
                                                 axis=1)
            spread_sol['up'] = np.concatenate((spread_sol['up'], np.reshape(spread_y0[N:2*N], (N,1))), \
                                                 axis=1)
            spread_sol['v'] = np.concatenate((spread_sol['v'], np.reshape(spread_y0[2*N:3*N], (N,1))), \
                                                 axis=1)
            spread_sol['vp'] = np.concatenate((spread_sol['vp'], np.reshape(spread_y0[3*N:4*N], \
                                                 (N,1))), axis=1)
            spread_sol['qu'] = np.concatenate((spread_sol['qu'], np.reshape(spread_y0[4*N:5*N], \
                                                (N,1))), axis=1)
            spread_sol['qv'] = np.concatenate((spread_sol['qv'], np.reshape(spread_y0[5*N:6*N], \
                                                (N,1))), axis=1)
            spread_sol['a'] = np.concatenate((spread_sol['a'], np.reshape(spread_y0[6*N:7*N], \
                                                (N,1))), axis=1)
            spread_sol['b'] = np.concatenate((spread_sol['b'], np.reshape(spread_y0[7*N:8*N], \
                                                (N,1))), axis=1)
            spread_sol['c'] = np.concatenate((spread_sol['c'], np.reshape(spread_y0[8*N:9*N], \
                                                (N,1))), axis=1)
            spread_sol['w'] = np.concatenate((spread_sol['w'], np.reshape(spread_y0[9*N:9*N+M], \
                                                (M,1))), axis=1)
        # end simulation at last time point
        if t >= Ts_final:
            break

        # ------------------------------------------------------------------------------
        # set time interval to solve (if adaptive, analyze dynamics here)
        # ------------------------------------------------------------------------------
        if feedback:
            mods = (dyn_x_l**2 + dyn_y_l**2)**(1/2) 
            avg_mod = np.mean(mods, axis=1) 
            if t0==0:
                mod0 = np.mean(avg_mod)
                pf_0 = pf - 1e-5
            if adaptive:
                eqs = kf*(-mod0+avg_mod-pf+pf_0)
                funcs = 1 / (kf*(mod0 + pf - pf_0))
                step_size = np.amin( funcs )
                t = t + step_size
                print(f'\t\tAdaptive step size = {step_size}')
        if not adaptive:
            t = t_spread[i+1]
        spread_tspan = (t0, t)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # solve spreading from time t_(i-1) to t_(i)
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        print(f'\n\tSolving spread model for {spread_tspan} ...')
        sol = solve_ivp(rhs, spread_tspan, spread_y0, method=method,
                         max_step=spread_max_step, atol=spread_atol, rtol=spread_rtol)
        print('\tDone.')

        # ------------------------------------------------------------------------------
        # append spreading solution
        # ------------------------------------------------------------------------------
        spread_sol['t'] = np.concatenate((spread_sol['t'], sol.t))
        spread_sol['u'] = np.concatenate((spread_sol['u'], sol.y[0:N,:]), axis=1)
        spread_sol['up'] = np.concatenate((spread_sol['up'], sol.y[N:2*N,:]), axis=1)
        spread_sol['v'] = np.concatenate((spread_sol['v'], sol.y[2*N:3*N,:]), axis=1)
        spread_sol['vp'] = np.concatenate((spread_sol['vp'], sol.y[3*N:4*N,:]), axis=1)
        spread_sol['qu'] = np.concatenate((spread_sol['qu'], sol.y[4*N:5*N,:]), axis=1)
        spread_sol['qv'] = np.concatenate((spread_sol['qv'], sol.y[5*N:6*N,:]), axis=1)
        spread_sol['a'] = np.concatenate((spread_sol['a'], sol.y[6*N:7*N,:]), axis=1)
        spread_sol['b'] = np.concatenate((spread_sol['b'], sol.y[7*N:8*N,:]), axis=1)
        spread_sol['c'] = np.concatenate((spread_sol['c'], sol.y[8*N:9*N,:]), axis=1)
        spread_sol['w'] = np.concatenate((spread_sol['w'], sol.y[9*N:9*N+M,:]), axis=1)
        spread_sol['disc_t'].append(t)

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