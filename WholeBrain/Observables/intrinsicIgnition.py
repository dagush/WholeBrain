# --------------------------------------------------------------------------------------
# Full pipeline for Intrinsic Ignition computation
#
# From:
# [DecoKringelbach2017] Hierarchy of Information Processing in the Brain: A Novel ‘Intrinsic Ignition’ Framework,
# Gustavo Deco and Morten L. Kringelbach, Neuron, Volume 94, Issue 5, 961 - 968
#
# [DecoEtAl2017] Novel Intrinsic Ignition Method Measuring Local-Global Integration Characterizes Wakefulness and
# Deep Sleep, Gustavo Deco, Enzo Tagliazucchi, Helmut Laufs, Ana Sanjuán and Morten L. Kringelbach
# eNeuro 15 September 2017, 4 (5) ENEURO.0106-17.2017; DOI: https://doi.org/10.1523/ENEURO.0106-17.2017
#
# [EscrichsEtAl2019] Characterizing the Dynamical Complexity Underlying Meditation, Escrichs A, Sanjuán A, Atasoy S,
# López-González A, Garrido C, Càmara E, Deco, G. Front. Syst. Neurosci., 10 July 2019
# DOI: https://doi.org/10.3389/fnsys.2019.00027
#
# [EscrichsEtAl2021] Whole-Brain Dynamics in Aging: Disruptions in Functional Connectivity and the Role of the Rich
# Club, Anira Escrichs, Carles Biarnes, Josep Garre-Olmo, José Manuel Fernández-Real, Rafel Ramos, Reinald Pamplona,
# Ramon Brugada, Joaquin Serena, Lluís Ramió-Torrentà, Gabriel Coll-De-Tuero, Luís Gallart, Jordi Barretina, Joan C
# Vilanova, Jordi Mayneris-Perxachs, Marco Essig, Chase R Figley, Salvador Pedraza, Josep Puig, Gustavo Deco
# Cereb Cortex. 2021 Mar 31;31(5):2466-2481. doi: 10.1093/cercor/bhaa367
#
# Code by Gustavo Deco and Anira Escrichs
# Adapted to python by Gustavo Patow
#
# By changing the modality variable we can change the way the ignition is computed:
#   - EventBasedIntrinsicIgnition: computes the FC at each time-point, as explained in [DecoKringelbach2017]
#   and [EscrichsEtAl2019]
#   - PhaseBasedIntrinsicIgnition: uses the phase lock matrix at each time-point, as described in [DecoEtAl2017]
#   and [EscrichsEtAl2021]
# --------------------------------------------------------------------------------------

import numpy as np
import warnings
from scipy import signal
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

from Observables import BOLDFilters, demean
# import WholeBrain.Observables.Integration as Integration

# ==================================
# import the matlab engine. I hate this, but...
# ==================================
import matlab.engine
eng = matlab.engine.start_matlab()
# ==================================

print("Going to use Intrinsic Ignition...")

name = 'intrinsicIgnition'

# ==============================================================
nTRs = 5  # TRs to compute ignition after spontaneous events
PhaseBasedIntrinsicIgnition = 0
EventBasedIntrinsicIgnition = 1
modalityName = ['Phase', 'Event']
# ==============================================================
modality = EventBasedIntrinsicIgnition
# ==============================================================


# @jit(nopython=True)
def adif(a,b):
    if np.abs(a-b)>np.pi:
        c = 2*np.pi - np.abs(a-b)
    else:
        c = np.abs(a-b)
    return c


def dmperm(A):
    (useless1,p,useless2,r) = eng.dmperm(eng.double(A), nargout=4)  # Apply MATLABs dmperm
    outp = np.asarray(p).flatten()
    outr = np.asarray(r).flatten()
    return outp, outr


# # @jit(nopython=True)
def get_components(A):
    if A.shape[0] != A.shape[1]:
        raise Exception('this adjacency matrix is not square')

    if not np.any(A-np.triu(A)):
        A = np.logical_or(A, A.T)
        A = A.astype(np.int64)

    # if main diagonal of adj do not contain all ones, i.e., autoloops
    if np.sum(np.diag(A)) != A.shape[0]:
        # the main diagonal is set to ones
        A = np.logical_or(A, np.eye(A.shape[0]))
        A = A.astype(np.int64)

    # i = Integration.IntegrationFromFC_Fast(A, nbins=20)
    p, r = dmperm(A)
    # p indicates a permutation (along rows and columns)
    # r is a vector indicating the component boundaries
    # List including the number of nodes of each component. ith entry is r(i+1)-r(i)
    comp_sizes = np.diff(r)
    # Number of components found.
    num_comps = np.size(comp_sizes)
    # initialization
    comps = np.zeros(A.shape[0])
    # first position of each component is set to one
    comps[r[0:num_comps].astype(int)-1] = np.ones(num_comps)
    # cumulative sum produces a label for each component (in a consecutive way)
    comps = np.cumsum(comps)
    # re-order component labels according to adj.
    comps[p.astype(int)-1] = comps

    return comps, comp_sizes

# # @jit
def computePhases(nodeSignal, N, Tmax):
    phases = np.zeros((N, Tmax))
    for seed in range(N):  # obtain phases for each seed, i.e., each node
        Xanalytic = signal.hilbert(demean.demean(nodeSignal[seed, :]))
        phases[seed,:] = np.angle(Xanalytic)
    return phases


# @jit(nopython=True)
def computeEvents(nodeSignal, N, Tmax):
    events = np.zeros((N, Tmax))
    # Let's compute the events. From [DecoEtAl2017]:
    # An intrinsic ignition event for a given brain region is defined by binarizing the transformed
    # functional time series (BOLD fMRI) into z-scores z_i(t) and imposing a threshold \theta such that
    # the binary sequence \sigma_i(t) = 1 if z_i(t) > \theta, and is crossing the threshold from below,
    # and \sigma_i(t) = 0 otherwise
    for seed in range(N):  # obtain events for each seed, i.e., each node
        tise = nodeSignal[seed, :Tmax]

        # This part of the code computes the \sigma_i as a difference between the binary series ev1
        # and itself shifted (to the right) by 1, to fulfill the "from below" condition: imagine that
        # we have    ev1 = [0 0 1 1 1 1 0],
        # and then   ev2 = [0 0 0 1 1 1 1]
        # the difference will be
        #        ev1-ev2 = [0 0 1 0 0 0 -1]
        # Then, the verification (ev1-ev2) > 0 will give
        #  (ev1-ev2) > 0 = [0 0 1 0 0 0 0],
        # assuming 0 is False and 1 is True. As we see, we have a 1 at the third position, indicating
        # that that's where the event (signal above threshold) started.
        ev1 = tise > (np.std(tise)+np.mean(tise))
        ev1 = ev1.astype(np.int64)  # conversion to int because numpy does not like boolean subtraction...
        ev2 = np.roll(ev1, shift=1); ev2[0] = 0  # originally, it was ev2 = [0 ev1(1:end-1)])
        events[seed,:] = (ev1-ev2) > 0
    return events


# @jit(nopython=True)
def computeEventsMax(events):
    return events.shape[1]  # int(np.max(np.sum(events, axis=1)))


# # @jit(nopython=True)
def computePhaseBasedIntegration(phases, N, Tmax):
    # Integration
    # -----------
    # obtain 'events connectivity matrix' and integration value (integ):
    # for each time point:
    #    Compute the phase lock matrix P_{ij}(t), which describes the state of pair-wise phase synchronization
    #    at time t between regions i and k (from [EscrichsEtAl2021]).
    phasematrix = np.zeros((N,N))  # nodes x nodes
    integ = np.zeros(Tmax)
    for t in range(Tmax):  # (Tmax-1,Tmax)
        print(f"===========  Computing for t={t}/{Tmax} ")
        for i in range(N):
            for j in range(N):
                phasematrix[i,j] = np.exp(-3*adif(phases[i,t], phases[j,t]))
        cc = phasematrix - np.eye(N)
        PR = np.arange(0, 0.99, 0.01)
        cs = np.zeros(len(PR))
        for pos, p in enumerate(PR):
            print(f'Processing PR={p} (t={t}/{Tmax})')
            A = (np.abs(cc) > p).astype(np.int64)
            comps, csize = get_components(A)
            cs[pos] = np.max(csize)
        integ[t] = np.sum(cs)/100 / N  # area under the curve / N = mean integration
    return integ


# # @jit(nopython=True)
def computeEventBasedIntegration(events, N, Tmax):
    # Integration
    # -----------
    # obtain 'events connectivity matrix' and integration value (integ)
    eventsmatrix = np.zeros([N,N])  # nodes x nodes
    integ = np.zeros(Tmax)
    for t in range(Tmax):
        for i in range(N):
            for j in range(N):
                eventsmatrix[i,j] = events[i,t] * events[j,t]
        cc = eventsmatrix - np.eye(N)
        comps, csize = get_components(cc)
        integ[t] = max(csize)/N
    return integ


# @jit(nopython=True)
def eventBasedTrigger(events, integ, N, Tmax):
    # event trigger
    eventCounter = np.zeros(N, dtype=np.uint)  # matrix with 1 x node and number of events in each cell
    IntegStim = np.zeros((N, nTRs-1, computeEventsMax(events)))  # (nodes x (nTR-1) x events)
    # save events and integration values for nTRs after the event
    for seed in range(N):
        flag = 0
        for t in range(Tmax):
            # detect first event (nevents = matrix with (1 x node) and number of events in each cell)
            if events[seed,t] == 1 and flag == 0:  # if there is an event...
                flag = 1  # ... initialize the flag, and ...
                # real events for each subject
                eventCounter[seed] += 1  # ... count it!!!
            # save integration value for nTRs after the first event (nodes x (nTRs-1) x events)
            if flag > 0:
                # integration for each subject
                IntegStim[seed, flag-1, int(eventCounter[seed])-1] = integ[t]
                flag = flag + 1
            # after nTRs, set flag to 0 and wait for the next event (then, integ saved for (nTRs-1) events)
            if flag == nTRs:
                flag = 0
    return eventCounter, IntegStim


# @jit(nopython=True)
def meanAndStdDevIgnition(eventCounter, IntegStim, N):
    # mean and std of the max ignition in the nTRs for each subject and for each node
    mevokedinteg = np.zeros(N)
    stdevokedinteg = np.zeros(N)
    varevokedinteg = np.zeros(N)
    for seed in range(N):
        # Mean integration is called ignition [EscrichsEtAl2021]
        mevokedinteg[seed] = np.mean(np.max(np.squeeze(IntegStim[seed, :, 0:eventCounter[seed]]), axis=0))
        # The standard deviation is called metastability [EscrichsEtAl2021]. Greater metastability in a brain
        # area means that this activity changes more frequently across time within the network.
        stdevokedinteg[seed] = np.std(np.max(np.squeeze(IntegStim[seed, :, 0:eventCounter[seed]]), axis=0))
        varevokedinteg[seed] = np.var(np.max(np.squeeze(IntegStim[seed, :, 0:eventCounter[seed]]), axis=0))
    return mevokedinteg, stdevokedinteg, varevokedinteg


# output: mean and variability of ignition across the events for a single subject (SS = single subject)
# 'mevokedintegSS', 'stdevokedintegSS',
#  Spontaneous events for each subject
# 'SubjEvents'
# mean and variability of ignition across nodes for each single subject (AN = across nodes)
# 'mignitionAN', 'stdignitionAN'
def computeIgnition(nodeSignal):
    (N, Tmax) = nodeSignal.shape
    # Both alternatives, event-based and phase-based, require the events for the ignition.
    events = computeEvents(nodeSignal, N, Tmax)

    # Once we have the events, we can compute the corresponding integrations.
    if modality == PhaseBasedIntrinsicIgnition:
        phases = computePhases(nodeSignal, N, Tmax)
        integ = computePhaseBasedIntegration(phases, N, Tmax)
    elif modality == EventBasedIntrinsicIgnition:
        integ = computeEventBasedIntegration(events, N, Tmax)
    else:
        raise Exception("Sorry, modality not recognized")

    eventCounter, IntegStim = eventBasedTrigger(events, integ, N, Tmax)
    # eventCounter=eventCounter_IntegStim[0]; IntegStim=eventCounter_IntegStim[1]

    mevokedinteg, stdevokedinteg, varevokedinteg = meanAndStdDevIgnition(eventCounter, IntegStim, N)
    # mevokedinteg=mevokedinteg_stdevokedinteg_varevokedinteg[0]; stdevokedinteg=mevokedinteg_stdevokedinteg_varevokedinteg[1]; varevokedinteg=mevokedinteg_stdevokedinteg_varevokedinteg[2]

    # mean and std ignition across events for each subject in each node (Single Subject -> SS)
    # Done for compatibility with Deco's code
    mevokedintegSS = mevokedinteg
    stdevokedintegSS = stdevokedinteg
    fanofactorevokedintegSS = varevokedinteg/mevokedinteg  # calculate Fano factor var()./mevokedinteg
    # mean and std ignition for a subject across nodes(AN)
    mignitionAN = np.mean(mevokedinteg)

    return {'mevokedinteg': mevokedintegSS,
            'stdevokedinteg': stdevokedintegSS,
            'fanofactorevokedinteg': fanofactorevokedintegSS,
            'mignition': mignitionAN}


def from_fMRI(nodeSignal, applyFilters=True, removeStrongArtefacts=True):
    if not np.isnan(nodeSignal).any():  # No problems, go ahead!!!
        if applyFilters:
            signal_filt = BOLDFilters.BandPassFilter(nodeSignal, removeStrongArtefacts=removeStrongArtefacts)
        else:
            signal_filt = nodeSignal
        cc = computeIgnition(signal_filt)
        return cc
    else:
        warnings.warn('############ Warning!!! intrinsicIgnition.from_fMRI: NAN found ############')
        # n = nodeSignal.shape[0]
        return np.nan


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
