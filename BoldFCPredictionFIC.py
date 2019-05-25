#
#
# This prog. simulates the FIC model + associated BOLD time series,
# and calculates the fitting between the empirical FC and
# the model prediction for varying global couplings (G)
#
# For comparison with the empirical data, we considered the FC of simulated
# BOLD signals which are obtained by transforming the model synaptic activity
# through a hemodynamic model.
#
# The feedback inhition weights need to be calculated previously
# using Get_balanced_weights.m
#
# It uses: BOLD_HemModel.m
#
# see:
# Deco et al. (2014) J Neurosci.
# http://www.jneurosci.org/content/34/23/7886.long
# Ponce-Alvarez et al. (2015)
# http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.100
# 4445
#
# Adrian Ponce-Alvarez
#--------------------------------------------------------------------------
import numpy as np
import scipy.io as sio
from BOLDHemModel
from randn2 import randn2

def randn(N):
    ra = randn2(N)
    return ra #.reshape(-1,1)

np.random.seed(42) # to have "fixed" results so I can debug...


# HERE --> Load C = the DTI matrix
# N = size(C,1);
# HERE --> Load FC_emp = the empirical FC

CFile = sio.loadmat('Human_66.mat') # load(mstring('Human_66.mat'), mstring('C'), mstring('FC_emp'))
C = CFile['C']
FC_emp = CFile['FC_emp']

N = CFile["Order"].shape[1] # N = size(C, 1)

subdiag = np.tril(np.ones((N,N)), -1)
Isubdiag = np.nonzero(subdiag) # Indexes of all the values below the diagonal.
fc = np.arctanh(FC_emp[Isubdiag])# Vector containing all the FC values below the diagonal, z-transform
dsb = 100# BOLD downsampling rate
dtt = 1e-3# Sampling rate of simulated neuronal activity (seconds)
npairs = Isubdiag[0].size


# Parameters:
#---------------
dt = .1
tmax = 100000
tspan=np.arange(0,tmax+dt,dt) # tspan=0:dt:tmax;
ds = 10# downsampling stepsize
Tds = tmax / (ds * dt)
res = ds * dt / 1000
T = (Tds - 1) * ds * dt / 1000# Total time in seconds (for bold model)


w = 1.4# local recurrence
tau_exc = 100
tau_inh = 10
gamma = 0.641
JN = 0.15
I0 = 0.382
Jexte = 1
Jexti = 0.7
I_exc = I0 * Jexte
I_inh = I0 * Jexti
Io = np.concatenate([I_exc * np.ones(N), I_inh * np.ones(N)])
beta = 0.001# additive ("finite size") noise

# number of stochastic realizations:
nTrials = 5


# transfer functions:
# transfert function: excitatory
#--------------------------------------------------------------------------
ae = 310
be = 125
de = 0.16
def He(x):
    return (ae*x-be)/(1-np.exp(-de*(ae*x-be)))

# transfert function: inhibitory
#--------------------------------------------------------------------------
ai = 615
bi = 177
di = 0.087
def Hi(x):
    return (ai*x-bi)/(1-np.exp(-di*(ai*x-bi)))


# load the Ji weights (previously calculated with Get_balanced_weights.m):
#--------------------------------------------------------------------------
#load Bifurcations_BalancedModel_stochastic wes JI Se_init Si_init

CFile = sio.loadmat('BenjiBalancedWeights.mat') #load 'Benji_Balanced_weights' 'wes' 'JI' 'Se_init' 'Si_init'
wes = CFile['wes'][0]
JI = CFile['JI']
Se_init = CFile['Se_init']
Si_init = CFile['Si_init']

numG=wes.size

# initialization:
fittcorr = np.zeros(numG)
Steady_FR = np.zeros((N, numG))
neuro_act = np.zeros((int(Tds), N))
neuro_act2 = np.zeros((int(Tds), N))
FC_z = np.zeros((npairs, numG))

for kk in range(numG):

    we = wes[kk]

    print('Global coupling =', we)

    # feedback inhibition weights:
    J = JI[:, kk]
    mu0 = np.concatenate([Se_init[:, kk], Si_init[:, kk]])
    Ut = np.zeros((N, int(Tds)))
    Rt = np.zeros((N, int(Tds)))


    # Coupling matrix:
    #----------------------------------
    W11 = JN * we * C + w * JN * np.eye(N,N)
    W12 = np.diag(-J)
    W21 = JN * np.eye(N)
    W22 = -np.eye(N)
    Wmat = np.block([[W11, W12], [W21, W22]])


    cb = np.zeros(Isubdiag[0].size)

    for tr in range(nTrials):

        print('   trial:', tr)
        mu = mu0


        # Warm-Up to reach stationarity:
        #--------------------------------

        for t in range(1000):
            u = Wmat @ mu + Io
            re = He(u[0:N])
            re = gamma * re / 1000.
            ri = Hi(u[N:2*N])
            ri = ri / 1000.
            ke = -mu[0:N] / tau_exc + (1 - mu[0:N]) * re
            ki = -mu[N:2*N] / tau_inh + ri
            kei = np.concatenate([ke, ki])
            mu = mu + dt * kei + np.sqrt(dt) * beta * randn(2 * N)
            mu[mu > 1] = 1
            mu[mu < 0] = 0

        # Simulate dynamical model:
        # -------------------------

        nn = 0
        for t in range(1,tspan.size):
            u = Wmat @ mu + Io
            re = He(u[0:N])
            re = gamma * re / 1000.
            ri = Hi(u[N:2*N])
            ri = ri / 1000.
            ke = -mu[0:N] / tau_exc + (1 - mu[0:N]) * re
            ki = -mu[N:2*N] / tau_inh + ri
            kei = np.concatenate([ke, ki])
            mu = mu + dt * kei + np.sqrt(dt) * beta * randn(2 * N)
            mu[mu > 1] = 1
            mu[mu < 0] = 0

            if np.mod(t, ds) == 0:
                Ut[:, nn] = u[0:N]            #excitatory synaptic activity
                Rt[:, nn] = re                #excitatory firing rate
                nn = nn + 1


        #%%% BOLD Model
        # Friston BALLOON MODEL
        #--------------------------------------------------------------------------
        B = BOLDHemModel(T, Ut[0,:], res)                # B=BOLD activity
        Tnew = B.size
        BOLD_act = np.zeros([Tnew, N])
        BOLD_act[:, 0] = B
        for i in range(1,N):
            B = BOLDHemModel.Model_Friston2003(T, Ut[i, :], res)
            BOLD_act[:,i] = B
        # Downsampling
        bds = BOLD_act[499:Tnew:dsb, :]
        # BOLD correlation matrix = Simulated Functional Connectivity
        #if bds.size != 0:
        Cb = np.corrcoef(bds, rowvar=False)
        cb = cb + np.arctanh(Cb[Isubdiag]) / nTrials           # Vector containing all the FC values below the diagonal
        # Firing rate:
        meanrate = np.mean(Rt, axis=1) * 1000 / gamma
        Steady_FR[:, kk] = Steady_FR[:, kk] + meanrate / nTrials
    # end loop over trials

    Coef = np.corrcoef(cb, fc)
    fittcorr[kk] = Coef[1,0]
    FC_z[:, kk] = cb

import matplotlib.pyplot as plt
plt.plot(wes, fittcorr, label='wes/fittcorr') # plot(wes, fittcorr)
plt.legend()
plt.show()


plt.plot(wes, np.max(Steady_FR, axis=0), 'ro', label='wes/max(Steady_FR)') # plot(wes, max(Steady_FR), mstring('.'))
plt.legend()
plt.show()
