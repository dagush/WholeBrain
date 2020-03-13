# ================================================================================================================
#
# This prog. simply computes the JR model, and returns the associated frequencies and Spectrum or PSD...
#
# by Gustavo Patow to the JR model
# ================================================================================================================
import numpy as np
import functions.Utils.fft as fft
# import importlib
JR = None  # importlib.import_module("functions.Models.JansenRit+FIC")
import functions.Integrator_Euler as integrator
# integrator.neuronalModel = JR
integrator.clamping = False
import functions.BalanceFIC as Balance_J9
Balance_J9.integrator = integrator


tmax = 20.
dt = 5e-5
Tmaxneuronal = int((tmax+dt))


def runSim(Conn):
    N = Conn.shape[0]
    JR.SC = Conn
    JR.initBookkeeping(N, tmax)
    integrator.simulate(dt, Tmaxneuronal)
    v = JR.returnBookkeeping()

    freqs, power = fft.fft(v, JR.ds)  # we make use of linearity of the fft to avoid too high values...
    lowCut = 0
    f = freqs[np.argmax(power[lowCut:,:], axis=0)]
    while np.min(f) == 0. and lowCut < 20:
        lowCut += 1
        f = freqs[np.argmax(power[lowCut:,:], axis=0)]
    p = np.max(power[lowCut:,:], axis=0)

    return f, p, freqs, power, v


def runSim2(Conn):
    import scipy.signal as sig

    N = Conn.shape[0]
    JR.SC = Conn
    JR.initBookkeeping(N, tmax)
    integrator.simulate(dt, Tmaxneuronal)
    v = JR.returnBookkeeping()
    PSP = v[400:,:]

    ##### Analyze PSP
    # analyze signal, get baseline and frequency
    psp_baseline = PSP.mean(axis=0)
    psp_f, psp_pxx = sig.periodogram(PSP-psp_baseline, axis=0) # nfft=1024, fs=200,
    psp_f *= 10./(dt*tmax)  # needed because of...
    psp_peak_freq = psp_f[np.argmax(psp_pxx, axis=0)]
    p = np.max(psp_pxx, axis=0)

    return psp_peak_freq, p, psp_f, psp_pxx, v
