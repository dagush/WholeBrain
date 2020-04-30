#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Applies filters to a BOLD signal
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
from scipy.signal import butter, detrend, filtfilt
from functions.Utils import demean
from scipy import signal


TR = 2.

# FILTER SETTINGS (from Gustavo Deco's FCD_LSD_model.m)
# -----------------------------------------------------
k = 2                             # 2nd order butterworth filter
flp = .02                         # lowpass frequency of filter
fhi = 0.1                         # highpass


def BandPassFilter(boldSignal):
    (N, Tmax) = boldSignal.shape
    fnq = 1./(2.*TR)              # Nyquist frequency
    Wn = [flp/fnq, fhi/fnq]                                   # butterworth bandpass non-dimensional frequency
    bfilt, afilt = butter(k,Wn, btype='band', analog=False)   # construct the filter
    signal_filt = np.zeros(boldSignal.shape)
    for seed in range(N):
        ts = demean.demean(detrend(boldSignal[seed, :]))
        ts[ts>3.*np.std(ts)] = 3.*np.std(ts)    # Remove strong artefacts
        ts[ts<-3.*np.std(ts)] = -3.*np.std(ts)  # Remove strong artefacts
        signal_filt[seed,:] = filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab
    return signal_filt


def filterBrainArea(BOLDSignal, seed):
    ts = signal.detrend(BOLDSignal[seed, :])

    ts[ts>3.*np.std(ts)] = 3.*np.std(ts)   # Remove strong artefacts
    ts[ts<-3.*np.std(ts)] = -3.*np.std(ts)  # Remove strong artefacts

    fnq = 1./(2.*TR)                  # Nyquist frequency
    Wn = [flp/fnq, fhi/fnq]           # butterworth bandpass non-dimensional frequency
    bfilt, afilt = signal.butter(k,Wn, btype='band', analog=False)   # construct the filter
    signal_filt = signal.filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab
    return signal_filt
