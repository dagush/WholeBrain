#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Applies filters to a BOLD signal
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import warnings
import numpy as np
from scipy.signal import butter, detrend, filtfilt
from Observables import demean
from scipy import signal
# from numba import jit

# FILTER SETTINGS (from Gustavo Deco's FCD_LSD_model.m)
# -----------------------------------------------------
TR = None                         # sampling interval. Original was 2.
k = 2                             # 2nd order butterworth filter
flp = None                        # lowpass frequency of filter. Original was .02
fhi = None                        # highpass. Original was 0.1

finalDetrend = False              # Only for compatibility with some Decolab codes...


# @jit(nopython=True)
def BandPassFilter(boldSignal, removeStrongArtefacts=True):
    # Convenience method to apply a filter (always the same one) to all areas in a BOLD signal. For a single,
    # isolated area evaluation, better use the method below.
    (N, Tmax) = boldSignal.shape
    fnq = 1./(2.*TR)              # Nyquist frequency
    Wn = [flp/fnq, fhi/fnq]                                   # butterworth bandpass non-dimensional frequency
    bfilt, afilt = butter(k,Wn, btype='band', analog=False)   # construct the filter
    # bfilt = bfilt_afilt[0]; afilt = bfilt_afilt[1]  # numba doesn't like unpacking...
    signal_filt = np.zeros(boldSignal.shape)
    for seed in range(N):
        if not np.isnan(boldSignal[seed, :]).any():  # No problems, go ahead!!!
            ts = demean.demean(detrend(boldSignal[seed, :]))  # Probably, we do not need to demean here, detrend already does the job...

            if removeStrongArtefacts:
                ts[ts>3.*np.std(ts)] = 3.*np.std(ts)    # Remove strong artefacts
                ts[ts<-3.*np.std(ts)] = -3.*np.std(ts)  # Remove strong artefacts

            signal_filt[seed,:] = filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab

            if finalDetrend:  # Only for compatibility reasons. By default, don't!
                signal_filt[seed,:] = detrend(signal_filt[seed,:])
        else:  # We've found problems, mark this region as "problematic", to say the least...
            warnings.warn(f'############ Warning!!! BandPassFilter: NAN found at region {seed} ############')
            signal_filt[seed,0] = np.nan
    return signal_filt


# @jit(nopython=True)
def filterBrainArea(BOLDSignal, seed, removeStrongArtefacts=True):
    # If we only want to filter ONE area. Observe this computes the whole filter at each function call,
    # so for many areas better use the convenience method above...
    signal_filt = np.zeros(BOLDSignal.shape)
    if not np.isnan(BOLDSignal[seed, :]).any():  # No problems, go ahead!!!
        ts = demean.demean(detrend(BOLDSignal[seed, :]))

        if removeStrongArtefacts:
            ts[ts>3.*np.std(ts)] = 3.*np.std(ts)   # Remove strong artefacts
            ts[ts<-3.*np.std(ts)] = -3.*np.std(ts)  # Remove strong artefacts

        fnq = 1./(2.*TR)                  # Nyquist frequency
        Wn = [flp/fnq, fhi/fnq]           # butterworth bandpass non-dimensional frequency
        bfilt, afilt = signal.butter(k,Wn, btype='band', analog=False)   # construct the filter
        signal_filt = signal.filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab
    else:  # We've found problems, mark this region as "problematic", to say the least...
        warnings.warn(f'############ Warning!!! filterBrainArea: NAN found at region {seed} ############')
        signal_filt[seed, 0] = np.nan
    return signal_filt
