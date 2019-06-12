#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
from scipy.signal import butter, detrend, filtfilt
from functions import demean


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y


def corr2(a,b):  # 2-D correlation coefficient
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    return r

TR = 2

# FILTER SETTINGS
# ----------------------------------------------------
fnq = 1./(2.*TR)                  # Nyquist frequency
flp = .04                         # lowpass frequency of filter
fhi = 0.07                        # highpass
Wn = [flp/fnq, fhi/fnq]           # butterworth bandpass non-dimensional frequency
k = 2                             # 2nd order butterworth filter
bfilt, afilt = butter(k,Wn, btype='band', analog=False)   # construct the filter


def Hilbert(boldSignal):
    # Get the BOLD phase using the Hilbert transform
    (N, Tmax) = boldSignal.shape
    signal_filt = np.zeros(boldSignal.shape)
    for seed in range(N):
        ts = demean.demean(detrend(boldSignal[seed, :]))
        ts[ts>3*np.std(ts)] = 3*np.std(ts)   # Remove strong artefacts
        ts[ts<-3*np.std(ts)] = -3*np.std(ts)  # Remove strong artefacts
        signal_filt[seed,:] = filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab
    return signal_filt


def FCD(signal):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = signal.shape

    subdiag = np.tril(np.ones((N,N)), -1)
    Isubdiag = np.nonzero(subdiag) # Indices of triangular lower part of matrix

    signal_filt = Hilbert(signal)

    # For each pair of sliding windows calculate the FC at t and t2 and
    # compute the correlation between the two.
    N_windows=len(range(0,190,3))  # This shouldn't be done in Python!!!
    cotsampling=np.zeros([int(N_windows*(N_windows-1)/2)])
    kk=0
    ii2 = 0
    for t in range(0,190,3):
        jj2 = 0
        sfilt = (signal_filt[:, t:t+31]).T  # Extracts a (sliding) window between t and t+30 (included)
        cc = np.corrcoef(sfilt, rowvar=False)  # Pearson correlation coefficients
        for t2 in range(0,190,3):
            sfilt2 = (signal_filt[:, t2:t2+31]).T  # Extracts a (sliding) window between t2 and t2+30 (included)
            cc2 = np.corrcoef(sfilt2, rowvar=False)  # Pearson correlation coefficients
            ca = corr2(cc[Isubdiag],cc2[Isubdiag])  # Correlation between both FC
            if jj2 > ii2:  # Only keep the upper triangular part
                cotsampling[kk] = ca
                kk = kk+1
            jj2 = jj2+1
        ii2 = ii2+1

    return cotsampling
