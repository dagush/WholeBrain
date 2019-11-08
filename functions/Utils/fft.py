# ==========================================================================
# ==========================================================================
# fft taken from PyRates [Pyrates_2019]
# Gast, R., Daniel, R., Moeller, H. E., Weiskopf, N. and Knoesche, T. R. (2019). “PyRates – A Python Framework
# for rate-based neural Simulations.” bioRxiv (https://www.biorxiv.org/content/10.1101/608067v2).
import numpy as np
from scipy import signal

def fft(data, dt): #, **kwargs):
    # Compute spectrum
    try:
        # n = data.shape[0]
        # n_two = 1 if n == 0 else 2 ** (n - 1).bit_length()  # Get closest power of 2 that includes n for zero padding
        data_tmp = signal.detrend(data, axis=0)
        freqs = np.linspace(0., 1. / dt, len(data_tmp))  #n_two)  # start, stop, num samples...
        # From Discrete Fourier Transform (numpy.fft) docs:
        # The values in the result follow so-called “standard” order: If A = fft(a, n), then A[0] contains the
        # zero-frequency term (the sum of the signal), which is always purely real for real inputs. Then A[1:n/2]
        # contains the positive-frequency terms, and A[n/2+1:] contains the negative-frequency terms, in order of
        # decreasingly negative frequency.
        spec = np.fft.fft(data_tmp, axis=0) #, n=n_two, **kwargs)
        # So, we cut off the spectrum and frequency arrays since they are mirrored at N/2
        spec = np.abs(spec[0:int(len(spec) / 2),:])
        freqs = freqs[0:int(len(freqs) / 2)]
        return freqs, spec
    except IndexError:
        return np.NaN, np.NaN
