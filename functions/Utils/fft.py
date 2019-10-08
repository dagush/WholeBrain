# ==========================================================================
# ==========================================================================
# fft taken from PyRates [Pyrates_2019]
# Gast, R., Daniel, R., Moeller, H. E., Weiskopf, N. and Knoesche, T. R. (2019). “PyRates – A Python Framework
# for rate-based neural Simulations.” bioRxiv (https://www.biorxiv.org/content/10.1101/608067v2).
import numpy as np

def fft(data, dt, **kwargs):
    # Compute spectrum
    try:
        n = data.shape[0]
        n_two = 1 if n == 0 else 2 ** (n - 1).bit_length()  # Get closest power of 2 that includes n for zero padding
        data_tmp = data - np.mean(data)
        freqs = np.linspace(0., 1. / dt, n_two)
        spec = np.fft.fft(data_tmp, n=n_two, axis=0, **kwargs)
        # Cut of PSD and frequency arrays since its mirrored at N/2
        spec = np.abs(spec[:int(len(spec) / 2)])
        freqs = freqs[:int(len(freqs) / 2)]
        return freqs, spec
    except IndexError:
        return np.NaN, np.NaN
