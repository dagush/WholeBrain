# ======================================================================
# Comparison between signal.periodogram and the periodogram manually computed with the FFT.
# Taken from: https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
import functions.randn2 as randn

Fs = 1000.

def buildFunc():
    t = np.arange(0, 1, 1/Fs)
    x = np.cos(2.*np.pi*100.*t) + randn.randn2(t.shape[0])
    return x

def testFFT(x):
    # Obtain the periodogram using fft. The signal is real-valued and has even length. Because the signal is
    # real-valued, you only need power estimates for the positive or negative frequencies. In order to conserve
    # the total power, multiply all frequencies that occur in both sets — the positive and negative frequencies — by
    # a factor of 2. Zero frequency (DC) and the Nyquist frequency do not occur twice.
    N = len(x)
    xdft = np.fft.fft(x)
    xdft = xdft[0:int(N/2)]
    psdx = (1/(Fs*N)) * np.square(np.abs(xdft))
    psdx[1:] = 2*psdx[1:]
    freq = np.arange(0, 1./2., 1/len(x))
    return freq, psdx


def testPeriodogram(x):
    # Compute the periodogram using signal.periodogram
    import scipy.signal as sig
    psp_f, psp_pxx = sig.periodogram(x, axis=0) # nfft=1024, fs=200,
    return psp_f, psp_pxx

# ======================================================================
# ======================================================================
# ======================================================================
if __name__ == '__main__':
    x = buildFunc()
    freqFFT, psdxFFT = testFFT(x)
    freqPeriodogram, psdxPeriodogram = testPeriodogram(x)

    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.suptitle('signal.periodogram func vs Periodogram from FFT: comparison')

    axs[0].plot(freqFFT[1:], 10*np.log10(psdxFFT[1:]))
    axs[0].set_title('Periodogram Using FFT')
    # axs[0].xlabel('Frequency (Hz)')
    axs[0].set(ylabel='Power/Frequency (dB/Hz)')

    axs[1].plot(freqPeriodogram[1:], 10*np.log10(psdxPeriodogram[1:]/1000.))
    axs[1].set_title('Periodogram Using signal.periodogram')
    # axs[1].xlabel('Frequency (Hz)')
    axs[1].set(ylabel='Power/Frequency (dB/Hz)', xlabel='Frequency (Hz)')
    plt.show()

# ======================================================================
# ======================================================================
# ======================================================================
