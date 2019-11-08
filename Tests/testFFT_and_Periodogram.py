import numpy as np
import matplotlib.pyplot as plt
import functions.randn2 as randn

Fs = 1000.

def buildFunc():
    t = np.arange(0, 1, 1/Fs)
    x = np.cos(2.*np.pi*100.*t) + randn.randn2(t.shape[0])
    return x

def testFFT(x):
    N = len(x)
    xdft = np.fft.fft(x)
    xdft = xdft[0:int(N/2)]
    psdx = (1/(Fs*N)) * np.square(np.abs(xdft))
    psdx[1:] = 2*psdx[1:]
    freq = np.arange(0, Fs/2, Fs/len(x))

    plt.plot(freq, 10*np.log10(psdx))
    plt.title('Periodogram Using FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()

def testPeriodogram(x):
    import scipy.signal as sig
    # sig.periodogram(x,rectwin(length(x)),len(x),Fs)
    psp_f, psp_pxx = sig.periodogram(x, axis=0) # nfft=1024, fs=200,
    plt.plot(psp_f[1:], 10*np.log10(psp_pxx[1:]))
    plt.title('Periodogram Using priodogram')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.show()

# ======================================================================
# ======================================================================
# ======================================================================
if __name__ == '__main__':
    x = buildFunc()
    testFFT(x)
    testPeriodogram(x)

# ======================================================================
# ======================================================================
# ======================================================================
