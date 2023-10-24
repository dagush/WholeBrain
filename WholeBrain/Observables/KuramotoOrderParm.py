#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the Kuramoto Order Parameter
#
#  by Gustavo Patow based on the code by Gorka Zamora
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import warnings
import numpy as np
from scipy import signal
from WholeBrain.Observables import BOLDFilters, demean

print("Going to use Kuramoto Order Parameter...")

name = 'Kuramoto'


# @jit(nopython=True)
def from_fMRI(fMRI_signal, applyFilters=True, removeStrongArtefacts=True):
    if not np.isnan(fMRI_signal).any():  # No problems, go ahead!!!
        if applyFilters:
            signal_filt = BOLDFilters.BandPassFilter(fMRI_signal, removeStrongArtefacts=removeStrongArtefacts)
        else:
            signal_filt = fMRI_signal

        (N, Tmax) = fMRI_signal.shape
        # Data structures we are going to need...
        phases_emp = np.zeros([N, Tmax])
        # Time-series of the phases
        for n in range(N):
            Xanalytic = signal.hilbert(signal.detrend(signal_filt[n, :]))  # demean.demean
            phases_emp[n, :] = np.angle(Xanalytic)

        # Compute the temporal evolution of the order parameter
        orderparam = (np.exp(1j * phases_emp)).mean(axis=0)

        return orderparam
    else:
        warnings.warn('############ Warning!!! FC.from_fMRI: NAN found ############')
        # n = fMRI_signal.shape[0]
        return np.nan

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF