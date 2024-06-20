# =======================================================================
# Turbulence framework, from:
# Gustavo Deco, Morten L. Kringelbach, Turbulent-like Dynamics in the Human Brain,
# Cell Reports, Volume 33, Issue 10, 2020, 108471, ISSN 2211-1247,
# https://doi.org/10.1016/j.celrep.2020.108471.
# (https://www.sciencedirect.com/science/article/pii/S2211124720314601)
#
# Part of the Thermodynamics of Mind framework:
# Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
# Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
# https://doi.org/10.1016/j.tics.2024.03.009
#
# Code by Gustavo Deco, 2020.
# Translated by Marc Gregoris, May 21, 2024
# Refactored by Gustavo Patow, June 9, 2024
# =======================================================================
# from scipy.signal import butter, filtfilt, hilbert
# import warnings
import numpy as np
from scipy import signal
# import WholeBrain.Observables.BOLDFilters as BOLDFilters
from WholeBrain.Observables.observable import Observable
from WholeBrain.Utils import MatlabTricks as MTricks


lambda_val = 0.18


# ================================================================================================================
# Main Turbulence class.
# ================================================================================================================
class Turbulence(Observable):
    def __init__(self, COGDist, **kwargs):
        super().__init__(**kwargs)
        self._computeExpLaw(COGDist)

    # -------------------------------------------------------------------------
    def _computeExpLaw(self, SchaeferCOG):
        N = SchaeferCOG.shape[0]
        # Compute the distance matrix
        rr = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                rr[i, j] = np.linalg.norm(SchaeferCOG[i, :] - SchaeferCOG[j, :])
        # Build the exponential-distance matrix
        Cexp = np.exp(-lambda_val * rr)
        np.fill_diagonal(Cexp, 1)
        self.Cexp = Cexp

    # -------------------------------------------------------------------------
    def _compute_from_fMRI(self, fMRI):
        cc = self.compTurbulence(fMRI.T)
        return cc

    # -------------------------------------------------------------------------
    def compTurbulence(self, ts):
        NPARCELLS, Tmax = ts.shape
        # Initialization of results-storing data
        enstrophy = np.zeros((NPARCELLS, Tmax))
        Phases = np.zeros((NPARCELLS, Tmax))

        for seed in range(NPARCELLS):
            Xanalytic = signal.hilbert(ts[seed,:])
            Xanalytic = Xanalytic - np.mean(Xanalytic)
            Phases[seed, :] = np.angle(Xanalytic)

        for i in range(NPARCELLS):
            sumphases = np.nansum(np.tile(self.Cexp[i, :], (Tmax,1)).T * np.exp(1j * Phases), axis=0) / np.nansum(self.Cexp[i, :])
            enstrophy[i] = np.abs(sumphases)

        Rspatime = np.nanstd(enstrophy)
        Rspa = np.nanstd(enstrophy, axis=1).T
        Rtime = np.nanstd(enstrophy, axis=0)
        acfspa = MTricks.autocorr(Rspa, 100)
        acftime = MTricks.autocorr(Rtime, 100)

        return {
            'Rspatime': Rspatime,
            'Rspa': Rspa.T,
            'Rtime': Rtime,
            'acfspa': acfspa,
            'acftime': acftime,
        }


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF