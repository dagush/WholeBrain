#####################################################################################
# Based on:
#   https://github.com/dagush/WholeBrain/tree/6e8ffe77b7c65fa053f4ca8804cd1c8cb025e263/WholeBrain
#
# Adapted/Refactored from Gustavo Patow's code by Albert Juncà
#####################################################################################
import warnings
import numpy as np

from WholeBrain.Observables import BOLDFilters


# Abstract class for Observables. At the moment it has a main method "from_fMRI" that takes the signal and the filter
# as parameters and outputs the result if computable (or None if some problem occurred). Each implementation has to
# define "_compute_from_fMRI" method.
#
# NOTES: Implementation is as this to maximize the portability with the old class based library.
class Observable:
    def __init__(self, **kwargs):
        self.ignoreNaNs = False
        self.applyFilters = True
        self.removeStrongArtefacts = True
        self.setParms(kwargs)

    def setParms(self, parms):
        for parm in parms:
            if hasattr(self, parm):
                setattr(self, parm, parms[parm])  # no problem with shadowing, we do not have state variables here!
            else:
                warnings.warn(f'parameter undefined: {parm} (perhaps not needed?)')

    # Main method to compute the Observable from an fMRI BOLD signal.
    def from_fMRI(self, BOLD_signal):
        # ignoreNaN = 'ignoreNaNs' in kwargs and kwargs['ignoreNaNs']
        sfilt = self.filter(BOLD_signal)
        if not self.ignoreNaNs and np.isnan(sfilt).any():  # applyFilters may produce its own NaNs...
            return np.nan
        sfiltT = sfilt.T
        return self._compute_from_fMRI(sfiltT)

    def from_surrogate(self, BOLD_signal):
        NPARCELLS, Tmax = BOLD_signal.shape
        for seed in range(NPARCELLS):
            BOLD_signal[seed, :] = BOLD_signal[seed, np.random.permutation(Tmax)]
        BOLD_su = BOLD_signal[np.random.permutation(NPARCELLS), :]
        return self.from_fMRI(BOLD_su)

    # I separated this function just in case any descendant of this class wants to apply filters on their own...
    def filter(self, BOLD_signal):  # applyFilters=True, removeStrongArtefacts=True, ignoreNaNs=False):
        # First check that there are no NaNs in the signal. If NaNs found, rise a warning and return None
        # ignoreNaNs = 'ignoreNaNs' in kwargs and kwargs['ignoreNaNs']
        if not self.ignoreNaNs and np.isnan(BOLD_signal).any():
            warnings.warn(f'############ Warning!!! {self.__class__.__name__}.from_fMRI: NAN found ############')
            return np.nan
        # Compute bold filter if needed, if not leave the signal as it is
        # applyFilters = ('applyFilters' not in kwargs) or ('applyFilters' in kwargs and kwargs['applyFilters'])
        sfilt = BOLD_signal
        if self.applyFilters:
            # removeStrongArtefacts = ('removeStrongArtefacts' not in kwargs) or ('removeStrongArtefacts' in kwargs and kwargs['removeStrongArtefacts'])
            sfilt = BOLDFilters.BandPassFilter(BOLD_signal, removeStrongArtefacts=self.removeStrongArtefacts)
        return sfilt

    def _compute_from_fMRI(self, sfilt):
        raise NotImplemented('Should have been implemented by subclass!')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF