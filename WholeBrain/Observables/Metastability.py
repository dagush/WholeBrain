# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Metastability (network synchrony) of a signal, using the Kuramoto order paramter
#
#  Explained at
#  [Shanahan2010] Metastable chimera states in community-structured oscillator networks,
#             Shanahan, M.,
#             Chaos 20 (2010), 013108.
#             DOI: 10.1063/1.3305451
#  [Cabral2011] Role of local network oscillations in resting-state functional connectivity,
#             Joana Cabral, Etienne Hugues, Olaf Sporns, Gustavo Deco,
#             NeuroImage 57 (2011) 130–139,
#             DOI: 10.1016/j.neuroimage.2011.04.010
#  [Cabral2014] Exploring mechanisms of spontaneous functional connectivity in MEG: How delayed network interactions lead to structured amplitude envelopes of band-pass filtered oscillations,
#             Joana Cabral, Henry Luckhoo, Mark Woolrich, Morten Joensson, Hamid Mohseni, Adam Baker, Morten L. Kringelbach, Gustavo Deco,
#             NeuroImage 90 (2014) 423–435
#             DOI: 10.1016/j.neuroimage.2013.11.047
#  and probably many others
#
#  Code by... probably Gustavo Deco, provided by Xenia Kobeleva
#  Translated by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
from scipy import signal, stats
# from scipy import stats
from WholeBrain import BOLDFilters
from WholeBrain.Utils import demean

print("Going to use Metastability (Kuramoto order parm)...")

name = 'Metastability'

ERROR_VALUE = 10
BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


def distance(K1, K2):  # FCD similarity, convenience function
    if not (np.isnan(K1).any() or np.isnan(K2)):  # No problems, go ahead!!!
        return np.abs(K1-K2)
    else:
        return ERROR_VALUE


def from_fMRI(ts_emp, applyFilters = True):  # Compute the Metastability of an input BOLD signal
    # --------------------------------------------------------------------------
    # for isub=1:nsub
    #     for inode=1:nnodes
    #         ts_emp_sub(inode,:)=detrend(ts_emp(inode,:,isub)-mean(ts_emp(inode,:,isub)));
    #         ts_emp_filt(inode,:,isub)=filtfilt(bfilt,afilt,ts_emp_sub(inode,:));
    #         % pw = abs(fft(ts_emp_filt(inode,:,isub)));
    #         % PowSpect(:,inode,isub) = pw(1:floor(Tmax/2)).^2/(Tmax/Cfg.TRsec);
    #         Xanalytic(inode,:) = hilbert(demean(ts_emp_filt(inode,:,isub)));
    #         phases_emp(inode,:,isub) = angle(Xanalytic(inode,:));              % <--
    #     end
    #
    #     T=10:Tmax-10;
    #     sync = zeros(1, numel(T));
    #     for t=T
    #         ku=sum(complex(cos(phases_emp(:,t,isub)),sin(phases_emp(:,t,isub))))/nnodes;
    #         sync(t-9)=abs(ku);
    #     end
    #     % empirical metastability
    #     meta_emp_all(isub)=std(sync(:));
    #     % fc_emp_all(:,:,isub) = corrcoef(ts_emp_filt(:,:,isub)');
    #     % phfcd_emp_all(:,isub)=patternCons(phases_emp(:,:,isub),nnodes,Tmax);
    # end
    # --------------------------------------------------------------------------
    (N, Tmax) = ts_emp.shape
    npattmax = Tmax - 19  # calculates the size of phfcd vector
    # size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...

    if not np.isnan(ts_emp).any():  # No problems, go ahead!!!
        # Data structures we are going to need...
        phases_emp = np.zeros([N, Tmax])
        # sync = np.zeros(npattmax)

        # Filters seem to be always applied...
        ts_emp_filt = BOLDFilters.BandPassFilter(ts_emp)  # zero phase filter the data
        for n in range(N):
            Xanalytic = signal.hilbert(demean.demean(ts_emp_filt[n, :]))
            phases_emp[n, :] = np.angle(Xanalytic)

        T = np.arange(10, Tmax - 10 + 1)
        sync = np.zeros(T.size)
        for t in T:
            ku = np.sum(np.cos(phases_emp[:, t - 1]) + 1j * np.sin(phases_emp[:, t - 1])) / N
            sync[t - 10] = abs(ku)

        # empirical metastability
        meta_emp_all = np.std(sync)
    else:
        warnings.warn(f'############ Warning!!! Metastability.from_fMRI: NAN found ############')
        meta_emp_all = np.nan
    return meta_emp_all


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# ==================================================================
def init(S, N):
    return np.zeros(S)


def accumulate(Mets, nsub, signal):
    Mets[nsub] = signal
    return Mets


def postprocess(Mets):
    return Mets  # nothing to do here


def findMinMax(arrayValues):
    return np.min(arrayValues), np.argmin(arrayValues)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
