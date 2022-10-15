# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Phase-Interaction Matrix
#
#  Explained at
#  [Deco2019] Awakening: Predicting external stimulation to force transitions between different brain states
#       Gustavo Deco, Josephine Cruzat, Joana Cabral, Enzo Tagliazucchi, Helmut Laufs,
#       Nikos K. Logothetis, and Morten L. Kringelbach
#       PNAS September 3, 2019 116 (36) 18088-18097; https://doi.org/10.1073/pnas.1905534116
#  But defined as this at:
#  [Lopez-Gonzalez2020] Loss of consciousness reduces the stability of brain hubs and the heterogeneity of brain dynamics
#       Ane Lopez-Gonzalez, Rajanikant Panda, Adrian Ponce-Alvarez, Gorka Zamora-Lopez, Anira Escrichs,
#       Charlotte Martial, Aurore Thibaut, Olivia Gosseries, Morten L. Kringelbach, Jitka Annen,
#       Steven Laureys, and Gustavo Deco
#       bioRxiv preprint doi: https://doi.org/10.1101/2020.11.20.391482
#
#  Translated to Python by Xenia Kobeleva
#  Revised by Gustavo Patow
#  Refactored by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
from scipy import signal, stats
# from scipy import stats
from WholeBrain import BOLDFilters
from WholeBrain.Utils import demean

print("Going to use Phase-Interaction Matrix...")

name = 'PhaseInteractionMatrix'

saveMatrix = False
save_file = "./Data_Produced/" + name


discardOffset = 10  # This was necessary in the old days when, after pre-processing, data had many errors/outliers at
# the beginning and at the end. Thus, the first (and last) 10 samples used to be discarded. Nowadays this filtering is
# done at the pre-processing stage itself, so this value is set to 0. Thus, depends on your data...

BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c


# def tril_indices_column(N, k=0):
#     row_i, col_i = np.nonzero(
#         np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
#     Isubdiag = (col_i,
#                 row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
#     return Isubdiag


def from_fMRI(ts, applyFilters = True):  # Compute the Phase-Interaction Matrix of an input BOLD signal
    (N, Tmax) = ts.shape
    npattmax = Tmax - (2*discardOffset-1)  # calculates the size of phfcd matrix

    if not np.isnan(ts).any():  # No problems, go ahead!!!
        # Data structures we are going to need...
        phases = np.zeros((N, Tmax))
        dFC = np.zeros((N, N))
        # PhIntMatr = np.zeros((npattmax, int(N * (N - 1) / 2)))  # The int() is not needed, but... (see above)
        PhIntMatr = np.zeros((npattmax, N, N))
        # syncdata = np.zeros(npattmax)

        # Filters seem to be always applied...
        ts_filt = BOLDFilters.BandPassFilter(ts)  # zero phase filter the data
        for n in range(N):
            Xanalytic = signal.hilbert(demean.demean(ts_filt[n, :]))
            phases[n, :] = np.angle(Xanalytic)

        # Isubdiag = tril_indices_column(N, k=-1)  # Indices of triangular lower part of matrix
        T = np.arange(discardOffset, Tmax - discardOffset + 1)
        for t in T:
            # kudata = np.sum(np.cos(phases[:, t - 1]) + 1j * np.sin(phases[:, t - 1])) / N
            # syncdata[t - 10] = abs(kudata)
            for i in range(N):
                for j in range(N):
                    # print(f'processing {t}: ({i}, {j})')
                    dFC[i, j] = np.cos(adif(phases[i, t - 1], phases[j, t - 1]))
            PhIntMatr[t - discardOffset] = dFC
                # ================== if we need to save the matrix for some later use...
    else:
        warnings.warn('############ Warning!!! PhaseInteractionMatrix.from_fMRI: NAN found ############')
        PhIntMatr = np.array([np.nan])
    # ======== sometimes we need to plot the matrix. To simplify the code, we save it here if needed...
    if saveMatrix:
        import scipy.io as sio
        sio.savemat(save_file + '.mat', {name: PhIntMatr})
    return PhIntMatr

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
