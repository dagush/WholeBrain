# --------------------------------------------------------------------------------------
# Simply utility library for fMRI (and other time-based signals) manipulation
#
# By Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
from scipy import stats


def sortByCriterion(fMRIs, typeAvg):
    S = len(fMRIs.keys())
    N, T = fMRIs[list(fMRIs.keys())[0]].shape
    if typeAvg == 'avgNodes':  # This is organized by subjects (len(res) == S) -> S x (NxT) == {S: NxT}
        res = fMRIs
    elif typeAvg == 'avgSubjects':  # This is organized by nodes (len(res) == N) -> N x (SxT) == {N: SxT}
        res = {}
        for node in range(N):
            res[node] = np.zeros((S,T))
            for subj, name in enumerate(fMRIs.keys()):
                res[node][subj, :] = fMRIs[name][node, :]
    elif typeAvg == 'avgCohort':  # All together (len(res) == 1) -> S*N x T == {'avgCohort': (S*N)xT}
        allThefMRI = np.zeros((S*N, T))
        for subj, name in enumerate(fMRIs.keys()):
            allThefMRI[subj*N:(subj+1)*N, :] = fMRIs[name]
        res = {typeAvg: allThefMRI}
    else:
        raise Exception(f'Unrecognized type of average: {typeAvg}')
    return res


def zScore(fMRIs):
    zScored = {}
    for s in fMRIs:
        zScored[s] = stats.zscore(fMRIs[s])
    return zScored


def normalizeLike(ref_fMRI, target_fMRI):
    normalized = stats.zscore(target_fMRI) * np.std(ref_fMRI, axis=0) + np.mean(ref_fMRI, axis=0)
    return normalized


def getMeanAndStd(cohort_fMRI):
    res = {}
    for subj in cohort_fMRI:
        fMRI = cohort_fMRI[subj]
        res[subj] = np.empty((fMRI.shape[0],2))
        for n in range(fMRI.shape[0]):
            roi_signal = fMRI[n]
            res[subj][n] = [np.mean(roi_signal), np.std(roi_signal)]
    return res


# https://medium.com/@surajyadav37839/how-to-normalize-a-signal-to-zero-mean-and-unit-variance-e88d640aacf8
# Assume signal is stored in a variable called "signal"
# normalized_signal = (signal - np.mean(signal)) / np.std(signal)
def renormalizeCohort(cohort_fMRI, meanAndStd):
    for subj in cohort_fMRI:
        fMRI = cohort_fMRI[subj]
        for n in range(fMRI.shape[0]):
            fMRI_RoI = cohort_fMRI[subj][n]
            cohort_fMRI[subj][n] = (fMRI_RoI - np.mean(fMRI_RoI)) / np.std(fMRI_RoI) * meanAndStd[subj][n][1] + meanAndStd[subj][n][0]
    return cohort_fMRI

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF