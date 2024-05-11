# --------------------------------------------------------------------------------------
# Full pipeline for Leading Eigenvector Dynamics Analysis (LEiDA)
#
# From:
# [Cabral et al 2017] Cabral, J., Vidaurre, D., Marques, P. et al. Cognitive performance in healthy older adults
# relates to spontaneous switching between states of functional connectivity during rest. Sci Rep 7, 5135 (2017).
# https://doi.org/10.1038/s41598-017-05425-7
#
# Code by Joana Cabral (modified by Gustavo Deco)
# Translated by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import sklearn.metrics as metrics

import WholeBrain.Utils.decorators as decorators
import WholeBrain.Observables.BOLDFilters as BOLDFilters
import WholeBrain.Observables.LEigen as LEigen

import WholeBrain.Utils.dunns as dunns


print("Going to use LEiDA...")

save_folder = None

# --------------------------------------------------------------------------------------
#  Definitions
# --------------------------------------------------------------------------------------
# Set maximum/minimum number of clusters
# There is no fixed number of states the brain can display
# Extending depending on the hypothesis of each work
maxk = None  # set by the application
mink = None  # set by the application
# ------------ diferent sorting criterions...
scoringCriterions = {}
scoringCriterions['Dunn (fast) score'] = dunns.dunn_fast
scoringCriterions['Silhouette score'] = metrics.silhouette_score
scoringCriterions['Davies-Bouldin score'] = metrics.davies_bouldin_score
scoringCriterions['Calinski-Harabasz score'] = metrics.calinski_harabasz_score
# selectedCriteriorn = 'Dunn (fast) score'



# --------------------------------------------------------------------------------------
#  Score-plotting function...
# --------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plotPerformances(clussteringResults):
    numCols = int(np.ceil(len(scoringCriterions)/2))
    fig, axs = plt.subplots(numCols, 2, sharex=True)
    axis = axs.flat
    for pos, crit in enumerate(scoringCriterions):
        score = [clussteringResults[k]['scores'][crit] for k in clussteringResults]
        # ------ just report some results...
        ind_max = np.argmax(score) + mink
        print(f'Best clustering solution: {ind_max} clusters with {crit}')
        # ------ and plot them!
        axis[pos].plot(range(mink, maxk), score)
        axis[pos].set_title(crit)
        axis[pos].xaxis.set_major_locator(MaxNLocator(integer=True))
    for pos in range(numCols, len(scoringCriterions)):
        axis[pos].set_xlabel('Cluster #')
    plt.show()



# --------------------------------------------------------------------------------------
#  Convenience functions
# --------------------------------------------------------------------------------------
def packEigenvectors(LEigenvectors):
    # LEigs must a matrix containing all the eigenvectors:
    # Columns: N_areas are brain areas (variables)
    # Rows: T*NumSubjects are all the time points (independent observations)

    # As there is some discarded timepoints in the Phase Interaction Matrix, let's compute the first one and use its size
    NumSubjects = len(LEigenvectors)
    first = next(iter(LEigenvectors))
    N, T = LEigenvectors[first].shape
    LEigs = np.zeros((N, NumSubjects * T))
    for pos, s in enumerate(LEigenvectors):  # Loop over subjects
        Vs = LEigenvectors[s]
        LEigs[:, pos*T:(pos+1)*T] = Vs
    return LEigs


def averageProbabilities(probs, numClusters, subset=None):
    actualProbs = probs if subset is None else subset
    avg = np.zeros(numClusters)
    for s in actualProbs:
        avg += probs[s]
    avg /= np.sum(avg)  # re-normalize probabilities (Sum == 1)
    return avg


# --------------------------------------------------------------------------------------
# LEiDA processing functions
# --------------------------------------------------------------------------------------
@decorators.loadOrCompute
def computeLEigens(BOLDsignal):
    Vs = LEigen.from_fMRI(BOLDsignal, applyFilters=True, removeStrongArtefacts=True)
    return {'Vs': Vs}


# Compute the Leading Eigenvectors from the BOLD datasets
def computeEigenvectors(BOLDsignals):
    NumSubjects = len(BOLDsignals)
    print(f'\nGoing to compute Leading Eigenvectors for BOLD signals for {NumSubjects} subjects\n')

    LEigs = {}  #np.zeros((N, NumSubjects * T))
    for pos, s in enumerate(BOLDsignals):  # Loop over subjects
        if verbose: print(f'   Processing signal {pos+1}/{NumSubjects} Subject: {s}: ', end='', flush=True)
        Vs = computeLEigens(BOLDsignals[s], save_folder.format(s))['Vs']
        if np.isnan(Vs).any():  # Problems, full stop!!!
            raise Exception(f'############ Warning!!! LEiDA: NAN found @ {s} (=={pos}) ############')
        LEigs[s] = Vs  #[:, pos*T:(pos+1)*T] = Vs
    return LEigs


# ------------ Cluster the Leading Eigenvectors
def clusterEigenvectors(LEigenvect):
    print('Clustering Eigenvectors')
    LEigs = packEigenvectors(LEigenvect)
    Kmeans_results = {k:dict() for k in range(mink,maxk)}
    for k in range(mink,maxk):
        kmeans = KMeans(n_clusters=k).fit(LEigs.T)
        Kmeans_results[k]['clusterer'] = kmeans
        Kmeans_results[k]['labels'] = kmeans.labels_
        Kmeans_results[k]['centers'] = kmeans.cluster_centers_
    return LEigs, Kmeans_results


# ------------ Score the clusters
def scoreClusters(LEigs, Kmeans_results):
    # Now, let's evaluate each cluster performance and keep the best
    # distM_fcd = squareform(pdist(LEigs.T, metric='euclidean'))
    for k in Kmeans_results:
        Kmeans_results[k]['scores'] = {}
        for crit in scoringCriterions:
            score = scoringCriterions[crit](LEigs.T, Kmeans_results[k]['labels'])
            Kmeans_results[k]['scores'][crit] = score
            print(f'Performance for {k} clusters with {crit} is {score}')

    return Kmeans_results


# For every subject, calculate the probability of occurrence (or fractional occupancy) of each pattern c
def calculateProbabilitiesOfOccurrence(clustering, LEigs, numClusters):
    print('Calculating Probabilities Of Occurrence')
    P = {}
    labels = {}
    for s in LEigs:
        labels[s] = clustering['clusterer'].predict(LEigs[s].T)
        subjP = [np.sum(labels[s]==k) for k in range(numClusters)]
        P[s] = np.array(subjP, dtype=float)
        P[s] /= np.sum(P[s])  # normalize probabilities (Sum == 1)
    return labels, P


def normalizeTransitionMatrix(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix


# compute the switching matrix (if no subset, use all labels => global)
def calculateSwitchingMatrix(labels, numClusters, subset=None):
    actualLabels = labels if subset is None else subset
    IDX = np.concatenate([labels[s] for s in actualLabels])
    PTransition = np.zeros((numClusters, numClusters))
    for i in range(numClusters):  # compute all transitions FROM i...
        for j in range(numClusters):  # ...to j
            sumatr=0
            for t in range(len(IDX)-1):
                if IDX[t] == i and IDX[t+1] == j:  # only from i to j
                    sumatr = sumatr+1
            # if np.count_nonzero(IDX == i) != 0:  # if there are entries == i
            #     PTransition[i, j] = sumatr/np.count_nonzero(IDX == i)
            PTransition[i, j] = sumatr
    FinalPTransition = normalizeTransitionMatrix(PTransition)
    return FinalPTransition


# Compute the mean lifetimes
def calculateMeanLifetimes(labels, numClusters):
    print('Calculating Mean Lifetimes')
    LT = {}
    for s in labels:
        LT[s] = np.zeros(numClusters)
        for c in range(numClusters):
            Ctime = labels[s]

            # Detect switches in and out of this state
            diffs = np.diff((Ctime == c).astype(int))
            a = np.where(diffs == 1)[0]  # a stores the times when we ENTERED state c
            b = np.where(diffs == -1)[0]  # b sotres the times when we EXITED state c

            # We discard the cases where state starts or ends ON
            if len(b)>len(a):  # more exits than enters: we started in state C
                b = b[1:]  # remove first
            elif len(a) > len(b):  # more enters than exits, we ended on state c
                a = a[:-1]  # remove last element
            elif not a.size==0 and not b.size==0 and a[0] > b[0]:
                b = b[1:]  # remove first
                a = a[:-1]  # remove last element
            if not a.size==0 and not b.size==0:
                C_Durations = b-a
            else:
                C_Durations = 0
            LT[s][c] = np.mean(C_Durations) * BOLDFilters.TR
    return LT


# Analyse the Clustering results
def analyseClustering(clusterings, LEigs):
    print('Start analyseClustering!!!')
    for clusterID in clusterings:
        print(f'   Analysing cluster {clusterID}')
        clustering = clusterings[clusterID]
        numClusters = clustering['centers'].shape[0]

        # compute the labels and probabilities of occurrence of each pattern c
        labels, P = calculateProbabilitiesOfOccurrence(clustering, LEigs, numClusters)

        # Calculate the GLOBAL switching matrix
        # PTransition = calculateSwitchingMatrix(labels, numClusters)

        # Compute the mean lifetimes
        LT = calculateMeanLifetimes(labels, numClusters)

        clusterings[clusterID]['labels'] = labels
        clusterings[clusterID]['Probabilities'] = P
        clusterings[clusterID]['Lifetimes'] = LT
    print('Done analyseClustering!!!')
    return clusterings


# --------------------------------------------------------------------------------------
# Full LEiDA pipeline
# --------------------------------------------------------------------------------------
verbose = True
def processBOLDSignals(BOLDsignals):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # NumSubjects = len(BOLDsignals)
    # N = BOLDsignals[next(iter(BOLDsignals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # 1 - Compute the Leading Eigenvectors from the BOLD datasets
    LEigenvectors = computeEigenvectors(BOLDsignals)

    # 2 - Cluster the Leading Eigenvectors
    LEigs, KMeansResults_Initial = clusterEigenvectors(LEigenvectors)
    KMeansResult_PlusScores = scoreClusters(LEigs, KMeansResults_Initial)

    # 3 - Analyse the Clustering results
    KMeansResults = analyseClustering(KMeansResult_PlusScores, LEigenvectors)

    print('Done processBOLDSignals')
    return KMeansResults


# ============== a practical way to save recomputing necessary (but lengthy) results ==========
# @loadOrCompute
def pipeline(BOLDsignals):
    return processBOLDSignals(BOLDsignals)


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
