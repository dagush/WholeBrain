# =======================================================================
# INSIDEOUT framework, from:
# Deco, G., Sanz Perl, Y., Bocaccio, H. et al. The INSIDEOUT framework
# provides precise signatures of the balance of intrinsic and extrinsic
# dynamics in brain states. Commun Biol 5, 572 (2022).
# https://doi.org/10.1038/s42003-022-03505-7
#
# By Gustavo Deco,
# Translated by Marc Gregoris and Gustavo Patow
# =======================================================================
import numpy as np
from WholeBrain.Observables import Insideout

# =============== Load HCP data
import WholeBrain.Utils.DataLoaders.HCP_dbs80 as HCP
DL = HCP.HCP()

import WholeBrain.Observables.BOLDFilters as BOLDFilters
BOLDFilters.TR = 3
BOLDFilters.flp = 0.01
BOLDFilters.fhi = 0.09
import WholeBrain.Utils.p_values as p_values

NLAG = 6  # Number of taus (lag values) to compute



# ================================================================================================================
def processSubjects():
    # diccionari per guardar fowrev de cada subjecte
    Fr = {}
    subject_list = []
    index2 = []

    for subject in DL.get_allStudySubjects():
        subjData = DL.get_SubjectData(subject, correcSCMatrix=True, normalizeBurden=True)
        FowRev, AsymRev, AsymFow = Insideout.from_fMRI(subjData[subject]['timeseries'], applyFilters=False, removeStrongArtefacts=False)
        Fr[subject] = FowRev

    Tauwinner = Insideout.calculate_Tauwinner(DL, Fr)

    plotSets = {}
    for group in DL.get_groupLabels():
        plotSets[group] = []
        subjects = DL.get_groupSubjects(group)
        for subject in subjects:
            plotSets[group].append(Fr[subject][Tauwinner])
    return plotSets


if __name__ == '__main__':
    kk = DL.get_classification()
    plotSets = processSubjects()
    p_values.plotComparisonAcrossLabels2(plotSets)
    print("done")
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF