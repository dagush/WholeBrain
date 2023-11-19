# ==================================================================================
# Classify subjects in the AT(N) (ABeta, tau, neurodegeneration) classification,
# although we don't have any neurodegeneration information, so only AT for us...
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# ==================================================================================
import numpy as np
import csv

# Reuse setup_AD.py so save some work and redundant definitions...
from setup import *


# ====================================================
# =============== Load burdens =======================
def loadSubjectBurden():
    ABetaBurden = {}
    tauBurden = {}
    for s in subjects:
        AD_SCnorm, AD_Abeta, AD_tau, AD_fullSeries = dataLoader.loadSubjectData(s, normalizeBurden=False)
        ABetaBurden[s] = AD_Abeta
        tauBurden[s] = AD_tau
    return ABetaBurden, tauBurden


# ====================================================
# =============== save classification ================
def saveATNClassifications(Classif):
    fileName = base_folder + "/subjectsATN.csv"
    # Define the fields/columns for the CSV file
    fields = ["ID", "Classific"]

    # Open the CSV file with write permission
    with open(fileName, "w", newline="") as csvfile:
        # Create a CSV writer using the field/column names
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # Write the header row (column names)
        writer.writeheader()

        # Write the data
        for s in Classif:
            writer.writerow({'ID': s, 'Classific': Classif[s]})
    print(f'AT(N) classific savet to {fileName}')

# # ====================================================
# # =============== Compute burden avgs ================
def computeAvgSubjectBurden(burden):
    res = {}
    for s in burden:
        res[s] = np.average(burden[s])
    return res


def analyzeBurden(burden, name):
    print(f'{name}: min={np.min([burden[s] for s in burden])}')
    print(f'{name}: max={np.max([burden[s] for s in burden])}')
    print(f'{name}: avg={np.average([burden[s] for s in burden])}')
    print(f'{name}: var={np.var([burden[s] for s in burden])}')
    print()


def classifySubjects(ABeta, tau):
    counters = {'A+T+': 0, 'A-T+': 0, 'A+T-': 0, 'A-T-': 0}
    ABetaThreshold = np.average([ABeta[s] for s in ABeta]) * 0.9  # 1.422
    tauThreshold = np.average([tau[s] for s in tau]) * 0.9  # 1.679
    claasificationsATN = {}
    for s in ABeta:
        classif = f'A{"+" if ABeta[s]>ABetaThreshold else "-"}T{"+" if tau[s]>tauThreshold else "-"}'
        claasificationsATN[s] = classif
        print(f'{s}: {classif}')
        counters[classif] += 1
    print(f'Thresholds: ABeta={ABetaThreshold} tau={tauThreshold}')
    return counters, claasificationsATN


# ===============================================
# Load ABeta and tau burdens
# and do some computations
# ===============================================
ABetas, taus = loadSubjectBurden()
avgABetas = computeAvgSubjectBurden(ABetas)
avgTaus = computeAvgSubjectBurden(taus)

analyzeBurden(avgABetas, 'ABeta')
analyzeBurden(avgTaus, 'tau')

counters, classificationATN = classifySubjects(avgABetas, avgTaus)
print(f'Counters: {counters}')
print(f'Check: {np.sum([counters[c] for c in counters])}')

saveATNClassifications(classificationATN)

print()

