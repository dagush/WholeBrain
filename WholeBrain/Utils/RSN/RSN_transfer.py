# ======================================================
# Resting State Networks transfer between parcellations
#
# Code by Gustavo Patow
# ======================================================
import csv
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


def plotParcellation(parcellationData, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for region in parcellationData:
        r = region[2]
        ax.scatter(r[0], r[1], r[2], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    plt.show()


def readDestinationParcellation(filePath):
    res = []
    with open(filePath) as f:
        lines = f.readlines()
        for line in lines:
            res.append(('', '', np.fromstring(line, dtype=float, sep=' ')))
    return res


# Taken from
#   https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
#   The approach is generally to first use the point data to build up a k-d tree. The computational complexity
# of that is on the order of N log N, where N is the number of data points. Range queries and nearest neighbour
# searches can then be done with log N complexity. This is much more efficient than simply cycling through all
# points (complexity N).
#   Thus, if you have repeated range or nearest neighbor queries, a k-d tree is highly recommended.
def findClosestPoints(reference, target):
    tree = spatial.cKDTree(reference)
    dist, indexes = tree.query(target)
    return indexes


def assignRSNLabels(referenceSet, targetSet):
    # Shoe-horn existing data for entry into KDTree routines
    referenceArray = np.array([r[2] for r in referenceSet])
    targetArray = [p[2] for p in targetSet]
    res = findClosestPoints(referenceArray, targetArray)

    targetSetLabelled = [('', referenceSet[res[pos]][1], r[2]) for pos, r in enumerate(targetSet)]

    return targetSetLabelled


# Simple function to extract the RSN name from a description string in Yeo's format. For instance, from
# '7Networks_LH_Default_Temp_8' we extract 'Default'. If an detailedRSNs llist is added, these will be
# added to the output. If no detailes wanted, just pass []
def extractRSNName(name, useLR, detailedRSNs):
    rsnName = name.split('_')[2]
    if rsnName in detailedRSNs:
        subregionsTest = [sub in name.split('_')[3] for sub in detailedRSNs[rsnName]]  # check whether the subarea name is in the list
        if any(subregionsTest):
            rsnName += '_' + detailedRSNs[rsnName][subregionsTest.index(True)]
        else:
            if len(detailedRSNs[rsnName]) > 0:  # if we were given a list, but this particular area is missing...
                rsnName += '_OTHER'
            # If we weren't given a llist, nothing to do!
    if useLR:
        rsnName += '_' + name.split('_')[1]  # and clean them! (left/right separated)
    return rsnName


# detailedRSNs is a dictionary of {'RoI': subareas}, where
# subareas is a list of all subarea names to be considered. If empty, the default RoI value
# will be used. All nodes not in any of these subareas will be added to the 'OTHER' default area.
def collectNamesRSN(rsn, useLR=True, detailedRSNs={}):
    names = [(roi[1], int(roi[0])-1) for roi in rsn]  # extract names
    cleanNames = [extractRSNName(n[0], useLR, detailedRSNs) for n in names]
    return cleanNames


def indices4RSNs(parcellation):
    names = list(set(parcellation))
    res = {}
    for rsn in names:
        idx = [pos for pos,roi in enumerate(parcellation) if roi == rsn]
        res[rsn] = idx
    return res


def parcellationFormat(labelledTarget):
    parc = [[pos+1, roi[1], roi[2][0], roi[2][1], roi[2][2]] for pos,roi in enumerate(labelledTarget)]
    return parc


# ================================================================
# Load and save parcellation data
# ================================================================
def saveParcellation2CSV(filename, parcellation):
    header = ["ROI Label", "ROI Name", "R", "A", "S"]
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for r in parcellation:
            writer.writerow(r)


def readReferenceRSN(filePath, roundCoords=True):
    res = []
    with open(filePath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if roundCoords:
                res.append((row['ROI Label'], row['ROI Name'], np.array([int(float(row['R'])), int(float(row['A'])), int(float(row['S']))])))
            else:
                res.append((row['ROI Label'], row['ROI Name'], np.array([float(row['R']), float(row['A']), float(row['S'])])))
    return res


def saveRSNIndices(idxs, outFile):
    header = ["RSN Label", "Indices"]
    with open(outFile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rsn in idxs:
            writer.writerow([rsn, idxs[rsn]])


# ==================================================================
# test code: transfer RSNs to the Glasser360 parcellation
# ==================================================================
if __name__ == '__main__':
    numNodes = 1000
    # -------- As input, we are going to use Yeo's 1000 roi RSN info on Schaefer's 2018 parcellation
    inPath = '../../Data_Raw/Parcellations/'
    inFileNameRef = f'Schaefer2018-RSN_Centroid_coordinates/Schaefer2018_{numNodes}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
    inFileNameTarget = 'Glasser360/glasser_coords.txt'


    detailNetworks = {}  # If a mode detailed region is NOT needed, use an empty detailNetworks
    # detailNetworks = {'Default': ['PFC', 'Par', 'Temp', 'pCunPCC', 'PHC']}  # If a more detailed region is needed, especify it here (see comment for collectNamesAndIDsRSN)

    plotNodes = False

    # Read all reference values, i.e., from Yeo's Schaefer2018 RSN labels
    dataRef = readReferenceRSN(inPath+inFileNameRef)
    print(f'we have {len(dataRef)} elements')
    if plotNodes:
        plotParcellation(dataRef, title='Ref')

    # read all coords for the target parcellation (here, Glasser360)
    dataTarget = readDestinationParcellation(inPath+inFileNameTarget)
    print(f'we have {len(dataTarget)} elements')
    if plotNodes:
        plotParcellation(dataTarget, title='Target')

    # First, transfer the closest RSN label to each node of the target parcellation
    labelledTarget = assignRSNLabels(dataRef, dataTarget)
    # this keeps all the originalnames, no distinctions whether we use detailed regions or not...
    outPath = '../../Data_Produced/Parcellations/Glasser360RSN.csv'
    saveParcellation = parcellationFormat(labelledTarget)
    saveParcellation2CSV(outPath, saveParcellation)
    print(f'saved parcellation to: {outPath}')

    # This groups RSN labels into fewer sets, grouping by RSN name and, if needed, subregion name
    useLR = False
    names = collectNamesRSN(saveParcellation, useLR=useLR, detailedRSNs=detailNetworks)
    print(f'Names collected: {list(set(names))}')
    i = indices4RSNs(names)
    # The outfile should change according to whether we use or not detailed regions...
    outPath = f'../../Data_Produced/Parcellations/Glasser360RSN_{"14" if useLR else "7"}_indices.csv'  # if we do NOT use detailed regions
    # outPath = f'../../Data_Produced/Parcellations/Glasser360RSN_{"-".join(detailNetworks.keys())}_indices.csv'  # if we use detailed regions...
    saveRSNIndices(i, outPath)
    print(f'saved indices to: {outPath}')


# ======================================================
# ======================================================
# ======================================================EOF
