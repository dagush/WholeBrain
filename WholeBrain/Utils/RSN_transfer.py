# ======================================================
# Resting State Networks transfer between parcellations
#
# Code by Gustavo Patow
# ======================================================
import csv
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

plotNodes = False


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


def readReferenceRSN(filePath, roundCoords=True):
    res = []
    with open(filePath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if roundCoords:
                res.append((row['ROI Label'], row['ROI Name'], np.array([int(row['R']), int(row['A']), int(row['S'])])))
            else:
                res.append((row['ROI Label'], row['ROI Name'], np.array([float(row['R']), float(row['A']), float(row['S'])])))
    return res


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


def collectNamesAndIDsRSN(rsn, useLR=True):
    names = [(roi[1], int(roi[0])-1) for roi in rsn]  # extract names
    if useLR:
        cleanNames = ['_'.join(n[0].split('_')[1:3]) for n in names]  # and clean them! (left/right separated)
    else:
        cleanNames = [n[0].split('_')[2] for n in names]  # and clean them! (without left/right hemispheres)
    # onlyNames = list(set(cleanNames))
    return cleanNames  #, onlyNames


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


def saveParcellation2CSV(filename, parcellation):
    header = ["ROI Label", "ROI Name", "R", "A", "S"]
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for r in parcellation:
            writer.writerow(r)


def saveRSNIndices(idxs, outFile):
    with open(outFile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
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

    dataRef = readReferenceRSN(inPath+inFileNameRef)
    print(f'we have {len(dataRef)} elements')
    if plotNodes:
        plotParcellation(dataRef, title='Ref')

    dataTarget = readDestinationParcellation(inPath+inFileNameTarget)
    print(f'we have {len(dataTarget)} elements')
    if plotNodes:
        plotParcellation(dataTarget, title='Target')

    labelledTarget = assignRSNLabels(dataRef, dataTarget)
    outPath = '../../Data_Produced/Parcellations/Glasser360RSN.csv'
    saveParcellation = parcellationFormat(labelledTarget)
    saveParcellation2CSV(outPath, saveParcellation)

    names = collectNamesAndIDsRSN(saveParcellation, useLR=False)
    i = indices4RSNs(names)
    outPath = '../../Data_Produced/Parcellations/Glasser360RSN_indices.csv'
    saveRSNIndices(i, outPath)


# ======================================================
# ======================================================
# ======================================================EOF
