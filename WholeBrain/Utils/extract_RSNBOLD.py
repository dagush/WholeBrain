# ========================================================================
# Extract RSN BOLD signals
#
#  Code by Gustavo Patow
# ========================================================================
import numpy as np

def collectNamesAndIDsRSN(rsn, useLR=True):
    names = [(r[1], int(r[0])-1) for r in rsn]  # extract names
    if useLR:
        cleanNames = [('_'.join(n[0].split('_')[0:3]), n[1]) for n in names]  # and clean them! (left/right separated)
    else:
        cleanNames = [(n[0].split('_')[2], n[1]) for n in names]  # and clean them! (without left/right hemispheres)
    return cleanNames


def extractSubjectRSNfMRI(BOLD, namesAndIDs):
    allNames = list(set([n[0] for n in namesAndIDs]))  # no repeated entries
    res = {}
    for name in allNames:
        print(f'for {name} we have {len([rsn[1] for rsn in namesAndIDs if name in rsn[0]])} regions')
        res[name] = np.array([BOLD[rsn[1]] for rsn in namesAndIDs if name in rsn[0]])
    print(f'We have {sum([res[reg].shape[0] for reg in res])} regions in total.')
    return res


def extractRSNGroupfMRI(BOLDs, rsn, useLR=True):
    namesAndIDs = collectNamesAndIDsRSN(rsn, useLR=useLR)
    res = {}
    for s in BOLDs:
        res[s] = extractSubjectRSNfMRI(BOLDs[s], namesAndIDs)
    return res



# ==================================================================
# test code
# ==================================================================
if __name__ == '__main__':
    from RSN_transfer import readReferenceRSN, plotParcellation
    inPath = '../../Data_Produced/Glasser360RSN.csv'
    rsn = readReferenceRSN(inPath, roundCoords=False)
    print(f'len of RSN Glasser is {len(rsn)} (should be 360)')
    plotParcellation(rsn, 'Glasser360')
    print('Done')


# ======================================================
# ======================================================
# ======================================================EOF