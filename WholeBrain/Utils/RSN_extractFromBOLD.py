# -------------------------------------------------------------------------
#  Extract RSN signals from a whole-brain fMRI signal
#  by Gustavo Patow
# -------------------------------------------------------------------------
import numpy as np
import csv
import Utils.RSN_transfer as RSN_transfer


def readIndicesFile(filePath):
    res = {}
    with open(filePath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            res |= {row['RSN Label']: row['Indices']}
    return res


def simplifyIndicesFile(rsnIndices, subset):
    return {k: rsnIndices[k] for k in subset}


def extractSubjectRSNfMRI(BOLD, namesAndIDs):
    allNames = list(set(namesAndIDs))  # no repeated entries
    res = {}
    for name in allNames:
        ids = eval(namesAndIDs[name])
        print(f'for {name} we have {len(ids)} regions')
        res[name] = BOLD[ids]
    print(f'We have {sum([res[reg].shape[0] for reg in res])} regions in total.')
    return res


def extractRSNGroupfMRI(BOLDs, rsn):
    res = {}
    for s in BOLDs:
        res[s] = extractSubjectRSNfMRI(BOLDs[s], rsn)
    return res

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------EOF