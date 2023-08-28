# ========================================================================
# Extract BOLD signals corresponding to a given RSN partition
#
#  Code by Gustavo Patow
# ========================================================================
import numpy as np
from RSN_transfer import collectNamesAndIDsRSN


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


# ======================================================
# ======================================================
# ======================================================EOF