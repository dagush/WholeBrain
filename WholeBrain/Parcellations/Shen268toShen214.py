# ------------------------------------------------------------
#
# Convert data from Shen268 to Shen214...
#
# The collected data is in Shen268 format. However, the fMRI data we are given comes in a reduced Shen214 format. For
# the conversion, we have a label converter (called shen214to268_idx.mat) from 214 to 268 (standard Shen atlas).
# Important (from the original mails):
# 1) Regions 223 and 224 in the shen268 were merged, and correspond now to region 196 in the shen214;
# 2) With exception of 1), all the missing regions in the vector attached (i.e. not in 1:1:268), should correspond
#    to the cerebellum, and thus we are not interested in them...
# Note: The converter file uses Matlab notation, so all indexes are 1 more than what they should!!!
#
# Note: this code is in earlier stage, and would need some generalization... ;-)
# ------------------------------------------------------------
import numpy as np
import scipy.io as sio
import csv

Add = 0
Replace = 1
Leave = 2

def convertShen268ToShen214(data, path, addOrReplace=Add):
    convertIdxs = sio.loadmat(path + "shen214to268_idx.mat")['shen214to268_idx'].flatten()
    convertIdxs = convertIdxs.astype(int)
    # a quick verification check...
    print(f'index for 196: {convertIdxs[196-1]}')
    # some stupid checks...
    print(f'data@223: {data[223-1]}')
    print(f'data@224: {data[224-1]}')
    # Now, let's start with the recepee for Shen268 to Shen214 conversion. First, lets accumulate regions 223 and 224 (in
    # Matlab indexing, 1 less in Python)
    if addOrReplace == Add:
        data[223-1] += data[224-1]
    elif addOrReplace == Leave:
        pass  # leave data[223-1] unchanged...
    else:  # replace...
        data[223-1] = data[224-1]
    print(f'final value for data[223-1] = {data[223-1]}')
    # Now, let's keep only the indexed cells. Careful: Matlab indexing, 1 less in Python...
    converted = data[convertIdxs-1].flatten()
    print(f'shape: {converted.shape}')
    # Just a quick test:
    print(f'Last minute test: @ 196 we have {converted[196-1]}')
    return converted


def loadData(filePath):
    res = []
    with open(filePath) as f:
        lines = f.readlines()
        for line in lines:
            l = np.fromstring(line, dtype=int, sep=',')
            if len(l) != 0:
                res.append(l[1])
    return np.array(res)

# ======================================================================
# ======================================================================
# ======================================================================
if __name__ == '__main__':
    baseFolder = '../../Data_Raw/Parcellations/'
    networkLabels = 'shen_268_parcellation_networklabels.csv'
    data = loadData(baseFolder+networkLabels)

    # Do the conversion!!!
    converted = convertShen268ToShen214(data, baseFolder, addOrReplace=Replace)

    # Now, let's save the result
    fileName = '../../Data_Produced/Parcellations/shen_214_parcellation_networklabels.mat'
    sio.savemat(fileName, {'densities': converted})

    print('Done!')


# ======================================================================
# ======================================================================
# ======================================================================EOF
