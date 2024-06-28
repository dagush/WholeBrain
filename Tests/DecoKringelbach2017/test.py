# --------------------------------------------------------------------------------------
# Test for Intrinsic Ignition computation
#
# From:
# Hierarchy of Information Processing in the Brain: A Novel ‘Intrinsic Ignition’ Framework,
# Gustavo Deco and Morten L. Kringelbach, Neuron, Volume 94, Issue 5, 961 - 968
# Doi: https://doi.org/10.1016/j.neuron.2017.03.028
#
# Adapted to python by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
import hdf5storage as sio
import matplotlib.pyplot as plt


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    # Simple verification test, to check the info from the paper...

    plt.rcParams.update({'font.size': 15})

    # Load connectome:
    # --------------------------------
    inFilePath = '../../Data_Raw'
    # outFilePath = '../Data_Produced'
    CFile = sio.loadmat(inFilePath + '/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    # fileName = outFilePath + '/Human_66/Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'

    # ... TODO: finish ignition/metastability!!!
