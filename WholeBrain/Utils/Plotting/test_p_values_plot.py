# ==========================================================================
# Simple demo of the p_values.plotComparisonAcrossLabels2 function
# Observe that the distance between DataB and DataC is small. If you make it
# smaller (e.g., 3.5) you may get a non significative result. With 3.6, most
# of the results will be p<0.05
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt

import WholeBrain.Utils.p_values as p_values

base_folder = 'path to your data'

def loadResultsCohort(base_folder, cohort):
    means = {'DataA':1, 'DataB':3, 'DataC':3.6}
    # here load the results of the cohort abd burden, and RETURN them as an array of floats
    result = means[cohort] + 2 * np.random.rand(10)
    return result


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})
    # --------------------------------------------------
    # Some more setups...
    # --------------------------------------------------
    dataSetLabels = ['DataA', 'DataB', 'DataC']  # the set of labels over we will iterate...

    # --------------------------------------------------
    # Load results simulation
    # --------------------------------------------------
    resI = {}
    for cohort in dataSetLabels:
        cohort_results = loadResultsCohort(base_folder, cohort=cohort)
        resI[cohort] = cohort_results

    # --------------------------------------------------
    # Plot p_value comparison!
    # --------------------------------------------------
    p_values.plotComparisonAcrossLabels2(resI, columnLables=dataSetLabels, graphLabel=f'Results from Simulation')

    print('done!')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF