# --------------------------------------------------------------------------------------
# Plot the results for processing AD, MCI and HC subjects
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# By Gustavo Patow
#
# Pre-requisites: Before executing this, be sure to have correctly
#   * configured the setup_AD.py file
#   * run Prepro.py
#
# --------------------------------------------------------------------------------------
import numpy as np

import WholeBrain.Utils.plotFitting as plotFitting
import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.swFCD as swFCD
import WholeBrain.Observables.phFCD as phFCD

from setup import *


print("\n\n###################################################################")
print("# Plot !!!")
print("###################################################################\n")
subjectName = 'AvgHC'
distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}
outFilePath = save_folder + '/' + subjectName + '-temp'
exageratedRange = np.arange(0.0, 10.001, 0.001)  # loadAndPlot will load and show only the existing files...

plotFitting.loadAndPlot(outFilePath +'/fitting_we{}.mat', distanceSettings,
                        WEs=exageratedRange, weName='we',
                        empFilePath=outFilePath +'/fNeuro_emp.mat')

#####################################################################################################
# Results (in (0.0, 5.5):
# - Optimal FC = 0.28824680025389854 @ 3.25
# - Optimal swFCD = 0.10123722072740582 @ 3.4
# - Optimal phFCD = 0.03969905148884678 @ 3.1
#####################################################################################################

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF