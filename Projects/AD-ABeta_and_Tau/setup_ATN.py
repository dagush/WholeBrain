# --------------------------------------------------------------------------------------
# Setup file for processing subjects with the AT(N) classification (but we do not
# have Neurodegeneration information)
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import dataLoader

# ==========================================================================
# Important config options: filenames
# ==========================================================================
base_folder = "../../Data_Raw/from_Ritter"
save_folder = "../../Data_Produced/AD_ABeta_and_Tau"

# --------------------------------------------------
# Classify subject information into AT(N) sets...
# The file subjectsATN.csv is created with
# classific_AT(N).py, so run that one first...
# --------------------------------------------------
subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
classification = dataLoader.checkClassifications(subjects, "/subjectsATN.csv")

dataSetLabels = ['A+T+', 'A-T+', 'A+T-', 'A-T-']
