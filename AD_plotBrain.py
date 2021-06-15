# =================================================================
#  Plotting brain functions with
#    NiBabel for the gifti files, and
#    matplotlib for plotting
# =================================================================

import numpy as np
import nibabel as nib
from matplotlib import cm
import matplotlib.pyplot as plt
import functions.Utils.plotBrain as plot

base_folder = "./Data_Raw"

# ====================================================
# =============== Load the geometry ==================
glassers_L = nib.load(base_folder + '/Glasser360/' + 'Glasser360.L.mid.32k_fs_LR.surf.gii')
# glassers_L = nib.load(base_folder + '/Glasser360/' + 'Glasser360.L.inflated.32k_fs_LR.surf.gii')
# glassers_L = nib.load(base_folder + '/Glasser360/' + 'Glasser360.L.very_inflated.32k_fs_LR.surf.gii')

glassers_R = nib.load(base_folder + '/Glasser360/' + 'Glasser360.R.mid.32k_fs_LR.surf.gii')
# glassers_R = nib.load(base_folder + '/Glasser360/' + 'Glasser360.R.inflated.32k_fs_LR.surf.gii')
# glassers_R = nib.load(base_folder + '/Glasser360/' + 'Glasser360.R.very_inflated.32k_fs_LR.surf.gii')

flat_L = nib.load(base_folder + '/Glasser360/' + 'Glasser360.L.flat.32k_fs_LR.surf.gii')
flat_R = nib.load(base_folder + '/Glasser360/' + 'Glasser360.R.flat.32k_fs_LR.surf.gii')

# =============== Load the information to display =====
# atlas_l = nib.load(base_folder + '/DecoKringelbach2020/Schaefer1000_L.func.gii')
# atlas_r = nib.load(base_folder + '/DecoKringelbach2020/Schaefer1000_R.func.gii')

# test = nib.load(base_folder + '/Glasser360/' + 'fsaverage.L.DKT_org_Atlas.32k_fs_LR.label.gii')
# test_dlabel = nib.load(base_folder + '/Glasser360/' + 'Glasser32k_fs_LR.dlabel.nii')
# test_dlable2 = np.asanyarray(test_dlabel.dataobj)
mapL = nib.load(base_folder + '/Glasser360/' + 'fsaverage.L.glasser360_fs_LR.func.gii').agg_data()
mapR = nib.load(base_folder + '/Glasser360/' + 'fsaverage.R.glasser360_fs_LR.func.gii').agg_data()
# test2 = nib.load(base_folder + '/Glasser360/' + 'dbs80_left.func.gii').agg_data()

import AD_Auxiliar
subject = '168_S_6142'  # '114_S_6039'
AD_SCnorm, AD_Abeta, AD_tau, AD_fullSeries = AD_Auxiliar.loadSubjectData(subject)

# =============== Plot!!! =============================

cortex = {'model_L': glassers_L, 'model_R':glassers_R,
          'flat_L': flat_L, 'flat_R': flat_R,
          'map_L': mapL, 'map_R': mapR}

data = {'func_L': AD_Abeta, 'func_R': AD_Abeta}
ABetaColors = cm.Blues
plot.multiview5(cortex, data, ABetaColors, suptitle=subject)

data = {'func_L': AD_tau, 'func_R': AD_tau}
tauColors = cm.YlOrBr
plot.multiview5(cortex, data, tauColors, suptitle=subject)

# ====================================================
# ====================================================
# ====================================================EOF
