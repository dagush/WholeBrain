# =================================================================
#  Plotting brain WholeBrain with
#    NiBabel for the gifti files, and
#    matplotlib for plotting
#
# Based on the code from the paper:
# [DecoKringelbach2020] Turbulent-like Dynamics in the Human Brain, Deco & Kringelbach,
#                       2020, Cell Reports 33, 108471
#                       https://doi.org/10.1016/j.celrep.2020.108471
#
# To Do: increase the size of the plots
# =================================================================
#
# These files can be usually computed with the Connectome Workbench
# Workbench Command Quick reference (https://www.humanconnectome.org/software/workbench-command/-volume-to-surface-mapping)
# wb_command -volume-to-surface-mapping
#       <volume> - the volume to map data from
#       <surface> - the surface to map the data onto
#       <metric-out> - output - the output metric file
#       [-trilinear] - use trilinear volume interpolation
#
# Example:
# wb_command -volume-to-surface-mapping different_regions_CNT-UWS.nii.gz
#                                       Q1-Q6_R440.R.inflated.32k_fs_LR.surf.gii
#                                       CNT-UWSR.shape.gii
#                                       -trilinear

import numpy as np
import nibabel as nib
from matplotlib import cm
import matplotlib.pyplot as plt
import WholeBrain.Utils.plot3DBrain as plot

base_folder = "../Data_Raw"


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
atlas_l = nib.load(base_folder + '/DecoKringelbach2020/Schaefer1000_L.func.gii').agg_data()
atlas_r = nib.load(base_folder + '/DecoKringelbach2020/Schaefer1000_R.func.gii').agg_data()


# =============== Plot!!! =============================

leftColors = cm.tab20c
rightColors = cm.brg
cortex = {'model_L': glassers_L, 'model_R':glassers_R,
          'flat_L': flat_L, 'flat_R': flat_R}
data = {'func_L': atlas_l, 'func_R': atlas_r}
plot.multiview6(cortex, data, leftColors, rightColors, shaded=False)  # shadowed=True)

# ====================================================
# ====================================================
# ====================================================EOF
