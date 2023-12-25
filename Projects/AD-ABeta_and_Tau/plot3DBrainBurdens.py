# =================================================================
#  Plotting brain WholeBrain with
#    NiBabel for the gifti files, and
#    matplotlib for plotting
# =================================================================
import nibabel as nib
import numpy as np
from matplotlib import cm

import Utils.Plotting.plot3DBrain_Utils as plot

from setup import *

Glasser360_baseFolder = "../../Data_Raw/Parcellations"


# ====================================================
# =============== Load the geometry ==================
def loadGeo():
    glassers_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.mid.32k_fs_LR.surf.gii')
    # glassers_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.inflated.32k_fs_LR.surf.gii')
    # glassers_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.very_inflated.32k_fs_LR.surf.gii')

    glassers_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.mid.32k_fs_LR.surf.gii')
    # glassers_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.inflated.32k_fs_LR.surf.gii')
    # glassers_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.very_inflated.32k_fs_LR.surf.gii')

    flat_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.flat.32k_fs_LR.surf.gii')
    flat_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.flat.32k_fs_LR.surf.gii')

    # =============== Load the information to display =====
    # atlas_l = nib.load(base_folder + '/DecoKringelbach2020/Schaefer1000_L.func.gii')
    # atlas_r = nib.load(base_folder + '/DecoKringelbach2020/Schaefer1000_R.func.gii')

    # test = nib.load(base_folder + '/Glasser360/' + 'fsaverage.L.DKT_org_Atlas.32k_fs_LR.label.gii')
    # test_dlabel = nib.load(base_folder + '/Glasser360/' + 'Glasser32k_fs_LR.dlabel.nii')
    # test_dlable2 = np.asanyarray(test_dlabel.dataobj)
    mapL = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'fsaverage.L.glasser360_fs_LR.func.gii').agg_data()
    mapR = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'fsaverage.R.glasser360_fs_LR.func.gii').agg_data()
    # test2 = nib.load(base_folder + '/Glasser360/' + 'dbs80_left.func.gii').agg_data()

    cortex = {'model_L': glassers_L, 'model_R': glassers_R,
              'flat_L': flat_L, 'flat_R': flat_R,
              'map_L': mapL, 'map_R': mapR}
    return cortex


# ====================================================
# =============== Load burdens =======================
def loadCohortBurden(cohort):
    subjects = [s for s in classification if classification[s] == cohort]
    ABetaBurden = {}
    tauBurden = {}
    for s in subjects:
        AD_SCnorm, AD_Abeta, AD_tau, AD_fullSeries = dataLoader.loadSubjectData(s, normalizeBurden=False)
        ABetaBurden[s] = AD_Abeta
        tauBurden[s] = AD_tau
    return ABetaBurden, tauBurden


# ====================================================
# =============== Compute burden avg =================
def computeAvgBurden(burden, numValues):
    res = np.zeros(numValues)
    for s in burden:
        res += burden[s][:numValues]
    return res/len(burden)


# ===============================================
# Load ABeta and tau burdens
# and Plot them!!!
# ===============================================
N = 360
cort = loadGeo()
# subject = '036_S_4430'  # AD, but with the highest pvalue between ABeta and tau...
# '011_S_4827'  # AD
# '114_S_6039'  # AD
# '168_S_6142'  # AD
avgABetas = {}
avgTaus = {}
for cohort in dataSetLabels:
    ABetas, taus = loadCohortBurden(cohort)
    avgABetas[cohort] = computeAvgBurden(ABetas, N)
    avgTaus[cohort] = computeAvgBurden(taus, N)
    print(f'min ABeta {cohort} = {np.min(avgABetas[cohort])}')
    print(f'min ABeta {cohort} = {np.min(avgABetas[cohort])}')
    print(f'max tau {cohort} = {np.max(avgTaus[cohort])}')
    print(f'max tau {cohort} = {np.max(avgTaus[cohort])}')
    print()

ABetaVmin = np.min([np.min(avgABetas[cohort]) for cohort in avgABetas])
ABetaVmax = np.max([np.max(avgABetas[cohort]) for cohort in avgABetas])
tauVmin = np.min([np.min(avgTaus[cohort]) for cohort in avgTaus])
tauVmax = np.max([np.max(avgTaus[cohort]) for cohort in avgTaus])

for cohort in dataSetLabels:
    # =============== Plot!!! =============================
    data = {'func_L': avgABetas[cohort], 'func_R': avgABetas[cohort]}
    ABetaColors = cm.Blues
    plot.multiview5(cort, data, N, ABetaColors, suptitle=rf'A$\beta$ ({cohort})', mode='flatWire', vmin=ABetaVmin, vmax=ABetaVmax)

for cohort in dataSetLabels:
    data = {'func_L': avgTaus[cohort], 'func_R': avgTaus[cohort]}
    tauColors = cm.YlOrBr
    plot.multiview5(cort, data, N, tauColors, suptitle=f'tau ({cohort})', mode='flatWire', vmin=tauVmin, vmax=tauVmax)

# ====================================================
# ====================================================
# ====================================================EOF
