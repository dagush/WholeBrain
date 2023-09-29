# =================================================================
# =================================================================
# Utility WholeBrain to compute multi-views of the cortex data
# =================================================================
# =================================================================
import matplotlib as plt
from Utils.plot3DBrain import *


# =================================================================
# plots the 6-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
#           L-flat,         R-flat
# =================================================================
def multiview6(cortex, data, numRegions, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(3, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 3)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(3, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Rh-lateral', cmap=rightCmap, **kwds)
    ax = plt.subplot(3, 2, 4)
    plotColorView(ax, cortex, data, numRegions, 'Rh-medial', cmap=rightCmap, **kwds)

    # ================== flatmaps
    ax = fig.add_subplot(3, 2, 5)  # left hemisphere flat
    plotColorView(ax, cortex, data, numRegions, 'L-flat', cmap=leftCmap, **kwds)
    ax = fig.add_subplot(3, 2, 6)  # right hemisphere flat
    plotColorView(ax, cortex, data, numRegions, 'R-flat', cmap=rightCmap, **kwds)

    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots a 5-view plot:
#           lh-lateral,               rh-lateral,
#                       l/r-superior,
#           lh-medial,                rh-medial
# =================================================================
def multiview5(cortex, data, numRegions, cmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    plotColorView(axs[0,0], cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds)
    plotColorView(axs[1,0], cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds)
    plotColorView(axs[0,2], cortex, data, numRegions, 'Rh-lateral', cmap=cmap, **kwds)
    plotColorView(axs[1,2], cortex, data, numRegions, 'Rh-medial', cmap=cmap, **kwds)
    # === L/R-superior
    gs = axs[0, 1].get_gridspec()
    # remove the underlying axes
    for ax in axs[:,1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:,1])
    plotColorView(axbig, cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(axbig, cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    # ============= Adjust the sizes
    plt.subplots_adjust(left=0.0, right=0.8, bottom=0.0, top=1.0, wspace=0, hspace=0)
    # ============= now, let's add a colorbar...
    if 'norm' not in kwds:
        vmin = np.min(data['func_L']) if 'vmin' not in kwds else kwds['vmin']
        vmax = np.max(data['func_L']) if 'vmax' not in kwds else kwds['vmax']
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = kwds['norm']
    PCM = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # This parameter is the dimensions [left, bottom, width, height] of the new axes.
    fig.colorbar(PCM, cax=cbar_ax)
    # ============ and show!!!
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots the 4-plot
#           Lh-lateral,     Rh-lateral,
#           Lh-medial,      Rh-medial,
# =================================================================
def multiview4(cortex, data, numRegions, leftCmap=plt.cm.coolwarm, rightCmap=plt.cm.coolwarm,
               suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(2, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 3)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=leftCmap, **kwds)
    ax = plt.subplot(2, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Rh-medial', cmap=rightCmap, **kwds)
    ax = plt.subplot(2, 2, 4)
    plotColorView(ax, cortex, data, numRegions, 'Rh-lateral', cmap=rightCmap, **kwds)

    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots a left/Right-view plot:
#                       l/r-superior,
# =================================================================
def leftRightView(cortex, data, numRegions, cmap=plt.cm.coolwarm,
                  suptitle='', figsize=(15, 10), display=True, savePath=None, **kwds):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    plotColorView(ax, cortex, data, numRegions, 'Lh-medial', cmap=cmap, **kwds)
    ax = plt.subplot(1, 2, 2)
    plotColorView(ax, cortex, data, numRegions, 'Lh-lateral', cmap=cmap, **kwds)
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()


# =================================================================
# plots a top-view plot:
#                       l/r-superior,
# =================================================================
def topViewAxs(ax, cortex, data, numRegions, cmap=plt.cm.coolwarm,
               suptitle='', **kwds):
    plotColorView(ax, cortex, data, numRegions, 'L-superior', suptitle=suptitle, cmap=cmap, **kwds)
    plotColorView(ax, cortex, data, numRegions, 'R-superior', suptitle=suptitle, cmap=cmap, **kwds)
    if suptitle == '':
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)


def topView(cortex, data, numRegions,
            figsize=(15, 10), display=True, savePath=None, **kwd):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    topViewAxs(ax, cortex, data, numRegions, **kwd)
    if savePath is not None:
        plt.savefig(savePath)
        plt.close()
    if display:
        plt.show()

# =================================================================
# plots top views for multiple vectors... Now, only the top view.
# All plots have the same limits and use the same colorbar
# =================================================================
def plot_TopViewValuesForAllCohorts(burdens, cmap):
    fig, axs = plt.subplots(1, len(burdens)+1,
                            # we add an extra row to solve a strange bug I found, where the last panel dpes not show the ticks, and do not have the patience to fix
                            gridspec_kw={'wspace': 0.2, 'hspace': 0.2})

    vmin = np.min([np.min(burdens[c]) for c in burdens])
    vmax = np.max([np.max(burdens[c]) for c in burdens])
    crtx = setUpGlasser360()
    for c, cohort in enumerate(burdens):
        vect = burdens[cohort]
        data = {'func_L': vect, 'func_R': vect}
        topViewAxs(axs[c], crtx, data, 360, vmin=vmin, vmax=vmax,
                   cmap=cmap, suptitle=cohort, fontSize=15)
    norm = Normalize(vmin=vmin, vmax=vmax)
    PCM = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax = fig.add_axes([0.77, 0.3, 0.02, 0.4])  # This parameter is the dimensions [left, bottom, width, height] of the new axes.
    fig.colorbar(PCM, cax=cbar_ax)
    fig.tight_layout()
    plt.show()


# =================================================================
# functions to plot multiple mutiview5 plots for different sets,
# but all wiuth a common normalization and a common colorbar
# =================================================================
# This one plots a single multiview5 plot
def plot_multiview5Values(obs, title, fileName, display, cmap, modality, norm):
    crtx = setUpGlasser360()
    # =============== Plot!!! =============================
    data = {'func_L': obs, 'func_R': obs}
    resultsPath = f'../../Results/AD-Meta/Plot {modality}/'
    multiview5(crtx, data, 360, cmap, suptitle=title, lightingBias=0.1, mode='flatWire', shadowed=True,
               display=display, savePath=resultsPath+fileName+'.png', norm=norm)


# plots multiple multiview5 plots
def plot_multiview5ValuesForEachChort(burdens, title, igniMetaName, display, cmap, modality):
    vmin = np.min([np.min(burdens[c]) for c in burdens])
    vmax = np.max([np.max(burdens[c]) for c in burdens])
    norm = Normalize(vmin=vmin, vmax=vmax)
    for c in burdens:
        fullFileName = c + igniMetaName
        plot_multiview5Values(burdens[c], title, fullFileName, display, cmap, modality, norm)



# ===========================
#  Convenience function for the Glasser parcellation, for debug purposes only...
# ===========================
def setUpGlasser360():
    Glasser360_baseFolder = "../../Data_Raw/Parcellations"
    # =============== Load the geometry ==================
    glassers_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.mid.32k_fs_LR.surf.gii')
    # glassers_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.inflated.32k_fs_LR.surf.gii')
    # glassers_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.very_inflated.32k_fs_LR.surf.gii')

    glassers_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.mid.32k_fs_LR.surf.gii')
    # glassers_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.inflated.32k_fs_LR.surf.gii')
    # glassers_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.very_inflated.32k_fs_LR.surf.gii')

    flat_L = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.L.flat.32k_fs_LR.surf.gii')
    flat_R = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'Glasser360.R.flat.32k_fs_LR.surf.gii')
    mapL = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'fsaverage.L.glasser360_fs_LR.func.gii').agg_data()
    mapR = nib.load(Glasser360_baseFolder + '/Glasser360/' + 'fsaverage.R.glasser360_fs_LR.func.gii').agg_data()

    cortex = {'model_L': glassers_L, 'model_R': glassers_R,
              'flat_L': flat_L, 'flat_R': flat_R,
              'map_L': mapL, 'map_R': mapR}
    return cortex


# =================================================================
# ================================= module test code
if __name__ == '__main__':
    from matplotlib import cm

    crtx = setUpGlasser360()

    # =============== Plot!!! =============================
    testData = np.arange(0, 360)
    data = {'func_L': testData, 'func_R': testData}
    # testColors = cm.cividis
    testColors = cm.YlOrBr

    multiview5(crtx, data, 360, testColors, lightingBias=0.1, mode='flatWire', shadowed=True)

