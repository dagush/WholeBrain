# ==========================================================================
# ==========================================================================
# convenience plotting routines
#
# By Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import hdf5storage as sio
import os.path
import matplotlib.pyplot as plt


def plotFitting(ax, WEs, fitting, distanceSettings, title, graphLabel=None):
    print("\n\n#####################################################################################################")
    print(f"# Results (in ({WEs[0]}, {WEs[-1]}):")
    for ds in distanceSettings:
        optimValDist = distanceSettings[ds][0].findMinMax(fitting[ds])
        print(f"# - Optimal {ds} = {optimValDist[0]} @ {np.round(WEs[optimValDist[1]], decimals=3)}")

        # color = next(ax._get_lines.prop_cycler)['color']
        plotFCpla, = ax.plot(WEs, fitting[ds])  #, color=color)
        ax.axvline(x=WEs[optimValDist[1]], ls='--')  #, c=color)
        if graphLabel is None:
            plotFCpla.set_label(ds)
        else:
            plotFCpla.set_label(graphLabel)
        ax.set_title(title)

    print("#####################################################################################################\n\n")


def listAllFiles(filePath):
    import glob
    expr = filePath.format("*")
    allFiles = glob.glob(expr)
    return allFiles


def loadAndPlotAx(ax, filePath,
                  distanceSettings, title,
                  WEs=None,
                  weName=None,
                  decimals=3,
                  empFilePath=None,
                  graphLabel=None):
    def processFile(fileName, ds):
        simValues = sio.loadmat(fileName)
        we = simValues[weName]
        # ---- and now compute the final FC and FCD distances for this G (we)!!! ----
        print(f" Loaded {fileName}:", end='', flush=True)
        if empFilePath is not None:
            measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
            dist = measure.distance(empValues[ds], simValues[ds])
        else:
            dist = simValues[ds]

        print(f" {ds}={dist}", flush=True)
        return we, dist

    # ==============================================
    if empFilePath is not None:
        processed = sio.loadmat(empFilePath)
        empValues = {}
        for ds in distanceSettings:
            empValues[ds] = processed[ds]

    allFiles = listAllFiles(filePath)  # collect all files, both when WEs are provided and when they are not.
    fitting = np.zeros((1+len(distanceSettings), len(allFiles)), dtype=np.float64)
    # fitting = {}
    # for pos, ds in enumerate(distanceSettings):
    #     fitting[pos+1] = np.array([], dtype=np.float64)

    if WEs is None:
        for wePos, fileName in enumerate(allFiles):
            for dspos, ds in enumerate(distanceSettings):
                we, value = processFile(fileName, ds)
                fitting[0, wePos] = we
                fitting[dspos+1, wePos] = value
        fitting = fitting[::, fitting[0,].argsort()[::]]
    else:
        wePos = 0
        for we in WEs:
            fileName = filePath.format(np.round(we, decimals=decimals))
            if os.path.exists(fileName):
                fitting[0, wePos] = we  # first column are the we values
                for dspos, ds in enumerate(distanceSettings):
                    _, value = processFile(fileName, ds)  # we do not need the we value...
                    fitting[dspos+1, wePos] = value
                wePos += 1

    data = {ds: fitting[1+pos,] for pos,ds in enumerate(distanceSettings)}
    plotFitting(ax, fitting[0], data, distanceSettings, title, graphLabel=graphLabel)


def loadAndPlot(filePath,
                distanceSettings,
                WEs=None,
                weName=None,
                decimals=3,
                empFilePath=None,
                title=''):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0,0])
    localTitle = f"computing graph " + title
    loadAndPlotAx(ax, filePath, distanceSettings, localTitle,
                  WEs=WEs, weName=weName, decimals=decimals, empFilePath=empFilePath)
    plt.legend(loc='upper right')
    plt.show()


def loadAndPlotMultipleGraphs(filePaths,
                              distanceSettings,
                              WEs=None,
                              weName=None,
                              decimals=3,
                              empFilePath=None,
                              titles=None):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0,0])
    for pos, path in enumerate(filePaths):
        if titles is not None:
            localTitle = f"computing graph " + titles[pos]
        else:
            localTitle = "computing graph"
        loadAndPlotAx(ax, path, distanceSettings, localTitle,
                      WEs=WEs, weName=weName, decimals=decimals, empFilePath=empFilePath, graphLabel=titles[pos])
    plt.legend(loc='upper right')
    plt.show()


def pltFullParmRange(globalTitle, fullFilePath, distanceSettings, parms, parmRange, graphShape):
    plt.rcParams.update({'font.size': 15})
    graph = parms.reshape(graphShape)
    fig, axs = plt.subplots(graph.shape[0], graph.shape[1])
    for ix, iy in np.ndindex(graph.shape):
        print(f"plotting graph({ix}, {iy}) -> parm={graph[ix,iy]} for parm in {parmRange}")
        filePath = fullFilePath.format(graph[ix,iy], '{}')
        localTitle = f"computing @ parm={graph[ix,iy]}"
        loadAndPlotAx(axs[ix,iy], filePath,
                      distanceSettings, localTitle,
                      parmRange)

    for ax in fig.get_axes():
        ax.label_outer()
    plt.suptitle(globalTitle)
    plt.show()
