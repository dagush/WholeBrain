# ==========================================================================
# ==========================================================================
# convenience plotting routines
#
# By Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt


def plotFitting(ax, WEs, fitting, distanceSettings, title):
    print("\n\n#####################################################################################################")
    print(f"# Results (in ({WEs[0]}, {WEs[-1]}):")
    for ds in distanceSettings:
        optimValDist = distanceSettings[ds][0].findMinMax(fitting[ds])
        print(f"# - Optimal {ds} = {optimValDist[0]} @ {np.round(WEs[optimValDist[1]], decimals=3)}")

        color = next(ax._get_lines.prop_cycler)['color']
        plotFCpla, = ax.plot(WEs, fitting[ds], color=color)
        ax.axvline(x=WEs[optimValDist[1]], ls='--', c=color)
        plotFCpla.set_label(ds)
        ax.set_title(title)

    print("#####################################################################################################\n\n")


def loadAndPlotAx(ax, filePath,
                  distanceSettings, title,
                  WEs,
                  decimals=3,
                  empFilePath=None):
    if empFilePath is not None:
        processed = sio.loadmat(empFilePath)
        empValues = {}
        for ds in distanceSettings:
            empValues[ds] = processed[ds]

    realWEs = np.array([], dtype=np.float64)
    fitting = {}
    for ds in distanceSettings:
        fitting[ds] = np.array([], dtype=np.float64)

    for we in WEs:
        fileName = filePath.format(np.round(we, decimals=decimals))
        if Path(fileName).is_file():
            simValues = sio.loadmat(fileName)
            realWEs = np.append(realWEs, we)

            # ---- and now compute the final FC and FCD distances for this G (we)!!! ----
            print(f"Loaded {fileName}:", end='', flush=True)
            for ds in distanceSettings:
                if empFilePath is not None:
                    measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
                    dist = measure.distance(empValues[ds], simValues[ds])
                else:
                    dist = simValues[ds]
                fitting[ds] = np.append(fitting[ds], dist)
                print(f" {ds}={dist}", end='', flush=True)
            print()

    plotFitting(ax, realWEs, fitting, distanceSettings, title)


def loadAndPlot(filePath,
                distanceSettings,
                WEs,
                decimals=3,
                empFilePath=None):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0,0])
    localTitle = f"computing graph"
    loadAndPlotAx(ax, filePath, distanceSettings, localTitle, WEs, decimals=decimals, empFilePath=empFilePath)
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
