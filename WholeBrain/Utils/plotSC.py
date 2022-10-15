# =====================================================================================
# Methods to plot a few properties SC matrices
# =====================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plotSCHistogram(ax, SC, subjectName):
    # plt.rcParams["figure.figsize"] = (7,5)
    # plt.rcParams["figure.dpi"] = 300
    # plt.figure()  #num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    bins = 50 #'auto'
    n, bins, patches = ax.hist(SC.flatten(), bins=bins, color='#0504aa', alpha=0.7, histtype='step')  #, rwidth=0.85)
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel('SC weights')
    ax.set_ylabel('Counts')
    ax.set_title("SC histogram ({}: {})".format(subjectName, SC.shape), fontweight="bold", fontsize="18")
    # plt.savefig("./Results/Abeta/"+subject+".png", dpi=200)
    # plt.close()


def plotSC(ax, SC, subjectName):
    ax.imshow(np.asarray(SC))
    ax.set_xlabel("Regions")
    ax.set_ylabel("Regions")
    ax.set_title("Subject {}".format(subjectName))
    print("Scale({}): Max={}, Min={}".format(subjectName, np.max(SC), np.min(SC)))


def plotSC_and_Histogram(subjectName, SCnorm, plotColorBar = True):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 2)
    ax1 = fig.add_subplot(grid[0,0])
    plotSC(ax1, SCnorm, subjectName)
    if plotColorBar:
        img = ax1.get_images()[0]
        fig.colorbar(img)
    ax2 = fig.add_subplot(grid[0,1])
    plotSCHistogram(ax2, SCnorm, subjectName)
    plt.suptitle("Structural Connectivity ({})".format(subjectName), fontweight="bold", fontsize="18", y=1.05)
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    # img = ax1.get_images()[0]
    # fig.colorbar(img, cax=cbar_ax)
    plt.show()


def justPlotSC(subjectName, SCnorm, plottingFunction):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0,0])
    plottingFunction(ax, SCnorm, subjectName)
    plt.show()


# =====================================================================================================================
# get indices of n maximum values in a numpy array
# Code taken from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
# =====================================================================================================================
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# =====================================================================================================================
# Plot SC as a graph
# =====================================================================================================================
def plotSCMatrixAsFancyGraph(M):
    plt.rcParams.update({'font.size': 25})
    import networkx as nx

    # Using a figure to use it as a parameter when calling nx.draw_networkx
    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)

    # Keep only the 5% of the largest values
    M2 = largest_indices(M, int(M.size * .05))
    print(f"selected {M2[0].size} nodes")
    compl = M==M
    compl[M2] = False
    M[compl] = 0.0

    color_map = ['red']*180 + ['blue']*180 + ['green']*9 + ['orange']*9 + ['black']
    legend_elements = [Line2D([0], [0], marker='o', color='red', label='Right Cortex', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='blue', label='Left Cortex', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='green', label='Left Subcortical', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='orange', label='Right Subcortical', linewidth=0, markersize=15),
                       Line2D([0], [0], marker='o', color='black', label='Brainstem', linewidth=0, markersize=15)]

    # Now, plot it!!!
    G = nx.from_numpy_matrix(M)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"resulting edges => {G.number_of_edges()}")
    d = nx.degree(G)
    d = [(d[node]+1) * 20 for node in G.nodes()]
    pos=nx.fruchterman_reingold_layout(G, k=1/np.sqrt(M2[0].size))
    nx.draw(G, with_labels=False, node_size=d, node_color=color_map, pos=pos, ax=ax)
    plt.title("Structural connectivity Graph")

    plt.legend(handles=legend_elements)
    plt.show()


def plotFancyMatrix(M, axisName="Regions", matrixName="Structural Connectivity Matrix", showAxis='on', fontSize=25, cmap='viridis'):
    plt.rcParams.update({'font.size': fontSize})
    plt.matshow(M, cmap=cmap)
    plt.colorbar()
    plt.title(matrixName)
    plt.xlabel(axisName)
    plt.ylabel(axisName)
    plt.axis(showAxis)
    plt.show()


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================eof
