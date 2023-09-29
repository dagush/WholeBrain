# --------------------------------------------------------------------------------------
# Plot Figure 4 in:
# [StefanovskiEtAl2019] Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
#           R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
#           "Linking molecular pathways and large-scale computational modeling to assess candidate
#           disease mechanisms and pharmacodynamics in Alzheimer's disease."
#           Front. Comput. Neurosci., 13 August 2019 | https://doi.org/10.3389/fncom.2019.00054
#
# --------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Code taken from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def plotMatrixAsFancyGraph(M, fig, ax):
    plt.rcParams.update({'font.size': 15})
    import networkx as nx

    # Keep only the 5% of the largest values
    M2 = largest_indices(M, int(M.size * .05))
    print(f"selected {M2[0].size} nodes")
    compl = M==M
    compl[M2] = False
    M[compl] = 0.0

    color_map = ['red']*180 + ['blue']*180 + ['orange']*9 + ['green']*9 + ['black']
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
    radii = 0.5
    d = [(d[node]+1) * radii for node in G.nodes()]
    pos=nx.fruchterman_reingold_layout(G, k=1/np.sqrt(M2[0].size))
    nx.draw(G, with_labels=False, node_size=d, node_color=color_map, pos=pos, ax=ax)
    ax.title.set_text("Structural connectivity Graph")

    ax.legend(handles=legend_elements)


def plotFancyMatrix(M, fig, ax):
    cm = plt.get_cmap('rainbow')
    img = ax.matshow(M, cmap=cm)
    fig.colorbar(img, ax=ax)

    ax.title.set_text("Structural Connectivity Matrix")
    ax.set_xlabel("Regions")
    ax.set_ylabel("Regions")



if __name__ == '__main__':
    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    # load SC
    sc_folder = "../Data_Raw/surrogate_AD"
    SCnorm = np.loadtxt(sc_folder+"/avg_healthy_normSC_mod_379.txt")
    print("# of elements in AVG connectome: {}".format(SCnorm.shape))

    # Using a figure to use it as a parameter when calling nx.draw_networkx
    plt.rcParams.update({'font.size': 15})
    f = plt.figure(2)
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    plotFancyMatrix(SCnorm, f, ax1)
    plotMatrixAsFancyGraph(SCnorm, f, ax2)
    plt.show()
