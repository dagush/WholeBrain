# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# GET_COMPONENTS     connected components
#
#    comps,comp_sizes = get_components(adj)
#
#    Returns the components of an undirected graph specified by the binary and
#    undirected adjacency matrix adj. Components and their constitutent nodes are
#    assigned the same index and stored in the vector, comps. The vector, comp_sizes,
#    contains the number of nodes belonging to each component.
#
#    Inputs:         adj,    binary and undirected adjacency matrix
#
#    Outputs:      comps,    vector of component assignments for each node
#             comp_sizes,    vector of component sizes
#
#    Note: disconnected nodes will appear as components with a component size of 1
#
#    Original code by J Goni, University of Navarra and Indiana University, 2009/2011
#    Translated by Gustavo Patow, 2021
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Some definitions and explanations:
# ==================================
#
# From https://en.wikipedia.org/wiki/Perfect_matching
# In graph theory, a perfect matching in a graph is a matching that covers every vertex of the graph. More formally,
# given a graph G = (V, E), a perfect matching in G is a subset M of E, such that every vertex in V is adjacent to
# exactly one edge in M.
#
# From https://en.wikipedia.org/wiki/Maximum_cardinality_matching
# Maximum cardinality matching is a fundamental problem in graph theory.[1] We are given a graph G, and the goal is
# to find a matching containing as many edges as possible, that is, a maximum cardinality subset of the edges such
# that each vertex is adjacent to at most one edge of the subset. As each edge will cover exactly two vertices,
# this problem is equivalent to the task of finding a matching that covers as many vertices as possible.
#
# From https://en.wikipedia.org/wiki/Dulmage%E2%80%93Mendelsohn_decomposition
# In graph theory, the Dulmage–Mendelsohn decomposition is a partition of the vertices of a bipartite graph into
# subsets, with the property that two adjacent vertices belong to the same subset if and only if they are paired
# with each other in a perfect matching of the graph. It is named after A. L. Dulmage and Nathan Mendelsohn, who
# published it in 1958. A generalization to any graph is the Edmonds–Gallai decomposition, using the Blossom algorithm.
#
# From https://en.wikipedia.org/wiki/Blossom_algorithm
# The blossom algorithm is an algorithm in graph theory for constructing maximum matchings on graphs. The algorithm
# was developed by Jack Edmonds in 1961,[1] and published in 1965.[2] Given a general graph G = (V, E), the algorithm
# finds a matching M such that each vertex in V is incident with at most one edge in M and |M| is maximized. The
# matching is constructed by iteratively improving an initial empty matching along augmenting paths in the graph.
# Unlike bipartite matching, the key new idea is that an odd-length cycle in the graph (blossom) is contracted to a
# single vertex, with the search continuing iteratively in the contracted graph.


import numpy as np
from scipy.sparse import eye


def get_components(adj):
    if adj.shape[0] != adj.shape[1]:
        raise NameError('this adjacency matrix is not square')
    
    if not np.any(adj - np.triu(adj), axis=0):
        adj = np.logical_or(adj, adj.T)
    
    # if main diagonal of adj do not contain all ones, i.e. autoloops
    if np.sum(np.diag(adj)) != adj.shape[0]:
        # the main diagonal is set to ones
        adj = np.logical_or(adj, eye(adj.shape))
    
    # Dulmage-Mendelsohn decomposition
    [useless1,p,useless2,r] = dmperm(adj)
    
    # p indicates a permutation (along rows and columns)
    # r is a vector indicating the component boundaries
    
    # List including the number of nodes of each component. ith entry is r(i+1)-r(i)
    comp_sizes = diff(r)
    
    # Number of components found.
    num_comps = numel(comp_sizes)
    
    # initialization
    comps = zeros(1,size(adj,1))
    
    # first position of each component is set to one
    comps[r(1:num_comps)] = ones(1,num_comps)
    
    # cumulative sum produces a label for each component (in a consecutive way)
    comps = cumsum(comps)
    
    # re-order component labels according to adj.
    comps[p] = comps

    return num_comps, comps
    
