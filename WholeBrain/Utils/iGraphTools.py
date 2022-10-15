"""
=============================
MISCELLANOUS HELPER FUNCTIONS
=============================

This module contains miscellaneous helper WholeBrain I have defined for better
or more personalised use of the Python iGraph interface.

I/O AND DATA CONVERSIONS
========================
myRead_Pajek
    Reads a network from a text file with Pajek format.
Array2iGraph
    Converts a 2D numpy array into an iGraph graph object.
"""
from __future__ import division, print_function

__author__ = "Gorka Zamora-Lopez"
__email__ = "Gorka.zamora@ymail.com"
__copyright__ = "Copyright 2015"
__license__ = "GPL"
__update__ = "22/11/2015"

import numpy as np
import igraph as ig


## I/O AND DATA CONVERSIONS ################################################
def myRead_Pajek_Binary(filepath, getlabels=False):
    """Reads a network from a a file with Pajek format.

    This is an incomplete version that ignores the weights of the links if
    even if they are provided in the file
    """
    # 0) OPEN THE FILE AND READ THE SIZE OF THE NETWORK
    pajekfile = open(filepath, 'r')
    firstline = pajekfile.readline()
    firstline = firstline.split()
    N = int(firstline[1])

    # Declare the basic graph of size N
    newgraph = ig.Graph(n=N)

    # 1) READ THE LABELS OF THE NODES IF WANTED
    if getlabels:

        # Security check, make sure that labels of nodes are listed in file
        line = pajekfile.readline()
        if line.split()[0] != '1':
            pajekfile.seek(1)
            print( 'LoadFromPajek() warning: No labels found to read.' )

        # If labels are in file continue reading the labels.
        else:
            # If labels are wrapped in between quotes
            try:
                idx1 = line.index('"') + 1
                #print( 'Reading quotes' )
                # Add the first label
                idx2 = line[idx1:].index('"')
                label = line[idx1:idx1+idx2]
                newgraph.vs[0]['name'] = label

                # And now read the labels for the rest of the nodes
                for i in range(1,N):
                    line = pajekfile.readline()
                    idx1 = line.index('"') + 1
                    idx2 = line[idx1:].index('"')
                    label = line[idx1:idx1+idx2]
                    newgraph.vs[i]['name'] = label

            # Otherwise, make a wild guess of what the label is
            except ValueError:
                #print( 'Reading splits' )
                # Add the first label
                label = line.split()[1]
                newgraph.vs[0]['name'] = label

                # And now read the labels of the rest of the nodes
                for i in range(1,N):
                    line = pajekfile.readline()
                    label = line.split()[1]
                    newgraph.vs[i]['name'] = label

    # 2) READ THE LINKS AND CREATE THE ADJACENCY MATRIX
    # 2.1) Find out whether the network is directed or undirected
    # while loop to skip empty lines if needed or the lines of the labels
    done = False
    while not done:
        line = pajekfile.readline()
        if line[0] == '*':
            if 'Edges' in line:
                directed = False
            elif 'Arcs' in line:
                directed = True
            else:
                print( 'Could not find whether network is directed or undirected' )
                break
            done = True

    # 2.2) Convert the graph in directed if necessary
    if directed: newgraph.to_directed()

    # 2.3) Read through file and gather all links into a list
    linklist = []
    for line in pajekfile:
        line = line.split()
        i = int(line[0]) - 1
        j = int(line[1]) - 1
        linklist.append((i,j))

    # 2.4) Include the edges to the graph
    newgraph.add_edges(linklist)

    # 3) CLOSE FILE AND RETURN RESULTS
    pajekfile.close()

    return newgraph

def Array2iGraph(adjmatrix, weighted=False):
    """Converts a 2D numpy array into an iGraph graph object.

    The function automatically detects whether the input network is directed,
    or is undirected but contains asymmetric weights.

    Parameters
    ----------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network. Weighted links are ignored.
    weighted : boolean (optional)
        Specifies whether the network is weighted or not.

    Returns
    -------
    iggraph : graph object recognised by iGraph.
        The graph representation of the adjacency matrix with corresponding
        un/directed and/or un/weighted properties.
    """
    # 0) Security check
    assert len(np.shape(adjmatrix)) == 2, \
        'Input array not an adjacency matrix. Array dimension has to be 2.'

    # 1) Find out whether the network is directed or has asymmetric weights
    diff = np.abs(adjmatrix - adjmatrix.T)
    if diff.max() > 10**-6: directed = True
    else: directed = False
    del diff

    # 2) Declare the igraph graph object
    N = len(adjmatrix)
    iggraph = ig.Graph()

    # 2.1) Add the nodes
    iggraph.add_vertices(N)

    # 2.2) Add the links if the network is DIRECTED
    if directed:
        # Make the graph directed
        iggraph.to_directed()

        # Create a list with the links.
        idx = np.where(adjmatrix)
        links = []
        for l in range(len(idx[0])):
            links.append( (idx[0][l],idx[1][l]) )

        # Add the links in the graph.
        iggraph.add_edges(links)

        # If the network is directed, include the weights.
        if weighted:
            values = adjmatrix[idx]
            iggraph.es[:]['weight'] = values

    # 2.2) Add the links if the network is UNDIRECTED
    else:
        # Make a copy of the adjacency matrix to be modified.
        newadjmatrix = adjmatrix.copy()
        idx = np.triu_indices(N,k=1)
        newadjmatrix[idx] = 0

        # Create a list with the links.
        idx = np.where(newadjmatrix)
        links = []
        for l in range(len(idx[0])):
            links.append( (idx[0][l],idx[1][l]) )

        # Add the links in the graph.
        iggraph.add_edges(links)

        # If the network is directed, include the weights.
        if weighted:
            values = newadjmatrix[idx]
            iggraph.es[:]['weight'] = values

        # Clean trash
        del newadjmatrix

    return iggraph
