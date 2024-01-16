# --------------------------------------------------------------------------------------
# Handling of delay information
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# Code by Christoffer Alexandersen
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
import csv

# ----------------------------------------
# Matrix of delays. Input for rhythms series
# ----------------------------------------
def build_delay_matrix(distances, transmission_speed, N, discretize=40):
    # distances should be a list of 3-tuples like (node1, node2, distance)
    delay_matrix = np.zeros((N,N))
    for n,m,distance in distances:
        delay_matrix[n,m] = distance/transmission_speed
        delay_matrix[m,n] = delay_matrix[n,m]

    if discretize:
        nonzero_inds = np.nonzero(delay_matrix)
        max_delay = np.amax(delay_matrix)
        # n_delays = np.count_nonzero(delay_matrix)

        # lower_bounds = np.arange(0, discretize) * max_delay/discretize
        upper_bounds = np.zeros((discretize))
        for i in range(discretize):
            upper_bounds[i] = (i+1) * max_delay/discretize
        for l in range(len(nonzero_inds[0])):
            i = nonzero_inds[0][l]
            j = nonzero_inds[1][l]
            for k in range(discretize):
                w_ij = delay_matrix[i,j]
                if w_ij <= upper_bounds[k]:
                    w_ij = upper_bounds[k]  # round to upper (GorielyBick)
                    delay_matrix[i,j] = w_ij
                    delay_matrix[j,i] = w_ij
                    break

    return delay_matrix


def loadDistances(path):
    # create delay matrix per Bick & Goriely
    distances = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        node_i = 0
        for row in reader:
            node_j = 0
            for col in row:
                if float(col) > 0:
                    distances.append((node_i, node_j, float(col) / 10))
                node_j += 1
            node_i += 1
    return distances
