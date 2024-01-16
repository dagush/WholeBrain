# --------------------------------------------------------------------------------------
# Alexandersen's model of Alzheimer's disease progression
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
from Progression.ProgrModels.baseModel import base_progression


# -------------------------------------------------------------------------------
class Alexandersen2023(base_progression):
    N = None; M = None
    edges = None
    neighbours = None
    a0 = 0.75; ai = 1; api = 1; aii = 1
    b0 = 1; bi = 1; bii = 1; biii = 1
    gamma = 0; delta = 0.95
    bpi = 1; c1 = 1; c2 = 1; c3 = 1; c_init = 0; c_min = 0
    k1 = 1; k2 = 1; k3 = 1
    rho = 10 ** (-3); a_min = False; a_max = False; b_min = False

    def __init__(self):
        super().__init__()

    # -------------------------------------------------------------------------------
    # spreading dynamics
    # -------------------------------------------------------------------------------
    def dfun(self, t, y):
        # set up variables as lists indexed by node k
        u = np.array([y[i] for i in range(self.N)])
        up = np.array([y[i] for i in range(self.N, 2 * self.N)])
        v = np.array([y[i] for i in range(2 * self.N, 3 * self.N)])
        vp = np.array([y[i] for i in range(3 * self.N, 4 * self.N)])
        qu = np.array([y[i] for i in range(4 * self.N, 5 * self.N)])
        qv = np.array([y[i] for i in range(5 * self.N, 6 * self.N)])
        a = np.array([y[i] for i in range(6 * self.N, 7 * self.N)])
        b = np.array([y[i] for i in range(7 * self.N, 8 * self.N)])
        c = np.array([y[i] for i in range(8 * self.N, 9 * self.N)])
        w = np.array([y[i] for i in range(9 * self.N, 9 * self.N + self.M)])

        # update laplacian from m weights
        L = np.zeros((self.N, self.N))
        for i in range(self.M):
            n, m = self.edges[i]
            # set (n,m) in l
            L[n, m] = -w[i]
            L[m, n] = L[n, m]
            # update (n,n) and (m,m) in l
            L[n, n] += w[i]
            L[m, m] += w[i]

        # check if l is defined correctly
        for i in range(self.N):
            if abs(sum(L[i, :])) > 10 ** -10:
                print('L is ill-defined')
                print(sum(L[i, :]))

        # scale Laplacian by diffusion constant
        L = self.rho * L

        # nodal dynamics
        du, dup, dv, dvp, dqu, dqv, da, db, dc = [[] for _ in range(9)]
        neighbours = self.neighbours
        for k in range(self.N):
            # index list of node k and its neighbours
            neighbours_k = neighbours[k] + [k]

            # heterodimer dynamics
            duk = sum([-L[k, l] * u[l] for l in neighbours_k]) + self.a0 - self.ai * u[k] - self.aii * u[k] * up[k]
            dupk = sum([-L[k, l] * up[l] for l in neighbours_k]) - self.api * up[k] + self.aii * u[k] * up[k]
            dvk = self.pf[k] * sum([-L[k, l] * v[l] for l in neighbours_k]) + self.b0 - self.bi * v[k] \
                  - self.bii * v[k] * vp[k] - self.biii * up[k] * v[k] * vp[k]
            dvpk = self.pf[k] * sum([-L[k, l] * vp[l] for l in neighbours_k]) - self.bpi * vp[k] \
                   + self.bii * v[k] * vp[k] + self.biii * up[k] * v[k] * vp[k]
            ## append
            du.append(duk)
            dup.append(dupk)
            dv.append(dvk)
            dvp.append(dvpk)

            # damage dynamics
            dquk = self.k1 * up[k] * (1 - qu[k])
            dqvk = self.k2 * vp[k] * (1 - qv[k]) + self.k3 * up[k] * vp[k]
            ## append
            dqu.append(dquk)
            dqv.append(dqvk)

            # excitatory-inhibitory dynamics
            dak = self.c1 * qu[k] * (self.a_max - a[k]) * (a[k] - self.a_min) - self.c2 * qv[k] * (a[k] - self.a_min)
            dbk = -self.c3 * qu[k] * (b[k] - self.b_min)
            dck = -self.c3 * qu[k] * (c[k] - self.c_min)
            ## append
            da.append(dak)
            db.append(dbk)
            dc.append(dck)

        # connecctivity dynamics
        dw = []
        for i in range(self.M):
            # extract edge
            n, m = self.edges[i]

            # axonopathy dynamcs
            dwi = -self.gamma * w[i] * (qv[n] + qv[m])
            ## append
            dw.append(dwi)

        # pack right-hand side
        rhs = [*du, *dup, *dv, *dvp, *dqu, *dqv, *da, *db, *dc, *dw]
        return rhs


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF