# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of several ODEs
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import functions.NumAnalysis as numA

# ==========================================================================
# ODE: FitzHugh–Nagumo model
# Eugene M. Izhikevich and Richard FitzHugh (2006), Scholarpedia, 1(9):1349. doi:10.4249/scholarpedia.1349
# http://www.scholarpedia.org/article/FitzHugh-Nagumo_model
# Also at:
# https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
# values and equations taken from [Izhikevich, Dynamical Systems in Neuroscience, 2010, fig 4.20]
# ==========================================================================
class FitzHugh_Nagumo_model:
    def __init__(self):
        self.I_ext = 0
        self.a = 0.7
        self.b = 0.01
        self.c = 0.02
        self.index = 0

    def dfun(self, x, t=None):
        # This formulation is slightly different from the one I used for Fig 1.18 in Izhikevich book,
        # but it is the one used for this figure...
        # Uses equations 4.11 and 4.12 from the book
        v, w = x
        dVdt = v*(self.a-v)*(v-1)-w+self.I_ext
        dwdt = self.b*v-self.c*w
        return np.array([dVdt, dwdt])

    def parmNames(self):
        return ['u', 'w']

    def getObservationVarName(self):
        return self.parmNames()[self.index]

    def setControlParm(self, lmbd):
        self.I_ext = lmbd

    def getControlParmName(self):
        return '$I_{ext}$'

    def getObservationVar(self, simVars, lmbd):
        return np.array([simVars[self.index]])


# ==========================================================================
# ==========================================================================
# ======================== FitzHugh-Nagumo... ==============================
print("=====================================================")
print("=  FitzHugh-Nagumo Equation to compute numerically  =")
print("=      (from [Izhikevich2010, fig 4.20])            =")
print("=====================================================")
# print("Equation:", FitzHugh_Nagumo())
print("=====================================================")
model = FitzHugh_Nagumo_model()
interval = {'left': -0.4, 'right': 1., 'bottom': -0.05, 'top': 0.22}

plt.rcParams.update({'font.size': 15})
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ratio = 0.8

model.a = 0.1
numA.plotPhasePlane(ax1, model, interval,
                    # trajectories=[[-0.5, -0.75], [-2.5,-0.75], [0, 0.4]],
                    background='quiver-B&W')  # quiver, flow

model.a = -0.1
numA.plotPhasePlane(ax2, model, interval,
                    # trajectories=[[-0.5, -0.75], [-2.5,-0.75], [0, 0.4]],
                    background='quiver-B&W')

for ax in [ax1, ax2]:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # print((xmax-xmin)/(ymax-ymin))
    ax.set_aspect(abs((xmax-xmin)/(ymax-ymin))*ratio, adjustable='box-forced')

# plt.axes().set_aspect('equal', 'datalim')
plt.suptitle('[Izhikevich2010, fig 4.20]', fontsize=24)
plt.show()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
