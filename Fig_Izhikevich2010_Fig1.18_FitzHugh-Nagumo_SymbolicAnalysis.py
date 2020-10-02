# ==========================================================================
# ==========================================================================
# Test to draw phase diagrams, isoclines et al. of several ODEs
#
# by Gustavo Patow
# ==========================================================================
# ==========================================================================
from functions.SymAnalysis import *
import sympy as sm

# ==========================================================================
# ODE: FitzHugh–Nagumo model
# Eugene M. Izhikevich and Richard FitzHugh (2006), Scholarpedia, 1(9):1349. doi:10.4249/scholarpedia.1349
# http://www.scholarpedia.org/article/FitzHugh-Nagumo_model
# Also at:
# https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
# values from [Izhikevich, Dynamical Systems in Neuroscience, 2010, fig 1.18]
# ==========================================================================
I_ext = 0.5
a = 0.7
b = 0.8
tau = 12.5  # 1/tau = 0.08
def FitzHugh_Nagumo():
    v, w = sm.symbols('v, w')
    eq1 = v - v**3/3 - w + I_ext
    eq2 = (v + a - b*w)/tau
    return [v, w], [eq1, eq2]


# ==========================================================================
# ==========================================================================
# ==========================================================================
# ======================== FitzHugh-Nagumo... ==============================
print("=====================================================")
print("=  FitzHugh-Nagumo Equation to compute symbolically =")
print("=====================================================")
print("Equation:", FitzHugh_Nagumo())
print("=====================================================")
interval = {'left': -2.5, 'right': 2.5, 'bottom': -2.5, 'top': 2.5}
# plotODEInt(FitzHugh_Nagumo, [1, 0.01])
plotPhasePlane(FitzHugh_Nagumo, interval,
               trajectories=[[1, 0.01], [-2.5,-0.75]],
               background='flow', type='solve explicit')

# ======================== Genetic Toggle... ==========================
# interval = {'left': 0, 'right': 6, 'bottom': 0, 'top': 6}
# plotPhasePlane(toggle, interval, trajectories=[[0.01, 1], [1, 0.01], [3, 6], [6, 3]])
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
