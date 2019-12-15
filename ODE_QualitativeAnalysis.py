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
# ODE: Simple systems from different sources
# ==========================================================================
# From http://www.math.ubc.ca/~israel/m215/nonlin/nonlin.html
def simpleODE_1():
    x, y = sm.symbols('x, y')  # negative=False)
    # beta, gamma, n = sm.symbols('beta, gamma, n', real=True, constant=True)
    eq1 = x*(y-1)
    eq2 = y*(x-1)
    return [x,y], [eq1,eq2]

# From https://mcb.berkeley.edu/courses/mcb137/exercises/Nullclines.pdf
def simpleODE_2():
    x, y = sm.symbols('x, y')  # negative=False)
    # beta, gamma, n = sm.symbols('beta, gamma, n', real=True, constant=True)
    eq1 = x*(1-x)-x*y  # x*(y-1)
    eq2 = 2*y*(1-y**2/2)-3*x**2*y  # y*(x-1)
    return [x,y], [eq1,eq2]


# ==========================================================================
# ODE: Bistable gene circuit (genetic toggle)
# from
# Biomolecular Feedback Systems, chapter 3
# Domitilla Del Vecchio and Richard M. Murray
# http://www.cds.caltech.edu/~murray/BFSwiki/index.php/Main_Page
# ==========================================================================
gamma = 2
beta = 5
n = 2

# gamma = 1
# beta = 5
# n = 2

def toggle():
    a, b = sm.symbols('a, b')  # negative=False)
    # beta, gamma, n = sm.symbols('beta, gamma, n', real=True, constant=True)
    eq1 = beta / (1 + b**n) - a
    eq2 = gamma * (beta / (1 + a**n) - b)
    return [a,b], [eq1,eq2]


# ==========================================================================
# ODE: FitzHugh–Nagumo model
# Eugene M. Izhikevich and Richard FitzHugh (2006), Scholarpedia, 1(9):1349. doi:10.4249/scholarpedia.1349
# http://www.scholarpedia.org/article/FitzHugh-Nagumo_model
# Also at:
# https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
# ==========================================================================
I_ext = 0.5
a = 0.7
b = 1.8
tau = 12.5  # 1/tau = 0.08
def FitzHugh_Nagumo():
    v, w = sm.symbols('v, w')
    eq1 = v - v**3/3 - w + I_ext
    eq2 = (v + a - b*w)/tau
    return [v, w], [eq1, eq2]


# ==========================================================================
# Van der Pol oscillator
# https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
# ==========================================================================
mu = 2.
def vanDerPol():
    x, y = sm.symbols('x, y')
    # Applying the Liénard transformation {\displaystyle y=x-x^{3}/3-{\dot {x}}/\mu }y=x-x^{3}/3-{\dot {x}}/\mu ,
    # where the dot indicates the time derivative,  the Van der Pol oscillator can be written in its
    # two-dimensional form:
    eq1 = mu * (x - x**3/3 - y)
    eq2 = x / mu
    return [x,y], [eq1,eq2]


# ==========================================================================
# ==========================================================================
# ==========================================================================

# ======================== Simple ODE 2... ==========================
# print("=================================================")
# print("=  simpleODE_2 Equation to compute symbolically =")
# print("=================================================")
# print("Equation:", simpleODE_2())
# print("=====================================================")
# interval = {'left': -2.5, 'right': 2.5, 'bottom': -2.5, 'top': 2.5}
# plotODEInt(simpleODE_2, [1, 0.01])
# plotPhasePlane(simpleODE_2, interval,
#                background='flow', type='solve implicit')

# ======================== FitzHugh-Nagumo... ==========================
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
