# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
# ==========================================================================
from sympy import exp
import sympy as sm

print("Going to use the Symbolic Dynamic Mean Field (DMF) neuronal model...")

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) Model Constants
# --------------------------------------------------------------------------
taon = sm.symbols('taon')  # 100.
taog = sm.symbols('taog')  # 10.
gamma_e = sm.symbols('gamma_e')  # 0.641 / 1000
gamma_i = sm.symbols('gamma_i')  # 1. / 1000.
J_NMDA = sm.symbols('J_NMDA')  # 0.15       # [nA] NMDA current
I0 = sm.symbols('I0')  # 0.382  #.397  # [nA] overall effective external input
Jexte = sm.symbols('Jexte')  # 1.
Jexti = sm.symbols('Jexti')  # 0.7
w = sm.symbols('w')  # 1.4
we = sm.symbols('we')  # 2.1        # Global coupling scaling (G in the paper)
C = sm.symbols('C')  # 0.
J = sm.symbols('di')  # 1.

# transfer functions:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae=sm.symbols('a_E')  # 310.
be=sm.symbols('b_E')  # 125.
de=sm.symbols('d_E')  # 0.16
def phie(x):
    y = (ae*x-be)
    # if (y != 0):
    return y/(1.-exp(-de*y))
    # else:
    #     return 0

# transfer function: inhibitory
ai=sm.symbols('a_I')  # 615.
bi=sm.symbols('b_I')  # 177.
di=sm.symbols('d_I')  # 0.087
def phii(x):
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1.-exp(-di*y))
    # else:
    #     return 0

# transfer functions used by the simulation...
He = phie
Hi = phii


# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
sn, sg = sm.symbols('sn, sg')
I_e, I_i = sm.symbols('I^E_k, I^I_k')
r_e, r_i = sm.symbols('r_e, r_i')
def dfun(C, I_external):
    xn = I0 * Jexte + w * J_NMDA * sn + we * J_NMDA * (C * sn) - J * sg + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    xg = I0 * Jexti + J_NMDA * sn - sg  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)

    rn = He(I_e)  # Calls He(xn). r^E = H^E(I^E) in the paper (7)
    rg = Hi(I_i)  # Calls Hi(xg). r^I = H^I(I^I) in the paper (8)

    dsn = -sn / taon + (1. - sn) * gamma_e * r_e
    dsg = -sg / taog + r_i * gamma_i
    return xn, xg, rn, rg, dsn, dsg


# ==========================================================================
# some definitions...
# ==========================================================================
import functions.Stimuli.constant as stimuli
stimuli.onset = 0.
stimuli.amp = sm.symbols('I_ext')  # 0.
xn, xg, rn, rg, dsn, dsg = dfun(C, stimuli.stimulus(0.))
print('Equations:')
print('X_e=', xn, '\nX_i=', xg)
print('r_e=', rn, '\nr_i=', rg)
print('dSe/dt=', dsn, '\ndSi/dt=', dsg)

# print('\nSubstitutions:')
# rn2 = sm.simplify(rn.subs(I_e, xn))
# rg2 = sm.simplify(rg.subs(I_i, xg))
# print('r_e=', rn2, '\nr_i=', rg2)
# dsn2 = sm.simplify(dsn.subs(r_e, rn2))
# dsg2 = sm.simplify(dsg.subs(r_i, rg2))
# print('dSe/dt=', dsn2, '\ndSi/dt=', dsg2)

print('\nComputations:')
print('dr_e/dt=', sm.diff(rn, I_e))
# print('dr_e/dt=', sm.pretty(sm.simplify(sm.diff(rn, I_e))))
print('LaTeX  =', sm.latex(sm.simplify(sm.diff(rn, I_e))))
print()
print('dr_i/dt=', sm.diff(rg, I_i))
# print('dr_e/dt=', sm.pretty(sm.simplify(sm.diff(rn, I_e))))
print('LaTeX  =', sm.latex(sm.simplify(sm.diff(rg, I_i))))

