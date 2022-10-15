# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
# ==========================================================================
from sympy import exp

print("Going to use the Symbolic Dynamic Mean Field (DMF) neuronal model...")

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) Model Constants
# --------------------------------------------------------------------------
taon = 100.
taog = 10.
gamma_e = 0.641 / 1000
gamma_i = 1. / 1000.
J_NMDA = 0.15       # [nA] NMDA current
I0 = 0.382  #.397  # [nA] overall effective external input
Jexte = 1.
Jexti = 0.7
w = 1.4
we = 2.1        # Global coupling scaling (G in the paper)
SC = None       # Structural connectivity (should be provided externally)


# transfer WholeBrain:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae=310.
be=125.
de=0.16
def phie(x):
    y = (ae*x-be)
    # if (y != 0):
    return y/(1.-exp(-de*y))
    # else:
    #     return 0

# transfer function: inhibitory
ai=615.
bi=177.
di=0.087
def phii(x):
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1.-exp(-di*y))
    # else:
    #     return 0

# transfer WholeBrain used by the simulation...
He = phie
Hi = phii

J = None    # WARNING: In general, J must be initialized outside!

# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
def dfun(simVars, I_external):
    [sn, sg] = simVars
    xn = I0 * Jexte + w * J_NMDA * sn + we * J_NMDA * (SC * sn) - J * sg + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    xg = I0 * Jexti + J_NMDA * sn - sg  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    rn = He(xn)  # Calls He(xn). r^E = H^E(I^E) in the paper (7)
    rg = Hi(xg)  # Calls Hi(xg). r^I = H^I(I^I) in the paper (8)
    dsn = -sn / taon + (1. - sn) * gamma_e * rn
    dsg = -sg / taog + rg * gamma_i
    return [dsn, dsg]


# ==========================================================================
# ==========================================================================
# ==========================================================================
