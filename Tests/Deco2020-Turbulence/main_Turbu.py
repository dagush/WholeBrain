# =======================================================================
# Turbulence framework, main part. From:
# Gustavo Deco, Morten L. Kringelbach, Turbulent-like Dynamics in the Human Brain,
# Cell Reports, Volume 33, Issue 10, 2020, 108471, ISSN 2211-1247,
# https://doi.org/10.1016/j.celrep.2020.108471.
# (https://www.sciencedirect.com/science/article/pii/S2211124720314601)
#
# Part of the Thermodynamics of Mind framework:
# Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
# Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
# https://doi.org/10.1016/j.tics.2024.03.009
#
# Code by Gustavo Deco, 2020.
# Translated by Marc Gregoris, May 21, 2024
# Refactored by Gustavo Patow, June 9, 2024
# =======================================================================
import scipy.io as sio

import WholeBrain.Utils.DataLoaders.HCP_schaefer1000 as schaefer1000
DL = schaefer1000.HCP()
NPARCELLS = DL.N()
coords = DL.get_GlobalData()['coords']

import WholeBrain.Observables.BOLDFilters as BOLDFilters
BOLDFilters.TR = DL.TR()
BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08

from WholeBrain.Observables import Turbulence

Turbu = Turbulence.Turbulence(coords, ignoreNaNs=True)

import WholeBrain.Utils.decorators as decorators


dataPath = './Data_Produced/'


@decorators.loadOrCompute
def from_fMRI(ts):
    return Turbu.from_fMRI(ts)


@decorators.loadOrCompute
def from_fMRI_surrogate(ts):
    return Turbu.from_surrogate(ts)


# dictionaries for each subject
rspatime_lista = []
rspa_lista = []
rspatime_su_lista = []
rspa_su_lista = []
Rtime_lista = []
Rtime_su_lista = []
acfspa_lista = []
acfspa_su_lista = []
acftime_lista = []
acftime_su_lista = []

# decorators.forceCompute = True  # Use this to force recomputations.

for subj in DL.get_classification():
    subjData = DL.get_SubjectData(subj)
    ts = subjData[subj]['timeseries']
    # ------ main analysis
    subjPath = dataPath + f'turbu_{subj}.mat'
    turbuRes = from_fMRI(ts, subjPath)
    rspatime_lista.append(turbuRes['Rspatime'])  # Rspatime
    rspa_lista.append(turbuRes['Rspa'])  # Rspa
    Rtime_lista.append(turbuRes['Rtime'])  # Rtime
    acfspa_lista.append(turbuRes['acfspa'])  # acfspa
    acftime_lista.append(turbuRes['acftime'])  # acftime
    # ------ Surrogate analysis
    subjPath = dataPath + f'turbu_{subj}_surrogate.mat'
    turbu_su = from_fMRI_surrogate(ts, subjPath)
    rspatime_su_lista.append(turbu_su['Rspatime'])  # Rspatime_su
    rspa_su_lista.append(turbu_su['Rspa'])  # Rspa_su
    Rtime_su_lista.append(turbu_su['Rtime'])  # Rtime_su
    acfspa_su_lista.append(turbu_su['acfspa'])  # acfspa_su
    acftime_su_lista.append(turbu_su['acftime'])  # acftime_su
    print(f"done {subj} !!")

sio.savemat(dataPath + 'turbu_emp.mat',{
    'Rspatime': rspatime_lista,
    'Rspa': rspa_lista,
    'Rtime': Rtime_lista,
    'acfspa': acfspa_lista,
    'acftime': acftime_lista,
    # ------ Surrogate
    'Rspatime_su': rspatime_su_lista,
    'Rspa_su': rspa_su_lista,
    'Rtime_su': Rtime_su_lista,
    'acfspa_su': acfspa_su_lista,
    'acftime_su': acftime_su_lista,
})

print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF