# =======================================================================
# Turbulence framework, plotting part. From:
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
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import WholeBrain.Utils.p_values as pValues


dataPath = './Data_Produced/'


def calculate_stats(datas):
    means = np.nanmean(datas, axis=0)
    stds = np.nanstd(datas, axis=0)
    return means, stds


# --------------------------------------------------------------------------------------------
# load data
# --------------------------------------------------------------------------------------------
data = sio.loadmat(dataPath + 'turbu_emp.mat')
rspatime_lista = np.squeeze(data['Rspatime'])
rspa_lista = data['Rspa']
Rtime_lista = data['Rtime']
acfspa_lista = data['acfspa']
acftime_lista = data['acftime']
# ------ Surrogate
rspatime_su_lista = np.squeeze(data['Rspatime_su'])
rspa_su_lista = data['Rspa_su']
Rtime_su_lista = data['Rtime_su']
acfspa_su_lista = data['acfspa_su']
acftime_su_lista = data['acftime_su']


# --------------------------------------------------------------------------------------------
# BOXPLOT RSPATIME
# --------------------------------------------------------------------------------------------
BOXRSPA={'Rspatime':rspatime_lista, 'Rspatime_su': rspatime_su_lista}
pValues.plotComparisonAcrossLabels2(BOXRSPA,graphLabel='amplitude turbulence')


# --------------------------------------------------------------------------------------------
# std across time vs space
# --------------------------------------------------------------------------------------------
# -----------------Calcular media y desviación estándar para Rspa y Rspa_su
mean_Rspa, std_Rspa = calculate_stats(rspa_lista)
mean_Rspa_su, std_Rspa_su = calculate_stats(rspa_su_lista)
# ----------------- Plot RSPA
NPARCELLS = 1000
plt.figure(1)
plt.plot(np.arange(0, NPARCELLS), mean_Rspa, '-r')
plt.fill_between(np.arange(0, NPARCELLS), mean_Rspa - std_Rspa, mean_Rspa + std_Rspa,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, NPARCELLS), mean_Rspa_su, '-k')
plt.fill_between(np.arange(0, NPARCELLS), mean_Rspa_su - std_Rspa_su, mean_Rspa_su + std_Rspa_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Rspa')
plt.title('Rspa')
plt.show()

# --------------------------------------------------------------------------------------------
# std across space as a function of time (I cut border effects...thus 100:1100 (time)
# --------------------------------------------------------------------------------------------
startT = 100
endT = 1100
mean_Rtime, std_Rtime = calculate_stats(Rtime_lista[:,startT:endT])
mean_Rtime_su, std_Rtime_su = calculate_stats(Rtime_su_lista[:,startT:endT])
# --------------------------------------------------------------------------------------------
# ----------------- Plot Rtime
plt.figure(2)
plt.plot(np.arange(startT, endT), mean_Rtime, '-r')
plt.fill_between(np.arange(startT, endT), mean_Rtime - std_Rtime, mean_Rtime + std_Rtime,
                 color='r', alpha=0.7)
plt.plot(np.arange(startT, endT), mean_Rtime_su, '-k')
plt.fill_between(np.arange(startT, endT), mean_Rtime_su - std_Rtime_su, mean_Rtime_su + std_Rtime_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Rtime')
plt.title('Rtime')
plt.show()


# --------------------------------------------------------------------------------------------
# autocorr space
# --------------------------------------------------------------------------------------------
mean_acfspa, std_acfspa = calculate_stats(acfspa_lista)
mean_acfspa_su, std_acfspa_su = calculate_stats(acfspa_su_lista)
# --------------------------------------------------------------------------------------------
# ----------------- Plot acfspa
plt.figure(3)
plt.plot(np.arange(0, 101), mean_acfspa, '-r')
plt.fill_between(np.arange(0, 101), mean_acfspa - std_acfspa, mean_acfspa + std_acfspa,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, 101), mean_acfspa_su, '-k')
plt.fill_between(np.arange(0, 101), mean_acfspa_su - std_acfspa_su, mean_acfspa_su + std_acfspa_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('acfspa')
plt.title('acfspa')
plt.show()


# --------------------------------------------------------------------------------------------
# autocorr time
# --------------------------------------------------------------------------------------------
mean_acftime, std_acftime = calculate_stats(acftime_lista)
mean_acftime_su, std_acftime_su = calculate_stats(acftime_su_lista)
# --------------------------------------------------------------------------------------------
# ----------------- Plot acftime
plt.figure(4)
plt.plot(np.arange(0, 101), mean_acftime, '-r')
plt.fill_between(np.arange(0, 101), mean_acftime - std_acftime, mean_acftime + std_acftime,
                 color='r', alpha=0.7)
plt.plot(np.arange(0, 101), mean_acftime_su, '-k')
plt.fill_between(np.arange(0, 101), mean_acftime_su - std_acftime_su, mean_acftime_su + std_acftime_su,
                 color='k', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('acftime')
plt.title('acftime')
plt.show()

print("done")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF