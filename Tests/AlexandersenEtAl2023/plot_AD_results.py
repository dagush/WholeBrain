# --------------------------------------------------------------------------------------
# Simulation of Alzheimer's disease progression
#
# By Christoffer Alexandersen
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# Requirements:
#   To run this, first you need to perform the simulations with alzheimer_simulation.py
#
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
import pickle

from setup import *
from Progression.plotting_functions import *

# --------------------------------------------------------------------------------------
# Some initial settings
# --------------------------------------------------------------------------------------
t_stamps = np.linspace(0,35,11)
bands = [[0.008,12]]  # frequency bands to analyze

# --------------------------------------------------------------------------------------
# plot settings
# --------------------------------------------------------------------------------------
lobe_file = loadDataPath + 'LobeIndex.csv';
xlimit = t_stamps[-1];
wiggle = 0.1  # wiggles trials on same x tick
plt.style.use('seaborn-muted')
colours = sns.color_palette('hls', len(lobe_names)+4)

# --------------------------------------------------------------------------------------
# LOAD SOLUTIONS
# dump
# --------------------------------------------------------------------------------------
gamma = 0.0
print('\nLoading solutions...')
dyn_sols = pickle.load(open(dyn_save_path.format(f'gamme={gamma}'), "rb"))
spread_sol = pickle.load(open(spread_save_path.format(f'gamme={gamma}'), "rb"))
print('Done.')

# --------------------------------------------------------------------------------------
# PLOT
# plot spreading
# --------------------------------------------------------------------------------------
t_spread = spread_sol['disc_t']
print('\nPlotting...')
figs, axs = plot_spreading(spread_sol, colours, lobe_names, xlimit=xlimit, regions=regions, averages=True)

# --------------------------------------------------------------------------------------
# analyze dynamics and plot
bandpowers, freq_peaks = spectral_properties(dyn_sols, bands, 0, freq_tol=0, relative=False)

sns.set_context(font_scale=2, rc={"axes.labelsize":18,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":8})
figs_PSD, figs_peaks = plot_spectral_properties(t_spread, bandpowers, freq_peaks, bands, wiggle, '',
                                                lobe_names[:-1], colours, regions=regions[:-1],
                                                only_average=False, n_ticks=6)
print('Done.')

# --------------------------------------------------------------------------------------
# save figues
# --------------------------------------------------------------------------------------
figs[0].savefig(plotsPath + 'ab_damage.pdf', dpi=300)
figs[1].savefig(plotsPath + 'protein_damage.pdf', dpi=300)
figs[2].savefig(plotsPath + 'toxic_concentration.pdf', dpi=300)
figs[3].savefig(plotsPath + 'weight_damage.pdf', dpi=300)
figs[4].savefig(plotsPath + 'healthy_concentration.pdf', dpi=300)
figs_PSD[0].savefig(plotsPath + f'oscillatory_power_gamma={gamma}.pdf', dpi=300)
figs_peaks[0].savefig(plotsPath + 'oscillatory_frequency.pdf', dpi=300)

# --------------------------------------------------------------------------------------
# show figures
# --------------------------------------------------------------------------------------
plt.show()

# WE'RE DONE
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
