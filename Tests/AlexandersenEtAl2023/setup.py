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
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# File paths...
# --------------------------------------------------------------------------------------
loadDataPath = './data/'
dataSavePath = '../../Data_Produced/Progression/'
plotsPath = '../../Results/Progression/'

# file paths, where to save dynamics (oscillations) and spreading (heterodimer model) solutions
file_name = 'alzheimers_default'
dyn_save_path = dataSavePath + file_name + '_neural_{}.p'
spread_save_path = dataSavePath + file_name + '_spread_{}.p'


# --------------------------------------------------------------------------------------
# Necessary definitions
# --------------------------------------------------------------------------------------
lobe_names = ['frontal', 'parietal', 'occipital', 'temporal', 'limbic', 'basal-ganglia', 'brain-stem']
# define brain regions (LobeIndex_I.txt)
regions = [[] for _ in range(len(lobe_names))]
with open(loadDataPath + 'LobeIndex_I.txt') as f:
    node = 0
    for line in f:
        lobe = int(float(line.strip()))-1
        regions[lobe].append(node)
        node += 1

print('Setup done.')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF