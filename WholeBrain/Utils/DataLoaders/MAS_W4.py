# --------------------------------------------------------------------------------------
# Full code for loading the HCB data in the AAL parcellation 88/512
# Subjects: HC , MCI , AD - RoIs: 88/512 - TR = 3 - timepoints: 197
# Info for each subject: timeseries
#
# Parcellated by Xenia Kobeleva
#
# Code by Gustavo Patow
# Note: We have 72 subjects and 52 areas
# --------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

import scipy.io as hdf

from WholeBrain.Utils.DataLoaders.baseDataLoader import DataLoader
# import DataLoaders.Parcellations.Schaefer2018 as Schaefer2018


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from WholeBrain.Utils.DataLoaders.WorkBrainFolder import *


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class MAS_W4(DataLoader):
    def __init__(self, path=None,
                 AALSize=512,  # by default, let's use the AAL 512 parcellation
                 ):
        self.groups = ['normal', 'MCI', 'AD']
        self.AALSize = AALSize
        if path is not None:
            self.set_basePath(path)
        else:
            self.set_basePath(WorkBrainDataFolder)
        self.timeseries = {}
        self.__loadAllData()

    def __loadAllData(self):
        xls = pd.read_excel(self.classific_path)
        split = xls.to_dict('split')['data']
        self.classi = {id: val for [id, val] in split}
        # ------------------ check for subjects with missing Dx info
        subjs = [x[0].split('/')[-1] for x in os.walk(self.fMRI_base_path)]
        del subjs[0]  # remove the root directory entry, not useful.
        diff = [val for val in subjs if val not in self.classi]
        print(f'Missing info for subjects: {diff}')
        # ------------------ end check
        for s in self.classi:
            print(f'Loading: {s}')
            fMRI_file_path = self.fMRI_path.format(s,s)
            self.timeseries[s] = hdf.loadmat(fMRI_file_path)['ROISignals'].T
        print('Donde loading MAS-W4 data')

    def name(self):
        return 'MAS_W4'

    def set_basePath(self, path):
        base_folder = path + "MAS-W4/"
        self.classific_path = base_folder + 'Dx-List-All subjects-w4.xlsx'
        self.fMRI_base_path = base_folder + ('512Results-Functional' if self.AALSize == 512 else 'AAL')
        self.fMRI_path = self.fMRI_base_path + '/{}/' + 'ROISignals_{}.mat'

    def TR(self):
        return 2  # Repetition Time (seconds)

    def N(self):
        first = list(self.timeseries.keys())[0]
        return self.timeseries[first].shape[0]

    def get_classification(self):
        return self.classi

    def get_subjectData(self, subjectID):
        ts = self.timeseries[subjectID]
        res = {subjectID: {'timeseries': ts}}
        return res

    def get_groupLabels(self):
        return self.groups

# ================================================================================================================
print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = MAS_W4(AALSize=88)  # 88 512
    sujes = DL.get_classification()
    N = DL.N()
    print(f'Classification: {sujes}')
    print(f'Group labels: {DL.get_groupLabels()}')
    gMCI = DL.get_groupSubjects('MCI')
    s1 = DL.get_subjectData('0720A')
    s2 = DL.get_subjectData('0302A')
    avgSC = DL.get_AvgSC_ctrl()
    print('done! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF