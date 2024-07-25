# --------------------------------------------------------------------------------------
# Full pipeline for loading Wakefulness data
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
import os
import csv
import random
import numpy as np
import scipy.io as sio
import h5py

from WholeBrain.Utils.DataLoaders.baseDataLoader import DataLoader


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from WholeBrain.Utils.DataLoaders.WholeBrainFolder import *
base_folder = WholeBrainFolder + "Data_Raw/Wakefulness/"
# ==========================================================================
# ==========================================================================
# ==========================================================================


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class Wakefulness(DataLoader):
    def __init__(self, path=None,
                 cutTimeSeries=True
                 ):
        if path is not None:
            self.set_basePath(self, path)
        data = sio.loadmat(base_folder + 'DataSleepW_N3.mat')
        self.SC = data['SC']
        self.Num = self.SC.shape[0]  # 90
        self.NumSubj = data['TS_N3'].size  # 15
        self.TS = {}
        self.TS['N3'] = {}
        self.TS['W'] = {}
        for s in range(self.NumSubj):
            self.TS['N3'][('N3', s)] = np.squeeze(data['TS_N3'])[s]
            self.TS['W'][('W', s)] = np.squeeze(data['TS_W'])[s]
        minT_N3 = np.min([self.TS['N3'][('N3', s)].shape[1] for s in range(self.NumSubj)])
        minT_W = np.min([self.TS['W'][('W', s)].shape[1] for s in range(self.NumSubj)])
        if cutTimeSeries:
            for s in range(self.NumSubj):
                self.TS['N3'][('N3', s)] = self.TS['N3'][('N3', s)][:,:minT_N3]
                self.TS['W'][('W', s)] = self.TS['W'][('W', s)][:,:minT_W]
        print(f'loaded, {self.NumSubj} subjects, N={self.N}, minimum length: N3={minT_N3} W={minT_W}')

    def name(self):
        return 'Wakefulness'

    def set_basePath(self, path):
        global WholeBrainFolder, base_folder
        # WholeBrainFolder = path
        base_folder = path

    def TR(self):
        return 2  # Repetition Time (seconds)

    def N(self):
        return self.Num  # 90

    # get_fullGroup_fMRI: convenience method to load all fMRIs for a given subject group
    def get_fullGroup_fMRI(self, group):
        return self.TS[group]

    def get_AvgSC_ctrl(self, normalized=True):
        return self.SC / np.max(self.SC) * 0.2

    def get_groupSubjects(self, group):
        test = self.TS[group].keys()
        return list(test)

    def get_groupLabels(self):
        return ['N3', 'W']

    def get_classification(self):
        classi = {}
        for task in self.get_groupLabels():
            test = self.TS[task].keys()
            for subj in test:
                classi[subj] = subj[0]
        return classi

    def discardSubject(self, subjectID):
        self.TS[subjectID[0]].pop(subjectID)

    def get_SubjectData(self, subjectID):
        ts = self.TS[subjectID[0]][subjectID]
        return {subjectID: {'timeseries': ts}}

    # def get_GlobalData(self):
    #     cog = sio.loadmat(base_folder + 'schaefercog.mat')['SchaeferCOG']
    #     return {'coords': cog} | super().get_GlobalData()


# ================================================================================================================
print('Data loading done!')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF