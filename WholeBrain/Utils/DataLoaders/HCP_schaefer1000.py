# --------------------------------------------------------------------------------------
# Full pipeline for applying Leading Eigenvector Dynamics Analysis (LEiDA) to AD data
# using pyLEiDA: https://github.com/PSYMARKER/leida-python
#
# By Gustavo Patow
#
# note: start by configuring this!!!
# --------------------------------------------------------------------------------------
import os
import csv
import random
import numpy as np
import hdf5storage as sio
import h5py

from WholeBrain.Utils.DataLoaders.baseDataLoader import DataLoader


# ==========================================================================
# Important config options: filenames
# ==========================================================================
WholeBrainFolder = "L:/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WholeBrain/"
# WholeBrainFolder = "/Users/dagush/Dpt. IMAE Dropbox/Gustavo Patow/SRC/WholeBrain/"
base_folder = WholeBrainFolder + "Data_Raw/HCP/"
# ==========================================================================
# ==========================================================================
# ==========================================================================

maxSubjects = 1003
tasks = ['REST1']  # ['REST1', 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
minLength = 175


# --------------------------------------------------------------------------
# functions to select which subjects to process
# --------------------------------------------------------------------------
# _numSampleSubjects = 20  # 20 for exploring the data
# save_path = WholeBrainFolder + 'WIP/HCP-LEiDA/Data_Produced/'
# selectedSubjectsFile = save_path + f'selected_{_numSampleSubjects}.txt'
fMRI_path = base_folder + 'hcp_{}_LR_schaefer1000.mat'
# SC_path = base_folder + 'SC_dbs80HARDIFULL.mat'
schaeferCOG = base_folder + 'schaefercog.mat'


# --------------------------------------------------------------------------
# functions to load fMRI data for certain subjects
# --------------------------------------------------------------------------
def read_matlab_h5py(filename, task, selectedIDs):
    with h5py.File(filename, "r") as h5File:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]
        # get the object type for a_group_key: usually group or dataset
        # print(type(f['subjects_idxs']))
        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array

        all_fMRI = {}
        excluded = []
        subjects = list(h5File['subject'])
        for pos, subj in enumerate(subjects):
            # print(f'reading subject {pos}')
            group = h5File[subj[0]]
            try:
                dbs80ts = np.array(group['schaeferts'])
                if dbs80ts.shape[0] < minLength:  # Some individuals have too short time series
                    print(f'should ignore register {subj} at {(pos,task)}: length {dbs80ts.shape[0]}')
                all_fMRI[(pos,task)] = dbs80ts.T
            except:
                print(f'ignoring register {subj} at {pos}')
                excluded.append(pos)

    return all_fMRI, excluded


def loadSubjectsData(fMRI_path, task, selectedIDs):
    print(f'Loading {fMRI_path}')
    fMRIs, excluded = read_matlab_h5py(fMRI_path, task, selectedIDs)   # ignore the excluded list
    return fMRIs, excluded


# --------------------------------------------------------------------------
# ---------------- load a previously saved list
# --------------------------------------------------------------------------
def loadSubjectList(path):
    subjects = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            subjects.append(int(row[0]))
    return subjects


# save a freshly created list
def saveSelectedSubjects(path, subj):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in subj:
            writer.writerow([s])


# # fix subset of subjects to sample
# def selectSubjectSubset(selectedSubjectsF, numSampleSubj, forceRecompute=False):
#     if not os.path.isfile(selectedSubjectsF) or forceRecompute:  # if we did not already select a list...
#         allExcl = set()
#         for task in tasks:
#             allExcl |= set(excluded[task])
#         listIDs = random.sample(range(0, maxSubjects), numSampleSubj)
#         while allExcl & set(listIDs):
#             listIDs = random.sample(range(0, maxSubjects), numSampleSubj)
#         saveSelectedSubjects(selectedSubjectsF, listIDs)
#     else:  # if we did, load it!
#         listIDs = loadSubjectList(selectedSubjectsF)
#     # ---------------- OK, let's proceed
#     return listIDs



# --------------------------------------------------------------------------
# ---------------- load and filter data (some entries are "broken")
# --------------------------------------------------------------------------
timeseries = {}
excluded = {}


# def filterSubjectsData(task, listRejectedIDs):
#     global timeseries
#     for i in listRejectedIDs:
#         id = (i, task)
#         if id in timeseries[task]:
#             timeseries[task].pop(id)


def loadFilteredData(chosenDatasets=tasks,  # forceUniqueSet=False
                     ):
    global timeseries, excluded
    allSubj = set([s for s in range(maxSubjects)])

    for task in chosenDatasets:
        print(f'----------- Checking: {task} --------------')
        fMRI_task_path = fMRI_path.format(task)
        timeseries[task], excluded[task] = loadSubjectsData(fMRI_task_path, task, allSubj)
        print(f'------ Excluded: {len(excluded)} for {task}------')

    # if forceUniqueSet:  # force same subjects across all the selected datasets
    #     # firstm, let;'s unify the excluded sets, which will result in 971 subjects out of the 1003 originals
    #     allExcl = set()
    #     for task in chosenDatasets:
    #         allExcl |= set(excluded[task])
    #     # now we have all excluded subjects selected, let's apply the filtering...
    #     for task in chosenDatasets:
    #         filterSubjectsData(task, allExcl)


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class HCP(DataLoader):
    def __init__(self, path=None, # chosenDatasets=tasks, forceUniqueSet=False, numSampleSubjects=None, correcSCMatrix=True
                 ):
        if path is not None:
            self.set_basePath(self, path)
        loadFilteredData()  # chosenDatasets=chosenDatasets, forceUniqueSet=forceUniqueSet)
        # if numSampleSubjects is not None:
        #     selected = selectSubjectSubset(selectedSubjectsFile, numSampleSubjects)
        #     for task in chosenDatasets:
        #         listRejectedIDs = set([s for s in range(maxSubjects)]) - set(selected)
        #         filterSubjectsData(task, listRejectedIDs)
        # self.correcSCMatrix = correcSCMatrix

    def set_basePath(self, path):
        global WholeBrainFolder, base_folder
        WholeBrainFolder = path
        base_folder = WholeBrainFolder + "Data_Raw/HCP/"

    def TR(self):
        return 0.72  # Repetition Time (seconds)

    def N(self):
        return 1000

    # get_fullGroup_fMRI: convenience method to load all fMRIs for a given subject group
    def get_fullGroup_fMRI(self, group):
        return timeseries[group]

    def get_AvgSC_ctrl(self, normalized=True):
        # I do not have this information...
        raise Exception("I do not have the SC for this dataset")

    def get_groupSubjects(self, group):
        test = timeseries[group].keys()
        return list(test)

    def get_groupLabels(self):
        return tasks

    def get_classification(self):
        classi = {}
        for task in tasks:
            test = timeseries[task].keys()
            for subj in test:
                classi[subj] = subj[1]
        return classi

    def discardSubject(self, subjectID):
        timeseries[subjectID[1]].pop(subjectID)

    def get_SubjectData(self, subjectID):
        ts = timeseries[subjectID[1]][subjectID]
        return {subjectID: {'timeseries': ts}}

    def get_GlobalData(self):
        cog = sio.loadmat(base_folder + 'schaefercog.mat')['SchaeferCOG']
        return {'coords': cog}


# ================================================================================================================
print('Data loading done!')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
