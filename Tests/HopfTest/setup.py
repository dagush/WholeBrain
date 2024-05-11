# --------------------------------------------------------------------------------------
#  Hopf code: setup
#  By gustavo Patow
#
# The very first thing to do is to configure this file
# --------------------------------------------------------------------------------------
import random
import os, csv
import numpy as np
import scipy.io as sio
import h5py


# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import WholeBrain.Models.supHopf as Hopf
Hopf.initialValueX = Hopf.initialValueY = 0.1

import WholeBrain.Integrators.EulerMaruyama as scheme
scheme.neuronalModel = Hopf
import WholeBrain.Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = Hopf
integrator.verbose = False

import WholeBrain.Utils.simulate_SimOnly as simulateBOLD
simulateBOLD.warmUp = True
simulateBOLD.warmUpFactor = 606./2000.
simulateBOLD.integrator = integrator

import WholeBrain.Observables.phFCD as phFCD
import WholeBrain.Optimizers.ParmSweep as ParmSweep
ParmSweep.simulateBOLD = simulateBOLD
ParmSweep.integrator = integrator
ParmSweep.verbose = True

import WholeBrain.Observables.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.Observables.BOLDFilters as BOLDFilters
# NARROW LOW BANDPASS
BOLDFilters.flp = 0.008      # lowpass frequency of filter
BOLDFilters.fhi = 0.08       # highpass
BOLDFilters.TR = 2.
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# functions to load fMRI data for certain subjects
# --------------------------------------------------------------------------
def read_matlab_h5py(filename):
    with h5py.File(filename, "r") as f:
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
        subjects = list(f['subject'])
        for pos, subj in enumerate(subjects):
            print(f'reading subject {pos}')
            group = f[subj[0]]
            dbs80ts = np.array(group['dbs80ts'])
            all_fMRI[pos] = dbs80ts.T

    return all_fMRI


def loadSubjectList(path):
    subjects = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            subjects.append(row[0])
    return subjects


def saveSelectedSubjcets(path, subj):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in subj:
            writer.writerow([s])


def loadSubjectsData(fMRI_path, numSampleSubjects):
    fMRIs = read_matlab_h5py(fMRI_path)
    # ---------------- fix subset of subjects to sample
    if not os.path.isfile(selectedSubjectsFile):  # if we did not already select a list...
        listIDs = random.sample(range(0, len(fMRIs)), numSampleSubjects)
        saveSelectedSubjcets(selectedSubjectsFile, listIDs)
    else:  # if we did, load it!
        listIDs = loadSubjectList(selectedSubjectsFile)
    # ---------------- OK, let's proceed
    nNodes, Tmax = fMRIs[next(iter(fMRIs))].shape
    res = np.zeros((numSampleSubjects, nNodes, Tmax))
    for pos, s in enumerate(listIDs):
        res[pos] = fMRIs[s]
    return res, listIDs


# --------------------------------------------------------------------------
# Paths and subject selection
# --------------------------------------------------------------------------
numSampleSubjects = 20

base_path = '../../Data_Raw/HCP/DataHCP80/'
fMRI_rest_path = base_path + 'hcp1003_REST1_LR_dbs80.mat'
SC_path = base_path + 'SC_dbs80HARDIFULL.mat'
save_path = '../../Data_Produced/Tests/TestHopf/'
selectedSubjectsFile = save_path + f'selected_{numSampleSubjects}.txt'

timeseries, listIDs = loadSubjectsData(fMRI_rest_path, numSampleSubjects)
all_fMRI = {s: d for s,d in enumerate(timeseries)}
numSubj, nNodes, Tmax = timeseries.shape  # actually, 80, 1200

mat0 = sio.loadmat(SC_path)['SC_dbs80FULL']
SCnorm = 0.2 * mat0 / mat0.max()
Hopf.setParms({'SC': SCnorm})
Hopf.couplingOp.setParms(SCnorm)

# ------------------------------------------------
# Configure and compute Simulation
# ------------------------------------------------
# distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}
selectedObservable = 'phFCD'
distanceSettings = {'phFCD': (phFCD, True)}

simulateBOLD.TR = BOLDFilters.TR   # Recording interval: 1 sample every X seconds
# ------------------------------------------------
# The Hopf model works in seconds, not in milliseconds, so
# all integration parms should be adjusted accordingly...
simulateBOLD.dt = 0.1 * simulateBOLD.TR / 2.
simulateBOLD.Tmax = Tmax  # This is the length, in seconds
simulateBOLD.dtt = simulateBOLD.TR  # We are NOT using milliseconds
simulateBOLD.t_min = 10 * simulateBOLD.TR
# simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
# simulateBOLD.warmUpFactor = 6.
simulateBOLD.Tmaxneuronal = (Tmax-1) * simulateBOLD.TR + 30
integrator.ds = simulateBOLD.TR  # record every TR seconds

base_a_value = -0.02
Hopf.setParms({'a': base_a_value})
# Hopf.beta = 0.01
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR=BOLDFilters.TR)  # baseline_group[0].reshape((1,52,193))
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
# Hopf.omega = repmat(2*pi*f_diff',1,2);     # f_diff is the frequency power
Hopf.setParms({'omega': 2 * np.pi * f_diff})

print("Hopf Setup done!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF