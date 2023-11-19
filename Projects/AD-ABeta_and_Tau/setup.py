# --------------------------------------------------------------------------------------
# Setup file for processing AD, MCI and HC subjects (MMSE classification)
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# By Gustavo Patow
#
# Pre-requisites:
#   This is the first file to configure in any project!
# --------------------------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import dataLoader

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
# import WholeBrain.Models.DynamicMeanField as DMF
import DMF_AD
neuronalModel = DMF_AD
import Models.Couplings as Couplings

import Integrators.EulerMaruyama as scheme
scheme.neuronalModel = neuronalModel
import Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = neuronalModel
integrator.verbose = False

import Utils.BOLD.BOLDHemModel_Stephan2007 as Stephan2007
import Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007
simulateBOLD.TR = 3.
# ============== chose a FIC mechanism
import Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
import Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
BalanceFIC.balancingMechanism = Deco2014Mechanism  # default behaviour for this project

import WholeBrain.Observables.BOLDFilters as BOLDFilters
# NARROW LOW BANDPASS
BOLDFilters.flp = .02      # lowpass frequency of filter
BOLDFilters.fhi = 0.1      # highpass
BOLDFilters.TR = 3.

import WholeBrain.Optimizers.ParmSweep as ParmSeep

ParmSeep.simulateBOLD = simulateBOLD
ParmSeep.integrator = integrator
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------

# ==========================================================================
# Important config options: filenames
# ==========================================================================
base_folder = "../../Data_Raw/from_ADNI"
save_folder = "../../Data_Produced/AD_ABeta_and_Tau"

# --------------------------------------------------
# Classify subject information into {HC, MCI, AD}
# --------------------------------------------------
subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
classification = dataLoader.checkClassifications(subjects)
HCSubjects = [s for s in classification if classification[s] == 'HC']
ADSubjects = [s for s in classification if classification[s] == 'AD']
MCISubjects = [s for s in classification if classification[s] == 'MCI']
print(f"We have {len(HCSubjects)} HC, {len(MCISubjects)} MCI and {len(ADSubjects)} AD \n")
# print("HCSubjects:", HCSubjects)
# print("ADSubjects", ADSubjects)
# print("MCISubjects", MCISubjects)

dataSetLabels = ['HC', 'MCI', 'AD']

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
