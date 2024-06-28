# =====================================================================================
# Methods to plot a few properties FCD matrices
# =====================================================================================
import hdf5storage as sio
import WholeBrain.Observables.phFCD as phFCD
from Observables import BOLDFilters

BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


def plotPhIntMatrSlice(PhIntMat, t,
                       axisName="Regions", matrixName="Phase Functional Connectivity Dynamics (phFCD)", showAxis='on'):
    import plotSC
    plotSC.plotFancyMatrix(PhIntMat[..., t], axisName=axisName, matrixName=matrixName, showAxis=showAxis)


def plot_from_fMRI(ts,
                  axisName="Time", matrixName="FC dynamics (FCD)", showAxis='on'):
    print(f'plotting {phFCD.name} from fMRI...')
    # ================ Change mode at phFCD to save the resulting matrix
    phFCD.saveMatrix = True
    save_file = "./Data_Produced/"+phFCD.name+"_from_fMRI.mat"
    phFCD.save_file = save_file
    M = phFCD.from_fMRI(ts)
    phIntMatr = sio.loadmat(save_file)
    import Utils.Plotting.plotSC as plotSC
    plotSC.plotFancyMatrix(phIntMatr[phFCD.name], axisName=axisName, matrixName=matrixName, showAxis=showAxis)
    # ============= Restore state at phFCD, do not keep saving anything!
    phFCD.saveMatrix = False


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================eof
