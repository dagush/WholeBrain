# --------------------------------------------------------------------------------------
import hdf5storage as sio
import dataLoader
import Utils.Plotting.plotSC as plotSC

base_folder = '../../Data_Produced/AD_ABeta_and_Tau'

if __name__ == '__main__':
    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    AvgHC = sio.loadmat(base_folder + '/AvgHC_SC.mat')['SC']
    dataLoader.analyzeMatrix("AvgHC norm", AvgHC)
    print("# of elements in AVG connectome: {}".format(AvgHC.shape))

    plotSC.plotFancyMatrix(AvgHC, fontSize=15, cmap='plasma')
    plotSC.plotSCMatrixAsFancyGraph(AvgHC)
