# =====================================================================================
# Methods to plot a few properties SC matrices
# =====================================================================================
import numpy as np
import matplotlib.pyplot as plt


def plotSCHistogram(ax, SC, subjectName):
    # plt.rcParams["figure.figsize"] = (7,5)
    # plt.rcParams["figure.dpi"] = 300
    # plt.figure()  #num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    bins = 50 #'auto'
    n, bins, patches = ax.hist(SC.flatten(), bins=bins, color='#0504aa', alpha=0.7, histtype='step')  #, rwidth=0.85)
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel('SC weights')
    ax.set_ylabel('Counts')
    ax.set_title("SC histogram ({}: {})".format(subjectName, SC.shape), fontweight="bold", fontsize="18")
    # plt.savefig("./Results/Abeta/"+subject+".png", dpi=200)
    # plt.close()


def plotSC(ax, SC, subjectName):
    ax.imshow(np.asarray(SC))
    ax.set_xlabel("Regions")
    ax.set_ylabel("Regions")
    ax.set_title("Subject {}".format(subjectName))
    print("Scale({}): Max={}, Min={}".format(subjectName, np.max(SC), np.min(SC)))


def plotSC_and_Histogram(subjectName, SCnorm, plotColorBar = True):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 2)
    ax1 = fig.add_subplot(grid[0,0])
    plotSC(ax1, SCnorm, subjectName)
    if plotColorBar:
        img = ax1.get_images()[0]
        fig.colorbar(img)
    ax2 = fig.add_subplot(grid[0,1])
    plotSCHistogram(ax2, SCnorm, subjectName)
    plt.suptitle("Structural Connectivity ({})".format(subjectName), fontweight="bold", fontsize="18", y=1.05)
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    # img = ax1.get_images()[0]
    # fig.colorbar(img, cax=cbar_ax)
    plt.show()


def justPlotSC(subjectName, SCnorm, plottingFunction):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0,0])
    plottingFunction(ax, SCnorm, subjectName)
    plt.show()

# --eof
