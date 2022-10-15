# --------------------------------------------------------------------------------------
# Full pipeline for wilcoxon test processing
#
# --------------------------------------------------------------------------------------
import os
import itertools

import numpy as np
import scipy.stats as stats


fontSize = 10


def plotMeanVars(ax, data, pos, title):
    points = [data[d] for d in data]
    positions = [pos[d] for d in data]
    ax.boxplot(points, positions=positions, labels=data.keys())  # notch='True', patch_artist=True,

    for d in data:
        ax.plot(pos[d]*np.ones(len(data[d])), np.array(data[d]).reshape(len(data[d])),
                'r.', alpha=0.2)
    ax.set_title(title)


# h = fontSize / 10
barHeight = fontSize / 2.
def plotWilcoxonTest(ax, tests, pos, plotOrder = None, col='grey'):
    def classify(test):
        if test < 0.001:
            text = f'*** p={test:.4f}'
        elif test < 0.01:
            text = f'** p={test:.3f}'
        elif test < 0.05:
            text = f'* p={test:.3f}'
        else:
            text = f'p={test:.3f}'
        return text

    def plotBar(x1, x2, h, text):  # (x1, x2, y, h, text):
        ylim = ax.get_ylim()
        y = ylim[1] + h
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col)
        ax.set_ylim([ylim[0], y + 5 * h])

    if plotOrder is None: plotOrder = tests
    # * statistical tests. From https://towardsdatascience.com/beautiful-boxplots-with-statistical-significance-annotation-e1b314927fc5
    # x1, x2 = -0.20, 0.20
    # y, h, col = df_long[df_long.Feature == feature][“Value”].max()+1, 2, ‘k’
    # axes[idx].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # axes[idx].text((x1+x2)*.5, y+h, “statistically significant”, ha=’center’, va=’bottom’, color=col)
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0]) / 50
    for order, pair in enumerate(plotOrder):
        labels = pair.split('_')
        plotBar(pos[labels[0]], pos[labels[1]], h, classify(tests[pair]))  # ylim[1] + delta * order
    print()


def computeWilcoxonTests(data):
    tests = {}
    for pair in itertools.combinations(data, r=2):
        if pair[0] != pair[1]:
            testName = pair[0]+'_'+pair[1]
            tests[testName] = stats.mannwhitneyu(data[pair[0]], data[pair[1]]).pvalue
            print(f'test[{testName}] = {tests[testName]}')
    return tests

# ----------------------------------------------------------------------------
# Some convenience WholeBrain
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Plotting func.
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

posA = 1; posB = 2; posC = 3; posD = 4

# Generates a boxPlot and the p-values for 3 different labels
def plotComparisonAcrossLabels(dataA, dataB, dataC, labels, titleLabel='test', yLimits = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    points = {labels[0]: dataA, labels[1]: dataB, labels[2]: dataC}
    positions = {labels[0]: posA, labels[1]: posB, labels[2]: posC}
    if yLimits is not None:
        ax.set_ylim(yLimits)
    plotMeanVars(ax, points, positions, title=titleLabel)  # f'Parm Comparison ({titleLabel})'
    test = computeWilcoxonTests(points)
    plotWilcoxonTest(ax, test, positions, plotOrder=[labels[0]+'_'+labels[1],
                                                     labels[1]+'_'+labels[2],
                                                     labels[0]+'_'+labels[2],
                                                    ])
    ax.set_ylabel("phFCD")
    plt.show()


# Same as previous one, but with 4 labels. Too lazy to refactor this... ;-)
def plotValuesComparisonAcross4Labels(dataA, dataB, dataC, dataD, labels, titleLabel='test', yLimits = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    points = {labels[0]: dataA, labels[1]: dataB, labels[2]: dataC, labels[3]: dataD}
    positions = {labels[0]: posA, labels[1]: posB, labels[2]: posC, labels[3]: posD}
    if yLimits is not None:
        ax.set_ylim(yLimits)
    plotMeanVars(ax, points, positions, title=titleLabel)  # f'Parm Comparison ({titleLabel})'
    test = computeWilcoxonTests(points)
    plotWilcoxonTest(ax, test, positions, plotOrder=[labels[0]+'_'+labels[1],
                                                     labels[1]+'_'+labels[2],
                                                     labels[0]+'_'+labels[2],
                                                     labels[2]+'_'+labels[3],
                                                     labels[1]+'_'+labels[3],
                                                     labels[0]+'_'+labels[3],
                                                    ])
    ax.set_ylabel("phFCD")
    plt.show()


def findMinMaxSpan(a,b):
    max = -np.inf; posMax = 0
    min = np.inf; posMin = 0
    for pos, (va, vb) in enumerate(zip(a,b)):
        span = np.abs(va-vb)
        if span > max:
            max = span
            posMax = pos
        if span < min:
            min = span
            posMin = pos
    return min, posMin, max, posMax


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------EOF
