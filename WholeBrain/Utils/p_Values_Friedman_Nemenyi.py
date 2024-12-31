# --------------------------------------------------------------------------------------
# Full pipeline for Friedman + post hoc Nemenyi test processing and plotting
#
# --------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import WholeBrain.Utils.p_values as p_values

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from WholeBrain.Utils.statannotations_Nemenyi import custom_nemenyi
custom_nemenyi_test = custom_nemenyi()  # "Wilcoxon"


def label_effect_size(W):
    # Kendall’s W uses the Cohen’s interpretation guidelines of
    #     0.1 - < 0.3 (small effect),
    #     0.3 - < 0.5 (moderate effect) and
    #     >= 0.5 (large effect). Confidence intervals are calculated by bootstap.
    if W >= 0.5: return 'large'
    elif W >= 0.3: return 'moderate'
    elif W >= 0.1: return 'small'
    else: return 'none'


def plotComparisonAcrossLabels2Ax(ax, tests, columnLables=None, graphLabel='', pairs=None):
    if columnLables is None:
        columnLables = tests.keys()
    print(f'test for {columnLables}')
    test_list = [tests[t] for t in tests]
    res = friedmanchisquare(*test_list)
    # Compute the effect size estimate (referred to as w) for Friedman test: W = X2/N(K-1);
    # where W is the Kendall's W value;
    #       X2 is the Friedman test statistic value;
    #       N is the sample size.
    #       k is the number of measurements per subject.
    N = len(test_list[0])
    K = 3
    W = res[0] / (N * (K - 1))
    print(f'\nFriedman chisquare-test: pvalue = {res[1]} stats = {res[0]}')
    print(f'Effect size = {W} ({label_effect_size(W)})')
    res2 = sp.posthoc_nemenyi_friedman(np.array(test_list).T)
    print(f'\nNemenyi post hoc test: \n{res2}')
    p_values.plotComparisonAcrossLabels2Ax(ax, tests,
                                           custom_test=custom_nemenyi_test,
                                           columnLables=columnLables,
                                           graphLabel=graphLabel,
                                           pairs=pairs)


def plotComparisonAcrossLabels2(tests, columnLables=None, graphLabel='', pairs=None):
    fig, ax = plt.subplots()
    plotComparisonAcrossLabels2Ax(ax, tests,
                                  columnLables=columnLables, graphLabel=graphLabel, pairs=pairs)
    plt.show()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------