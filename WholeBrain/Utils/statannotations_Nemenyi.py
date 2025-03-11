import numpy as np
import scikit_posthocs as sp
from statannotations.stats.StatTest import StatTest

def nemenyi_test(group_data1, group_data2, **stats_params):
    data = np.array([group_data1, group_data2])
    res = sp.posthoc_nemenyi_friedman(data.T)
    return np.nan, res[0][1]

def custom_nemenyi():
    custom_long_name = 'Nemenyi post hoc test'
    custom_short_name = 'Nemenyi'
    custom_func = nemenyi_test
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
    return custom_test

