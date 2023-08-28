# --------------------------------------------------------------------------------------
# PERMUTATION_HTEST2_NP - A "non-parametric" two-sample hypotesis test that, instead of
# relying on the test-type standard distribution, uses permutations of group labels to
# estimate the null distribution. The null distribution is computed
# independently for each data point (= row), i.e. we do not assume the same
# distribution for each datapoint. However, we do assume that the data
# points are comparable (e.g. they correspond to the same location
# collected across all subjects)
#
#  Henrique Fernandes 2014
#  Adapted from: Enrico Glerean 2013
#  Translated to Python by Gustavo Patow 2023
#
# --------------------------------------------------------------------------------------
import numpy as np
from scipy import stats


def permutation_htest2_np(data1, data2, niter, htest='ttest2'):
    # USAGE:
    #    stats = bramila_ttest2_np(data,design,niter)
    # INPUT:
    #    data1,2 - a matrix where each column is a subject and each row is a
    #              data-point for example a voxel intensity in fMRI, a node level
    #              value in a network, etc. NaN values will be ignored.
    #    niter   - number of permutations (recommended 5000)
    #    htest   - hypothesis test used to compare populations. The script is
    #              prepared to run the ttest2, kstest2, and ranksum tests.
    #
    # OUTPUT:
    #    result is a dict with the following subfields:
    #        pvals - p-values for each datapoint; it returns in order the p-values
    #                for the right tail and for the left tail
    #        tvals - test statistic values for datapoint, positive tvals mean
    #                group 1 > group 2
    #
    #  Notes: the null distribution is estimated using the matlab function
    #  ksdensity by interpolating the permuted data. The distribution is
    #  estimated over 200 points if niter<=5000, otherwise it is estimated over
    #  round(200*niter/5000) points, for greater precision.

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # INPUT VALIDATION
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # Nsubj = size(data, 2) # number of subjects
    # if (size(design, 2) != Nsubj):
    #     raise Exception('Mismatched number of subjects: the number of columns of data variable  should match the number of columns of the design variable.')
    # if (size(design, 1) != 1):
    #     raise Exception('The design variable should only contain 1 row')
    #
    # g1 = find(design == 1)
    # g2 = find(design == 2)
    # if ((length(g1) + length(g2)) != Nsubj):
    #     raise Exception('The design variable should only contain numbers 1 and 2.')

    if niter <= 0:
        print('The variable niter should be a positive integer, function will continue assuming niter=5000.')
        niter = 5000

    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    # HYPOTHESIS TESTING(for each row / area)
    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    # stats.tvals = tt_np(data, g1, g2); % similar to ttest2

    result = {}

    NC = 1 if data1.ndim == 1 else data1.shape[1]  # number of comparisons
    tvals = np.zeros(NC)
    diffs = np.zeros(NC)
    # means = np.zeros(NC)

    # if htest == 'ttest':
    #     # - the population means are not equal. (alternative hypothesis)
    #     # - the two groups are derived from normal distributions with unknown and unequal variances.
    #     for t = 1:NC:
    #         [H, P, CI, STATS] = ttest(data(t, g1)',data(t,g2)', pthr, 'both')
    #         tvals(t,:) = STATS.tstat
    #         diffs(t,:) = mean(data(t, g1)) - mean(data(t, g2))
    if htest == 'ttest2':
        # - the population means are not equal. (alternative hypothesis)
        # - the two groups are derived from normal distributions with unknown and unequal variances.
        if NC == 1:
            statstt = stats.ttest_ind(data1, data2, equal_var=False, alternative='two-sided')
            tvals[0] = statstt.statistic
            diffs[0] = np.mean(data1) - np.mean(data2)
        else:
            for t in range(NC):
                statstt = stats.ttest_ind(data1[t,:],data2[t,:], equal_var=False, alternative='two-sided')
                tvals[t] = statstt.statistic
                diffs[t] = np.mean(data1) - np.mean(data2)
    # case 'kstest'
    #     for t=1:NC
    #         [H,P,STATS]=kstest2(data(t,g1)',data(t,g2)',pthr);
    #         tvals(t,:)=STATS;
    #         diffs(t,:)      = mean(data(t,g1))-mean(data(t,g2));
    #     end
    # case 'ranksum'
    #     for t=1:NC
    #         [P,H,STATS]=ranksum(data(t,g1)',data(t,g2)','alpha',pthr);
    #         tvals(t,:)=STATS.zval;
    #         diffs(t,:)      = mean(data(t,g1))-mean(data(t,g2));
    #     end
    else:
        raise Exception('\n-------------------------------\n\nHypothesis test %s not recognized. \n\n-------------------------------\n',htest)

    result['tvals'] = tvals
    # % tvals(isnan(tvals)) = 0; % or tvals(tvals ~ = tvals) = 0
    result['diffs'] = diffs

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % PERMUTATION TESTING (for each row/area)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % outptus the pval (from the computed null distribution using permutation
    # % testing) given the tstat previously calculated.
    # % each comparison is treated independently

    pvals = np.zeros((NC,2))

    for n in range(NC):
        if np.median(data1) != 0 or np.median(data2) != 0:  # Exclude tests where all (tstat=NaN) or most of the population (median=0) as a null value.
            pvals[n] = test_np_pval(data1,data2,niter,tvals[n])
        else:
            pvals[n] = [np.NaN, np.NaN]

    result['pvals'] = pvals

    return result


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NESTED FUNCTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_np_pval(data1,data2,niter,tval):
    alldata = np.concatenate([data1, data2])
    a_n = data1.shape[0]

    outiter=np.zeros(niter)
    a_n=np.size(data1)
    for iter in range(niter):
        np.random.shuffle(alldata)
        # one could add a test to see that they are indeed permuted
        temp1 = alldata[:a_n]
        temp2 = alldata[a_n:]
        statsRes = stats.ttest_ind(temp1, temp2, equal_var=False, alternative='two-sided')
        outiter[iter] = statsRes.statistic

    NCDF = 200
    if niter > 5000:
        NCDF = np.int(np.round(200.*niter/5000))
    # estimated cumulative distribution function
    # [fi xi]=ksdensity(outiter,'function','cdf','npoints',NCDF)
    kde = stats.gaussian_kde(outiter)
    xi = np.linspace(outiter.min(), outiter.max(), NCDF)
    fi = kde(xi)


    # trick to avoid NaNs, we approximate the domain of the CDF between
    # -Inf and Inf using the atanh function and the eps matlab precision variable

    eps = np.spacing(1)
    pval_left = np.interp(tval,
                          np.concatenate([[np.arctanh(-1+eps)], xi, [np.arctanh(1-eps)]]),
                          np.concatenate([[0], fi, [1]]))  # G1 > G2
    pval_right = 1 - pval_left  # G1 < G2
    pval = [pval_right, pval_left]

    return pval

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF