# --------------------------------------------------------------------------------------
# Evidence for local effects of Abeta to inhibitory interneurons, e.g.:
# Ren et al. 2018 Scientific Reports: https://www.nature.com/articles/s41598-017-18729-5
# Ulrich 2015 J Neurosci: http://www.jneurosci.org/content/jneuro/35/24/9205.full.pdf
# Ripoli et al. 2014 J Neurosci: http://www.jneurosci.org/content/34/38/12893
# We define here the local inhibitory rate b as a function of Abeta burden:
# Please cite as:
# Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
# R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
# "Linking molecular pathways and large-scale computational modeling to assess candidate
# disease mechanisms and pharmacodynamics in Alzheimer's disease." bioRxiv: 600205.
# taken from
# https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB/blob/master/Educase_AD_study-LS-Surrogate.ipynb
# --------------------------------------------------------------------------------------
import numpy as np

# --------------------------------------------------------------------------------------
# min_val is the lower asymptote of the sigmoid with a inhibitory time constant tau_i = 1/b = 50 ms
# max_val is the differnece between lower and upper asymptote of the sigmoid
# max_val + min_val is the upper asymptote of the sigmoid at tau_i = 1/b = 14.29 ms
# abeta_max is the 95th perecentile of Abeta SUVRs in the original study population
# abeta_off is the cut-off SUVR from which on the sigmoid decreases,
# see Jack et al. 2014 Lancet Neurol: https://www.ncbi.nlm.nih.gov/pubmed/25201514
# --------------------------------------------------------------------------------------
def transform_abeta_exp(abeta, max_val=0.05, min_val=0.02, abeta_max=2.65, abeta_off=1.4):
    x_0 = (abeta_max - abeta_off) / 2 + abeta_off      # x value of  sigmoid midpoint
    k = np.log(max_val / ((min_val+0.001) - min_val) - 1) / (abeta_max - x_0)
    return max_val / (1 + np.exp(k * (abeta - x_0) )) + min_val


# ======================================================================
# ======================================================================
# ======================================================================
