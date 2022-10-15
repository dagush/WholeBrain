# --------------------------------------------------------------------------------------
# Plot Figure 3 in:
# [StefanovskiEtAl2019] Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
#           R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
#           "Linking molecular pathways and large-scale computational modeling to assess candidate
#           disease mechanisms and pharmacodynamics in Alzheimer's disease."
#           Front. Comput. Neurosci., 13 August 2019 | https://doi.org/10.3389/fncom.2019.00054
# Taken from the code at:
#           https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB/blob/master/Educase_AD_study-LS-Surrogate.ipynb
#
# --------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from WholeBrain.Models import Abeta_StefanovskiEtAl2019 as Abeta


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22})
    # Cause-and-effect model for Abeta: sigmoidal transfer function
    # --------------------------------------------------------------
    # visualize
    x = np.arange(1., 3, 0.01)
    plt.plot(x, Abeta.transform_abeta_exp(x))
    plt.xlabel("Abeta PET SUVR")
    plt.ylabel("inhibitory rate b")
    plt.suptitle("Sigmoidal transfer function", fontweight="bold", fontsize="18", y = 1.05)
    plt.grid()
    plt.show()
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
