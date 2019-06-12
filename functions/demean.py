#--------------------------------------------------------------------------
#
# demean(X)
#     Removes the Average or mean value.
#
#  Taken from the code from:
#    Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#    Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#        Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#
#--------------------------------------------------------------------------
import numpy as np
import numpy.matlib as mtlib

def demean(x,dim=0):
    # DEMEAN(X)
    # Removes the Average or mean value.
    #
    # DEMEAN(X,DIM)
    # Removes the mean along the dimension DIM of X.

    # if (dim == -1):
    #     dim = 0
    #     if (x.shape[0] > 1):
    #         dim = 0;
    #     elif (x.shape[1] > 1):
    #         dim = 1;

    dims = x.size
    # dimsize = x.shape[dim]
    # dimrep = np.ones(1,len(dims));
    # dimrep[dim] = dimsize;

    return x - mtlib.tile(np.mean(x,dim), dims)  # repmat(np.mean(x,dim),dimrep)
