#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the Phase Functional Connectivity Dynamics (phFCD)
#
#  Translated to Python & refactoring by Xenia Kobeleva
#  Revised by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
from scipy import signal
# from scipy import stats
from functions import BOLDFilters
from functions.Utils import demean

BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


def adif(a,b):
    if np.abs(a-b) > np.pi:
        c = 2*np.pi-np.abs(a-b)
    else:
        c = np.abs(a-b)
    return c


def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i, row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


def phFCD(ts_emp):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = ts_emp.shape
    # Data structures we are going to need...
    phases_emp=np.zeros([N,Tmax])   
    patt = np.zeros([N,N-1])
    pattern = np.zeros([Tmax-19, int(N*(N-1)/2)])  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
    syncdata=np.zeros(Tmax-19)

    ts_emp_filt = BOLDFilters.BandPassFilter(ts_emp)  # zero phase filter the data
    for n in range(N):
        Xanalytic = signal.hilbert(demean.demean(ts_emp_filt[n,:]))
        phases_emp[n,:]= np.angle(Xanalytic)

    Isubdiag = tril_indices_column(N, k=-1)  # Indices of triangular lower part of matrix
    T = np.arange(10,Tmax-10+1)
    for t in T:
        kudata = np.sum(np.cos(phases_emp[:,t-1]) + 1j * np.sin(phases_emp[:,t-1]))/N
        syncdata[t-10] = abs(kudata)
        for i in range(N):
            for j in range(i):
                patt[i,j] = np.cos(adif(phases_emp[i,t-1], phases_emp[j,t-1]))
        pattern[t-10,:] = patt[Isubdiag]

    npattmax = Tmax-19  # calculates the size of phfcd vector
    size_kk3 = int((npattmax-3)*(npattmax-2)/2)  # The int() is not needed, but... (see above)
    phfcd = np.zeros([size_kk3])

    kk3 = 0
    for t in range(npattmax-2):
        p1 = np.mean(pattern[t:t+3,:], axis=0)
        for t2 in range(t+1,npattmax-2):
            p2 = np.mean(pattern[t2:t2+3,:], axis=0)
            phfcd[kk3] = np.dot(p1,p2)/np.linalg.norm(p1)/np.linalg.norm(p2)
            kk3 = kk3 + 1

    return phfcd
