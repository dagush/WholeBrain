# --------------------------------------------------------------------------
#
# This prog. optimizes a function using the greedy optimization method  by [Deco et al. 2014]
# see:
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
# http://www.jneurosci.org/content/34/23/7886.long
#
# Adrian Ponce-Alvarez. Refactoring (& Python translation) by Gustavo Patow
# --------------------------------------------------------------------------
import numpy as np


def update(x, delta, f):
    N = x.shape[0]
    fx = f(x)
    flag = 0
    for n in range(N):
        if np.abs(fx[n]) > 0.005:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
            if fx[n] < 0.:  # if currm_i < -0.026
                x[n] = x[n] - delta[n]  # down-regulate
                delta[n] = delta[n] - 0.001
                if delta[n] < 0.001:
                    delta[n] = 0.001
            else:  # if currm_i >= -0.026 (in the paper, it reads =)
                x[n] = x[n] + delta[n]  # up-regulate
        else:
            flag = flag + 1
    return flag == N


def Optim(f):

    # initialization:
    # -------------------------
    x = init()
    delta = 0.02 * np.ones(x.shape[0])

    print()
    print("  Trials:", end=" ", flush=True)

    ### Balance (greedy algorithm)
    for k in range(5000):  # 5000 trials
        flag = update(x, delta, f)
        # print("k =", k, "  x =", x, "  Delta =", delta)
        if flag:
            # print('Out !!!', flush=True)
            break

    print()
    print('Final trials:', k, '(flag=', flag, ")")
    print('x=', x*180/np.pi, "(in degrees)")
    print('delta=', delta)
    print('f(X)=', f(x))
    return x


# ======================================================================
# ======================================================================
# ======================================================================
def f1(x):
    return np.array([np.cos(x[0])-1, np.cos(x[1])])

def init():
    return np.zeros(2)

if __name__ == '__main__':
    print('Testing simplistic optimization scheme - Test 1: [cos(x)-1, cos(y)] -> (0, 90)')
    Optim(lambda x: np.array([np.cos(x[0])-1, np.cos(x[1])]))  # This is f1
    print('\nTesting simplistic optimization scheme - Test 2: [cos(x), sin(x)] -> (90, 0)')
    Optim(lambda x: np.array([np.cos(x[0]), np.sin(x[1])]))

# ======================================================================
# ======================================================================
# ======================================================================
