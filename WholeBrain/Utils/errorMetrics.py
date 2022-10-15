# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Error metrics between vectors and matrices (arrays)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np


def l2(a,b):
    v = a - b
    vv = v*v
    return np.sqrt(np.sum(vv))


def KullbackLeibler(a,b):
    return np.sum(np.where(a != 0, a*np.log(a/b), 0))


if __name__ == "__main__":
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print("ä*b=", a*b)
    print("ä-b=", a-b)
    print("l2=", l2(a,b))
    print("KL=", KullbackLeibler(a,b))
