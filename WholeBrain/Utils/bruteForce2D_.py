# --------------------------------------------------------------
# Brute force (and plot) 2D optimization
#
# --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


def bruteForce2D(fun, xs, ys,
                 doPlot=True, doPrint=True):
    Z = np.empty(len(xs) * len(ys))
    i = 0
    for y in ys:
        for x in xs:
            Z[i] = fun(np.array([x, y]))
            if doPrint: print([i, x, y, Z[i]])
            i += 1
    X, Y = np.meshgrid(xs, ys)
    Z.shape = X.shape

    if doPlot:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        caxes = axes.matshow(Z, interpolation ='nearest')
        axes.set_xticks(np.arange(0, len(xs), 2))
        axes.set_xticklabels(np.round(xs[::2], decimals=3))
        axes.set_yticks(np.arange(0, len(ys), 2))
        axes.set_yticklabels(np.round(ys[::2], decimals=3))
        fig.colorbar(caxes)
        plt.show()

    minPos = np.where(Z == np.min(Z))
    print(f'x0={X[minPos], Y[minPos]}, f0={Z[minPos]}, niter={len(xs) * len(ys)}')
    return {'x': [X[minPos], Y[minPos]], 'fun': Z[minPos], 'nfev': len(xs) * len(ys), 'status': True, 'message': 'success'}
