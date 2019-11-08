# Bifurcation diagram test...

def firstTest():
    import matplotlib.pyplot as plt
    import numpy as np
    P=np.linspace(0.7,4,10000)
    m=0.7
    # Initialize your data containers identically
    X = []
    Y = []
    # l is never used, I removed it.
    for u in P:
        # Add one value to X instead of resetting it.
        X.append(u)
        # Start with a random value of m instead of remaining stuck
        # on a particular branch of the diagram
        m = np.random.random()
        for n in range(1001):
          m=(u*m)*(1-m)
        # The break is harmful here as it prevents completion of
        # the loop and collection of data in Y
        for l in range(1051):
          m=(u*m)*(1-m)
        # Collection of data in Y must be done once per value of u
        Y.append(m)
    # Remove the line between successive data points, this renders
    # the plot illegible. Use a small marker instead.
    plt.plot(X, Y, ls='', marker=',')
    plt.show()

def secondTest():
    # Bifurcation diagram of the logistic map

    import math
    from PIL import Image
    imgx = 1000
    imgy = 500
    image = Image.new("RGB", (imgx, imgy))

    xa = 2.9
    xb = 4.0
    maxit = 1000

    for i in range(imgx):
        r = xa + (xb - xa) * float(i) / (imgx - 1)
        x = 0.5
        for j in range(maxit):
            x = r * x * (1 - x)
            if j > maxit / 2:
                image.putpixel((i, int(x * imgy)), (255, 255, 255))

    image.save("Bifurcation.png", "PNG")

secondTest()
