from math import pi, cos

def rastrigin(x):
    s = 0
    for l in range(len(x)):
        s += x[l] ** 2 - 10 * cos(2 * pi * x[l])
    return 10 * len(x) + s