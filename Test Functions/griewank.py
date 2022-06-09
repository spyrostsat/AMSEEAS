from math import cos, sqrt

def griewank(x):
    s = 0
    prod = 1
    for l in range(len(x)):
        s += x[l] ** 2
        prod *= cos(x[l] / sqrt(l + 1))
    return (s / 4000) - prod + 1