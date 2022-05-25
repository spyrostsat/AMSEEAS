from math import cos, exp, pi, sqrt

def ackley(x):
    dd = len(x)
    sum1 = 0
    sum2 = 0
    for ii in range(dd):
        sum1 += x[ii] ** 2
        sum2 += cos(2 * pi * x[ii])
    return -20 * exp(-0.2 * sqrt(sum1 / dd)) - exp(sum2 / dd) + 20 + exp(1)