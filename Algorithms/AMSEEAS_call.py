# How to call AMSEEAS
# EXAMPLE ON SPHERE FUNCTION

import AMSEEAS
import SEEAS


def sphere(x):
    fx = 0
    for l in range(0, len(x)):
        fx += x[l] ** 2
    return fx


def zakharov(x):
    sum1 = 0
    sum2 = 0
    for l in range(len(x)):
        sum1 += x[l] ** 2
        sum2 += 0.5 * (l + 1) * x[l]
    return sum1 + sum2 ** 2 + sum2 ** 4


a, b, c, d, e, f, g, h, i, j, k = AMSEEAS.amseeas(n=15,m=32, xmin=[-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], xmax=[5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], fn=sphere, maxeval=500)
# a, b, c, d, e, f, g, h, i, j, k = SEEAS.seeas(n=15,m=32, xmin=[-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], xmax=[5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], fn=sphere, maxeval=500)
# a, b, c, d, e, f, g, h, i, j, k = SEEAS.seeas(15, 32, [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5], [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], zakharov, 500)
