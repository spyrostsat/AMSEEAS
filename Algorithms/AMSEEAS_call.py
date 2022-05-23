# How to call AMSEEAS
# EXAMPLE ON SPHERE FUNCTION

import AMSEEAS

def sphere(x):
    fx = 0
    for l in range(0, len(x)):
        fx += x[l] ** 2
    return fx


a, b, c, d, e, f, g, h, i, j, k = AMSEEAS.amseeas(n=15,m=32, xmin=[-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], xmax=[5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], fn=sphere, maxeval=500)
