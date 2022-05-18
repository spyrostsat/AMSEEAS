# AMSEEAS (Adaptive Multi-Surrogate Enhanced Evolutionary Annealing Simplex)

Official Repository of AMSEEAS surrogate-based global optimization algorithm

How to call AMSEEAS

EXAMPLE ON SPHERE FUNCTION

import amseeas

def sphere(x):
    fx = 0
    for l in range(0, len(x)):
        fx += x[l] ** 2
    return fx


a, b, c, d, e, f, g, h, i, j, k = amseeas.amseeas(15, 32, [-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], [5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], sphere, 500)
