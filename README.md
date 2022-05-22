# Official Repository of AMSEEAS (Adaptive Multi-Surrogate Enhanced Evolutionary Annealing Simplex)

This is the official repository of the paper "Advancing surrogate-based optimization of time-expensive environmental problems through adaptive multi-model search". It contains the implementation and the supplementary material of our work.


![Multiple Surrogates Implementation](/pictures/Surrogates_Plots.png)


![Sphere_15D_Example](/pictures/SPHERE_15D.png)


```
# How to call AMSEEAS (exaple on sphere function):


# INPUT ARGUMENTS

# n: problem dimension
# m: population count ( m > n+1 )
# xmin: lower parameter exterior bounds
# xmax: upper parameter exterior bounds
# fn: objective function
# maxeval: maximum number of function evaluations
# ftol: the fractional convergence tolerance to be achieved in the function value for an early return
# pmut: probability of accepting an offspring generated via mutation
# beta: annealing schedule parameter
# maxclimbs: maximum number of uphill steps

# OUTPUT ARGUMENTS

# BestValue: the minimal value of the objective function
# BestPar: the optimal values of control variables
# NumIter: number of iterations taken by algorithm
# NumfEval: number of function evaluations
# Ftolpop: fractional convergence tolerance of population generated at the last iteration
# fmin: array of all best population solutions at the end of every iteration cycle
# fmax: array of all worst population solutions at the end of every iteration cycle
# fanaiter: function evaluations per iteration
# temptemper: array depicting the evolution of the system's temperature
# S: Evaluated sample points
# Y: Objective function values for given S


import amseeas


def sphere(x):
    fx = 0    
    for l in range(0, len(x)):    
    fx += x[l] ** 2    
    return fx


BestValue, BestPar, NumIter, NumfEval, Ftolpop, fmin, fmax, fanaiter, temptemper, S, Y = amseeas.amseeas(15, 32, [-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], [5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], sphere, 500)
```
