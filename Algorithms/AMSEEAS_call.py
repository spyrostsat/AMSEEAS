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


import AMSEEAS


def sphere(x):
    fx = 0
    for l in range(0, len(x)):
        fx += x[l] ** 2
        return fx


params_lower_bounds = [-5.12] * 15
params_upper_bounds = [5.12] * 15

BestValue, BestPar, NumIter, NumfEval, Ftolpop, fmin, fmax, fanaiter, temptemper, S, Y = AMSEEAS.amseeas(n=15, m=32,
                                        xmin=params_lower_bounds, xmax=params_upper_bounds, fn=sphere, maxeval=500)
