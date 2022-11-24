# Official Repository of AMSEEAS (Adaptive Multi-Surrogate Enhanced Evolutionary Annealing Simplex)

This is the official repository of the paper "Advancing surrogate-based optimization of time-expensive environmental problems through adaptive multi-model search". It contains the implementation and supplementary material of our work.

If you find this useful, here is our citation:

Tsattalios, S., I. Tsoukalas, P. Dimas, P. Kossieris, A. Efstratiadis and C. Makropoulos (submitted). Advancing surrogate-based optimization of time-expensive environmental problems through adaptive multi-model search. *Environmental Modelling and Software*.

![Multiple Surrogates Implementation](/pictures/Surrogates_Plots.png)

```
# How to call AMSEEAS:

# ====================================================================================================
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
# sampling_method: Method used to create the initial population. Possible values are "lhd" (default) for Latin Hypercube Design and "slhd" for Symmetric Latin Hypercube Design

# ====================================================================================================

# ====================================================================================================
# OUTPUT ARGUMENTS
# A python dictionary with the following keys:
#       BestValue: the minimal value of the objective function
#       BestPar: the optimal values of control variables
#       NumIter: number of iterations taken by algorithm
#       NumfEval: number of function evaluations
#       Ftolpop: fractional convergence tolerance of population generated at the last iteration
#       fmin: array of all best population solutions at the end of every iteration cycle
#       fmax: array of all worst population solutions at the end of every iteration cycle
#       fanaiter: function evaluations per iteration
#       temptemper: array depicting the evolution of the system's temperature
#       S: Evaluated sample points
#       Y: Objective function values for given S
#       duration: Total execution time (s)
# ====================================================================================================

# First load the AMSEEAS global optimization algorithm
from optimization_algorithms.amseeas import amseeas

# Then load the objective function to be minimized
import test_functions

dim = 15
params_lower_bounds = [-5.12] * dim
params_upper_bounds = [5.12] * dim

results = amseeas(n=dim, m=2*dim+1, xmin=params_lower_bounds, xmax=params_upper_bounds,
                  fn=test_functions.sphere.sphere, maxeval=500)

print(f"Best Value = {results['BestValue']}")
```
