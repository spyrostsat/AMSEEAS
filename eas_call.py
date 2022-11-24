# How to call EAS:

# ====================================================================================================
# INPUT ARGUMENTS
# n: problem dimension
# m: population count ( m > n+1 )
# xmin: lower parameter interior bounds
# xmax: upper parameter interior bounds
# xlow: lower parameter exterior bounds
# xup: upper parameter exterior bounds
# fn: objective function
# maxeval: maximum number of function evaluations
# ftol: the fractional convergence tolerance to be achieved in the function value for an early return
# ratio: fraction of temperature reduction, when a local minimum is found
# pmut: probability of accepting an offspring generated via mutation
# beta: annealing schedule parameter
# maxclimbs: maximum number of uphill steps
# ====================================================================================================

# ====================================================================================================
# OUTPUT ARGUMENTS
# A python dictionary with the following keys:
#       BestValue: the minimal value of the objective function
#       BestPar: the optimal values of control variables
#       NumIter: number of iterations taken by algorithm
#       NumfEval: number of function evaluations
#       Ftolpop: fractional convergence tolerance of population generated at the last iteration
#       duration_in_secs: Total execution time
# ====================================================================================================

# First load the EAS global optimization algorithm
from optimization_algorithms.eas import eas

# Then load the objective function to be minimized
import test_functions

dim = 30
params_lower_bounds = [-5.12] * dim
params_upper_bounds = [5.12] * dim

results = eas(n=dim,m=2*dim+1, xmin=params_lower_bounds, xmax=params_upper_bounds,xlow=params_lower_bounds,
              xup=params_upper_bounds, fn=test_functions.sphere.sphere, maxeval=500)

print(f"Best Value = {results['BestValue']}")
