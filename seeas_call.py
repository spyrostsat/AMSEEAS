# How to call SEEAS:

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
#       duration_in_secs: Total execution time
# ====================================================================================================

# First load the SEEAS global optimization algorithm
from optimization_algorithms.seeas import seeas

# Then load the objective function to be minimized
import test_functions

dim = 15
params_lower_bounds = [-600] * dim
params_upper_bounds = [600] * dim

results = seeas(n=dim, m=2*dim+1, xmin=params_lower_bounds, xmax=params_upper_bounds,
                  fn=test_functions.griewank.griewank, maxeval=500)

print(f"Best Value = {results['BestValue']}")
