from scipy.optimize import dual_annealing, differential_evolution
from sphere import sphere
from ackley import ackley
from griewank import griewank
from zakharov import zakharov
from rastrigin import rastrigin
from levy import levy


lw = [-5.12] * 30
up = [5.12] * 30

res = dual_annealing(func=sphere, bounds=list(zip(lw, up)), maxfun=500)
print(res)
