# Official Repository of AMSEEAS (Adaptive Multi-Surrogate Enhanced Evolutionary Annealing Simplex)

How to call AMSEEAS (exaple on sphere function):
```
import amseeas


def sphere(x):
    fx = 0    
    for l in range(0, len(x)):    
    fx += x[l] ** 2    
    return fx


BestValue, BestPar, NumIter, NumfEval, Ftolpop, fmin, fmax, fanaiter, temptemper, S, Y = amseeas.amseeas(15, 32, [-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], [5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], sphere, 500)
```
