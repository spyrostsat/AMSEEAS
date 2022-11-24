import numpy as np
import time
import copy
from pyDOE import lhs


def eas(n, m, xmin, xmax, xlow, xup, fn, maxeval, ftol=10**(-6), ratio=0.95, pmut=0.1, beta=2, maxclimbs=3):


    def CheckBounds(n, xmin, xmax, vec):
        for i in range(n):
            if vec[i, 0] <= xmin[i]:
                vec[i, 0] = xmin[i]
            elif vec[i, 0] >= xmax[i]:
                vec[i, 0] = xmax[i]

        return vec


    def CheckTolerance(fmin, fmax):
        # Check convergence criteria
        if fmin != 0 or fmax != 0:
            tol = (2 * abs(fmax - fmin)) / (abs(fmax) + abs(fmin))
        else:
            tol = 0
        return tol


    # ============================================
    # THE EVOLUTIONARY ANNEALING-SIMPLEX ALGORITHM
    # ============================================

    # INPUT ARGUMENTS
    # n : problem dimension
    # m : population count ( m > n+1 )
    # xmin : lower parameter interior bounds
    # xmax : upper parameter interior bounds
    # xlow : lower parameter exterior bounds
    # xup : upper parameter exterior bounds
    # fn : objective function
    # maxeval : maximum number of function evaluations
    # ftol : the fractional convergence tolerance to be achieved in the function value for an early return
    # ratio : fraction of temperature reduction, when a local minimum is found
    # pmut : probability of accepting an offspring generated via mutation
    # beta : annealing schedule parameter
    # maxclimbs : maximum number of uphill steps

    # OUTPUT ARGUMENTS
    # A python dictionary with the following keys:
        # BestValue: the minimal value of the objective function
        # BestPar: the optimal values of control variables
        # NumIter: number of iterations taken by algorithm
        # NumfEval: number of function evaluations
        # Ftolpop: fractional convergence tolerance of population generated at the last iteration
        # duration_in_secs: Total execution time

    start_time = time.time()

    # generate initial population

    xmin1 = copy.deepcopy(xmin)
    xmin1 *= m
    matXmin = np.array(xmin1)
    matXmin = np.reshape(matXmin, (m, n))

    xmax1 = copy.deepcopy(xmax)
    xmax1 *= m
    matXmax = np.array(xmax1)
    matXmax = np.reshape(matXmax, (m, n))
    latin_hs = lhs(n, samples=m, criterion="cm")  # criterion="c"/"cm"/"m"
    pop = matXmin + (matXmax - matXmin) * latin_hs
    # fitness of initial population

    fpop = np.zeros((m, 1))
    for i in range(m):
        fpop[i, 0] = fn(pop[i, :])

    # temperature of initial population

    temper = fpop.max() - fpop.min()
    # number of function evaluations
    neval = m

    # number of iterations taken by the algorithm

    niter = 0

    ftolpop = CheckTolerance(fpop.min(), fpop.max())

    while neval <= maxeval:
        niter += 1

        # generate a simplex, selecting it's vertices randomly from the actual population

        sn = np.array([])
        while len(sn) < n + 1:
            new = np.random.randint(m)
            if new not in sn:
                sn = np.append(sn, new)
        sn = sn.astype("int")

        s = np.array([])
        for i in range(len(sn)):
            s = np.append(s, pop[sn[i], :], axis=0)
        s = s.reshape((n + 1, n))  # simplex
        fns = np.array([])
        for i in range(len(sn)):
            fns = np.append(fns, fpop[sn[i], 0])  # function value of each vertex of simplex
        fns = np.reshape(fns, (len(sn), 1))

        # determine the lowest (best) point of simplex

        s1pos = fns.argmin()

        # determine the highest (worst) point of simplex

        sn1pos = fns.argmax()

        # check temperature according to annealing schedule parameter, beta

        if temper > beta * (fns[sn1pos, 0] - fns[s1pos, 0]):
            temper = beta * (fns[sn1pos, 0] - fns[s1pos, 0])

        # determine the randomized worst point, according to the criterion xw = max f(x) + rnd*T
        fns1 = copy.deepcopy(fns)
        fns1 = np.delete(fns1, s1pos, axis=0)
        gsfunction = np.zeros((n, 1))
        for i in range(n):
            gsfunction[i, 0] = fns1[i, 0] + np.random.rand() * temper

        wposition = gsfunction.argmax()
        sn1 = copy.deepcopy(sn)
        sn1 = np.delete(sn1, s1pos)
        s1 = copy.deepcopy(s)
        s1 = np.delete(s1, s1pos, axis=0)
        wpopposition = sn1[wposition]
        w = pop[wpopposition, :]  # the randomized worst point
        w = np.reshape(w, (n, 1))
        fnw = fpop[wpopposition, 0]  # the function value of the randomized worst point

        # Compute the modified centroid of the simplex, according to the equation
        # ( max(|f|)/sum(|f|) )*fmin + ( min(|f|)/sum(|f|) )*fmax

        sw = copy.deepcopy(s1)
        sw = np.delete(sw, wposition, axis=0)
        fnsw = copy.deepcopy(fns1)
        fnsw = np.delete(fnsw, wposition, axis=0)
        fnss1 = fns[s1pos, 0]
        sumf = abs(fnsw) + abs(fnss1)
        if sumf.all() != 0:
            minf = np.zeros((len(fnsw), 1))
            maxf = np.zeros((len(fnsw), 1))
            for i in range(len(fnsw)):
                minf[i, 0] = min(abs(fnsw[i, 0]), abs(fnss1))
                maxf[i, 0] = max(abs(fnsw[i, 0]), abs(fnss1))
            pmin = minf / sumf
            pmax = maxf / sumf
            pmax1 = list(pmax)
            pmax1 = pmax1 * n
            pmax1 = np.array(pmax1)
            pmax1 = pmax1.reshape((n - 1, n), order="F")
            s_pmax = s[s1pos, :]
            s_pmax = list(s_pmax)
            s_pmax = s_pmax * (n - 1)
            s_pmax = np.array(s_pmax)
            s_pmax = s_pmax.reshape((n - 1), n)
            s1a = pmax1 * s_pmax
            pmin1 = list(pmin)
            pmin1 = pmin1 * n
            pmin1 = np.array(pmin1)
            pmin1 = pmin1.reshape((n - 1), n, order="F")
            s1b = sw * pmin1
            modpoint = s1a + s1b
            g = np.zeros((n, 1))
            for i in range(n):
                g[i, 0] = sum(modpoint[:, i]) / (n - 1)
        else:
            # compute the centroid of the simplex

            sum_s = sum(s)
            sum_s = np.reshape(sum_s, (n, 1))

            g = (sum_s - w) / n  # to g einai n x 1

        # make a reflection step

        r0 = g + ((0.5 + np.random.rand()) * (g - w))

        # check bounds

        r0 = CheckBounds(n, xlow, xup, r0).flatten()
        g = g.flatten()
        w = w.flatten()
        # function value of reflection step
        if neval >= maxeval:
            break

        fnr0 = fn(r0)

        # number of function evaluations

        neval += 1

        # check if the number of function evaluations exceeded the maximum value

        if neval >= maxeval:
            if fnr0 < fnw:
                w = r0
                pop[wpopposition, :] = w
                fpop[wpopposition, 0] = fnr0
            ftolpop = CheckTolerance(fpop.min(), fpop.max())
            break

        # check if the reflection point is either not accepted (no move) or fr<fw (downhill move) the method follows
        # the modified (quasi-stochastic) Nelder-Mead procedure, making contraction and expansion moves respectively

        if fnr0 < fnw:  # check if the reflected point is better than the randomized worst point of simplex (w)
            if fnr0 < fns[s1pos, 0]:  # check if the reflected point is better than the best point of simplex (s1)

                # line minimization employing subsequent random expansion steps

                ns = 1
                r01 = copy.deepcopy(r0)
                fnr01 = copy.deepcopy(fnr0)

                for i in range(maxeval):
                    if neval >= maxeval:
                        break

                    ns += np.random.rand()
                    rs = g + (ns * (r01 - g))
                    rs = CheckBounds(n, xlow, xup, rs.reshape((n, 1))).flatten()
                    fnrs = fn(rs)
                    neval += 1

                    if fnrs < fnr01:
                        if neval >= maxeval:
                            w = rs
                            fnw = fnrs
                            break
                        else:
                            r01 = rs
                            fnr01 = fnrs
                    else:
                        w = r01
                        fnw = fnr01
                        break

                pop[wpopposition, :] = w
                fpop[wpopposition, 0] = fnw

            else:  # the reflected point is not better than the lowest (best) point of simplex (s1)

                # outside contraction step between xc and xr
                if neval >= maxeval:
                    break

                # outside contraction step

                r1 = g + ((0.25 + 0.5 * np.random.rand()) * (r0 - g))

                # check bounds

                r1 = CheckBounds(n, xlow, xup, r1.reshape((n, 1))).flatten()

                fnr1 = fn(r1)
                neval += 1

                if fnr1 < fnr0:
                    w = r1
                    fnw = fnr1
                else:
                    w = r0
                    fnw = fnr0

                pop[wpopposition, :] = w
                fpop[wpopposition, 0] = fnw

        else:
            # the reflected point is not better than the randomized worst point of simplex (w)
            if neval >= maxeval:
                break

            if fnr0 - np.random.rand() * temper > fnw + np.random.rand() * temper:

                # don't accept the reflection step and try an inside contraction step

                # reduce temperature

                temper = ratio * temper

                # inside contraction

                r1 = g - ((0.25 + 0.5 * np.random.rand()) * (g - r0))

                # Check Bounds

                r1 = CheckBounds(n, xlow, xup, r1.reshape((n, 1))).flatten()

                fnr1 = fn(r1)
                neval += 1

                if fnr1 < fnw:  # successful inside contraction
                    w = r1
                    fnw = fnr1
                    pop[wpopposition, :] = w
                    fpop[wpopposition, 0] = fnw

                else:  # unsuccessful inside contraction - multiple contraction step

                    if neval > maxeval - n:
                        continue

                    s_max = s[s1pos, :]
                    s_max = list(s_max)
                    s_max = s_max * n
                    s_max = np.array(s_max)
                    s_max = s_max.reshape((n, n))
                    mcon = (s1 + s_max) * 0.5

                    for k in range(n):
                        mcon[k, :] = CheckBounds(n, xlow, xup, mcon[k, :].reshape((n, 1))).flatten()

                    # fitness of new points

                    fmcon = np.zeros((n, 1))
                    for it in range(n):
                        fmcon[it, 0] = fn(mcon[it, :])

                    neval += n

                    for p in range(n):
                        pop[sn1[p], :] = mcon[p, :]
                        fpop[sn1[p], 0] = fmcon[p, 0]

            else:  # accept reflection point and try some random uphill steps along the reflection direction
                ns = 1
                r01 = copy.deepcopy(r0)
                fnr01 = copy.deepcopy(fnr0)
                fnrs = 10**50
                for i in range(maxclimbs):  # uphill steps
                    if neval >= maxeval:
                        break
                    ns += np.random.rand()
                    rs = g + (ns * (r01 - g))
                    rs = CheckBounds(n, xlow, xup, rs.reshape((n, 1))).flatten()

                    fnrs = fn(rs)
                    neval += 1

                    if fnrs < fnr01:
                        w = rs
                        fnw = fnrs
                        break
                    elif neval >= maxeval:
                        break
                    else:
                        r01 = rs
                        fnr01 = fnrs

                if fnrs < fnr01:
                    pop[wpopposition, :] = w
                    fpop[wpopposition, 0] = fnw
                else:  # try a mutation step by generating a random point out of the range (xmean-xstdev, xmean+xstdev)

                    if neval >= maxeval:
                        break

                    # compute the statistics of the population

                    meanpop = pop.mean(axis=0).reshape((n, 1))
                    sdpop = pop.std(axis=0).reshape((n, 1))

                    # mutation

                    samplen = np.random.choice([-1, 1], size=n)

                    newpoint = np.zeros((n, 1))
                    for i in range(n):
                        if samplen[i] == 1:
                            if meanpop[i, 0] + sdpop[i, 0] >= xup[i]:
                                newpoint[i, 0] = xup[i]
                            else:
                                newpoint[i, 0] = meanpop[i, 0] + sdpop[i, 0] + (
                                        (xup[i] - (meanpop[i, 0] - sdpop[i, 0])) * np.random.rand())
                        else:
                            if meanpop[i, 0] - sdpop[i, 0] <= xlow[i]:
                                newpoint[i, 0] = xlow[i]
                            else:
                                newpoint[i, 0] = meanpop[i, 0] - sdpop[i, 0] - (
                                        meanpop[i, 0] - sdpop[i, 0] - xlow[i]) * np.random.rand()

                    newpoint = CheckBounds(n, xlow, xup, newpoint).flatten()

                    fnnewpoint = fn(newpoint)
                    neval += 1

                    if fnnewpoint < fnr0:
                        w = newpoint
                        fnw = fnnewpoint
                    else:
                        if np.random.rand() <= pmut:
                            w = newpoint
                            fnw = fnnewpoint
                        else:
                            w = r0
                            fnw = fnr0

                    pop[wpopposition, :] = w
                    fpop[wpopposition, 0] = fnw

        # convergence criteria

        ftolpop = CheckTolerance(fpop.min(), fpop.max())

        # check convergence criteria

        if ftolpop <= ftol or neval >= maxeval:
            break

        # reduce temperature according to annealing schedule parameter, beta

        if temper > beta * (fpop.max() - fpop.min()):
            temper = beta * (fns[sn1pos, 0] - fns[s1pos, 0])

    # determine the optimal solution
    BestValue = fpop.min()
    NumfEval = neval
    NumIter = niter
    Ftolpop = ftolpop
    bestcoor = fpop.argmin()
    BestPar = pop[bestcoor, :]

    end_time = time.time()
    duration = end_time - start_time

    results = {
        'BestValue': BestValue,
        'BestPar': BestPar,
        'NumIter': NumIter,
        'NumfEval': NumfEval,
        'Ftolpop': Ftolpop,
        'duration_in_secs': duration
        }

    return results
