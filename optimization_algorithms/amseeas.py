import time
import copy
from typing import Any
import numpy as np
import statistics
from pyDOE import lhs
from math import log, inf, sqrt
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern
from sklearn.model_selection import train_test_split
from scipy.optimize import dual_annealing
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from optimization_algorithms.initial_sampling import SymmetricLatinHypercube


@ignore_warnings(category=ConvergenceWarning)
def amseeas(n: int,
            m: int,
            xmin: (np.ndarray, list),
            xmax: (np.ndarray, list),
            fn: Any,
            maxeval: int,
            ftol: float = 10 ** (-6),
            pmut: float = 0.1,
            beta: float = 2.0,
            maxclimbs: int = 3,
            sampling_method: str = "lhd") -> dict:

    # ==============================================================================
    # THE ADAPTIVE MULTI-SURROGATE ENHANCED EVOLUTIONARY ANNEALING-SIMPLEX ALGORITHM
    # ==============================================================================

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

    # OUTPUT ARGUMENTS
    # A python dictionary with the following keys:
        # BestValue: the minimal value of the objective function
        # BestPar: the optimal values of control variables
        # NumIter: number of iterations taken by algorithm
        # NumfEval: number of function evaluations
        # Ftolpop: fractional convergence tolerance of population generated at the last iteration
        # fanaiter: function evaluations per iteration
        # S: Evaluated sample points
        # Y: Objective function values for given S
        # duration: Total execution time

    record_Ypred = np.array([])
    record_dist = np.array([])


    def MyRBFcreate(ext_S, Y):
        m = np.shape(ext_S)[0]
        dx = np.shape(ext_S)[1]
        Distances = pairwise_distances(ext_S)  # 2d array m epi m
        # Cubic RBF phi=r^3
        PHI = Distances ** 3
        # Polynomial linear Tail
        PolyDim = dx + 1
        P = np.ones((m, 1))
        P = np.concatenate((P, ext_S), axis=1)  # to P einai m x (n+1)
        # Replace large function values with the median of Y
        Ftransform = copy.deepcopy(Y)  # m x 1
        medianY = statistics.median(Y)[0]
        for i in range(len(Ftransform)):
            if Ftransform[i, 0] > medianY:
                Ftransform[i, 0] = medianY
        # Fit Surrogate model
        a = np.concatenate((PHI, P), axis=1)  # to a einai m x (m+n+1)
        a_rest = np.transpose(P)  # to a_rest einai (n+1) x m
        zeros = np.zeros((PolyDim, PolyDim))  # (n+1) x (n+1)
        a_rest = np.concatenate((a_rest, zeros), axis=1)  # # (n+1) x (m+n+1)
        a = np.concatenate((a, a_rest), axis=0)  # teliko mhtrwo syntelestwn a (m+n+1) x (m+n+1)
        Ftransform = Ftransform.flatten()  # 1 x m
        Ftransform_rest = np.zeros(PolyDim)  # 1 x (n+1)
        Ftransform = np.concatenate((Ftransform, Ftransform_rest))  # 1 x (m+n+1)
        Ftransform.reshape((m + PolyDim, 1))

        eta = sqrt(10 ** (-16) * np.linalg.norm(a, ord=1) * np.linalg.norm(a, ord=np.inf))

        coefs = np.dot(np.linalg.inv(a + eta * np.identity(m + PolyDim)), Ftransform)
        return coefs


    def SVM_EAS(candidate):
        candidate = candidate.reshape(1, -1)
        return svm_regressor.predict(candidate)[0]


    def RandomForest_EAS(candidate):
        candidate = candidate.reshape(1, -1)
        return forest_regressor.predict(candidate)[0]


    def GaussianProcess_EAS_1(candidate):
        candidate = candidate.reshape(1, -1)
        return gaussian_regressor_1.predict(candidate)[0]


    def GaussianProcess_EAS_2(candidate):
        candidate = candidate.reshape(1, -1)
        return gaussian_regressor_2.predict(candidate)[0]


    def MyRBFpred_EAS(ext_S, candidate, coefs):  # where S denotes the population (m x n) and candidate the new point (1 x n)
        m = ext_S.shape[0]  # e.g. MyRBFpred(S, np.array([[1, 2, 3]]), coef)
        n = ext_S.shape[1]
        normY = pairwise_distances(candidate, Y=ext_S, force_all_finite=False)  # 1 x m

        normY = normY.reshape(1, -1)

        # Cubic RBF
        Uy = normY ** 3  # 1 x m
        # Determine the Polynomial Tail (linear)
        Ypred = np.array([[0]], dtype=float)
        b_poly = coefs[-n:]  # einai prwta o statheros oros a kai meta ta bi ston seeas toy matlab
        b_poly = b_poly.reshape((n, 1))
        Ypred += np.dot(candidate, b_poly)
        Ypred += coefs[-n - 1]  # einai prwta o statheros oros a kai meta ta bi ston seeas toy matlab
        li_rbf = coefs[:-n - 1]
        try:
            li_rbf = li_rbf.reshape((m, 1))
        except ValueError:
            li_rbf = li_rbf[:m]
            li_rbf = li_rbf.reshape((m, 1))
        Ypred += np.dot(Uy, li_rbf)
        Ypred = Ypred[0, 0]  # to Ypred einai 2d array, to kanw auto gia na parw thn timh tou Ypred
        return Ypred, normY


    def AF_EAS(ext_S, candidate, coefs, w):  # opou S o population ( m x n ) kai candidate to neo shmeio (1 x n )
        nonlocal record_Ypred, record_dist  # to candidate einai 2d array px np.array([[1, 2, 3]])

        bound_range = xup_help - xlow_help
        minrange = bound_range.min()
        d_size = len(xlow_help)
        tolerance = 0.001 * minrange * np.linalg.norm(np.ones(d_size))
        Ypred, normY = MyRBFpred_EAS(ext_S, candidate, coefs)

        # H kanonikopoihsh ginetai gia na eksasfalisw oti oles oi times tha einai apo 0 ews 1
        # kai o EAS na prospathei na tis ferei konta sto 0

        record_Ypred = np.append(record_Ypred, Ypred)
        record_Ypred_max = record_Ypred.max()
        record_Ypred_min = record_Ypred.min()
        if record_Ypred_max != record_Ypred_min and len(record_Ypred) > 2:
            Ypred_stand = Ypred / (record_Ypred_max - record_Ypred_min)
        else:
            Ypred_stand = 1

        can_dist = min(normY[0])
        record_dist = np.append(record_dist, can_dist)

        record_dist_max = record_dist.max()
        record_dist_min = record_dist.min()
        if record_dist_max != record_dist_min and len(record_dist) > 2:
            can_dist_stand = can_dist / (record_dist_max - record_dist_min)
        else:
            can_dist_stand = 1

        af_can = w * Ypred_stand + (1 - w) * can_dist_stand
        # af_can = w * Ypred + (1 - w) * can_dist
        if can_dist < tolerance:
            af_can = 10 ** 50

        if af_can is inf:
            af_can = 10 ** 50

        return af_can


    def AF_help_EAS(candidate):
        candidate = candidate.reshape(1, -1)
        val = AF_EAS(S, candidate, coef, we)
        return val


    def perfo_perturbEAS(xp, nevalNow, nevalMax, initSample, r, lb, ub):
        d = xp.shape[0]  # to d einai praktika to n (diastash tou provlhmatos)
        lb = np.array(lb)
        ub = np.array(ub)
        bound_range = ub - lb  # 1d array 1 x n
        TFleft = nevalNow - initSample
        p = 0.7 * (1 - (log(TFleft) / log(nevalMax)))
        if p < 0.1:
            p = 0.1
        p_array = np.full((RefTrials, d), p)

        u = np.random.rand(RefTrials, d)
        vect = np.zeros((RefTrials, d), dtype=int)
        xx = u < p_array
        vect[xx] = 1
        Nvect = ((np.random.normal(0, 1, size=(RefTrials, d))) * bound_range) * r
        perturb = Nvect * vect
        xnew = xp + perturb
        return xnew


    def MyRBFpred(ext_S, candidates, coefs):  # opou ext_S to external archive S kai ta candidates einai (trials x n)
        m = ext_S.shape[0]
        n = ext_S.shape[1]
        normY = pairwise_distances(candidates, Y=ext_S, force_all_finite=False)  # trials x m

        # Cubic RBF
        Uy = normY ** 3  # trials x m
        # Determine the Polynomial Tail (linear)

        Ypred = np.zeros((candidates.shape[0], 1), dtype=float)  # trials x 1
        b_poly = coefs[-n:]  # einai prwta o statheros oros a kai meta ta bi ston seeas toy matlab
        b_poly = b_poly.reshape((n, 1))
        Ypred += np.dot(candidates, b_poly)
        Ypred += coefs[-n - 1]  # einai prwta o statheros oros a kai meta ta bi ston seeas toy matlab
        li_rbf = coefs[:-n - 1]
        try:
            li_rbf = li_rbf.reshape((m, 1))
        except ValueError:
            li_rbf = li_rbf[:m]
            li_rbf = li_rbf.reshape((m, 1))
        Ypred += np.dot(Uy, li_rbf)

        return Ypred, normY


    def AF(ext_S, candidates, coefs, w):  # opou ext_S to external archive S kai ta candidates einai (trials x n)
        bound_range = xup_help - xlow_help
        minrange = bound_range.min()
        d_size = len(xlow_help)
        tolerance = 0.001 * minrange * np.linalg.norm(np.ones(d_size))
        Ypred, normY = MyRBFpred(ext_S, candidates, coefs)

        Ypred_max = Ypred.max()
        Ypred_min = Ypred.min()
        # H kanonikopoihsh ginetai gia na eksasfalisw oti oles oi times tha einai apo 0 ews 1
        if Ypred_max != Ypred_min:
            Ypred_stand = (Ypred - Ypred_min) / (Ypred_max - Ypred_min)
        else:
            Ypred_stand = np.ones((Ypred.shape[0], 1))

        Ypred_stand = Ypred_stand.flatten()  # 1 x trials
        can_dist = normY.min(axis=1)  # 1 x trials

        can_dist_max = can_dist.max()
        can_dist_min = can_dist.min()
        if can_dist_max != can_dist_min:
            can_dist_stand = (can_dist_max - can_dist) / (can_dist_max - can_dist_min)  # 1 x trials
        else:
            can_dist_stand = np.ones(can_dist.shape[0])

        af_can = w * Ypred_stand + (1 - w) * can_dist_stand  # 1 x trials

        af_can[can_dist < tolerance] = 10 ** 50

        af_can[af_can == inf] = 10 ** 50

        return af_can.argmin()


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


    # Here starts the amseeas() function

    start_time = time.time()

    xlow = copy.deepcopy(xmin)
    xup = copy.deepcopy(xmax)
    xlow_help = copy.deepcopy(xmin)
    xup_help = copy.deepcopy(xmax)
    xlow_help = np.array(xlow_help)
    xup_help = np.array(xup_help)
    xlow_help_2 = np.array(xmin)
    xup_help_2 = np.array(xmax)
    xlow_help_2 = xlow_help_2.reshape((n, 1))
    xup_help_2 = xup_help_2.reshape((n, 1))

    variable_bounds = np.concatenate((xlow_help_2, xup_help_2), axis=1)

    trials = 10 * n

    RefTrials = 200 * n

    # Preliminary checks
    assert len(xmin) == n and len(xmax) == n, "Boundaries sizes do not match the dimensionality of the problem"
    assert isinstance(n, int) and isinstance(m, int) and isinstance(maxeval, int) and isinstance(maxclimbs, int), "Error in input arguments"
    assert sampling_method == "lhd" or sampling_method == "slhd", "Possible values for the sampling_method argument are 'lhd' and 'slhd'"

    # START ALGORITHM
    ftolpop = None
    fmin = np.array([])
    fmax = np.array([])
    fanaiter = np.array([], dtype=int)
    temptemper = np.array([])

    # generate initial population

    xlow1 = copy.deepcopy(xlow)
    xlow1 *= m
    matXmin = np.array(xlow1, dtype=float)
    matXmin = np.reshape(matXmin, (m, n))

    xup1 = copy.deepcopy(xup)
    xup1 *= m
    matXmax = np.array(xup1, dtype=float)
    matXmax = np.reshape(matXmax, (m, n))

    # Initial Sampling Method
    if sampling_method == "lhd":
        latin_hs = lhs(n, samples=m, criterion="cm")
    elif sampling_method == "slhd":
        latin_hs = SymmetricLatinHypercube(dim=n, npts=m).generate_points()

    pop = matXmin + ((matXmax - matXmin) * latin_hs)
    # fitness of initial population through the expensive objective function

    fpop = np.zeros((m, 1), dtype=float)
    for i in range(m):
        fpop[i, 0] = fn(pop[i, :])

    # Increase the number of function evaluations

    neval = m
    Cmut = 0  # Reset Mutation counter
    # Temperature of initial population

    temper = fpop.max() - fpop.min()
    temper0 = copy.deepcopy(temper)
    temptemper = np.append(temptemper, temper)

    # Initiate the number of iterations taken by the algorithm

    niter = 0

    # External Archive for Surrogate model

    S = copy.deepcopy(pop)
    Y = copy.deepcopy(fpop)

    svm_regressor = SVR(kernel='rbf')  # Gaussian RBF
    gaussian_regressor_1 = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(10 ** (-13), 10 ** 13)))
    forest_regressor = RandomForestRegressor(n_estimators=int(m / 2))
    gaussian_regressor_2 = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds=(10 ** (-13), 10 ** 13)))

    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0

    term_surrogates = False

    coef = MyRBFcreate(S, Y)

    max_pen = max(int(maxeval / 166), 3)

    while neval <= maxeval:

        niter += 1
        we = max(0.85, min((log(neval) / log(maxeval)), 0.95))

        if term_surrogates is False:

            index = np.array([])
            for z in range(S.shape[0]):
                index = np.append(index, z)
            index = index.astype("int")
            index = index.reshape((len(index), 1))

            index_train, index_test = train_test_split(index, test_size=0.2)
            index_train = index_train.flatten()
            index_test = index_test.flatten()

            if counter_1 < max_pen:
                gaussian_regressor_1.fit(S[index_train], Y[index_train].ravel())
                pred_1 = gaussian_regressor_1.predict(S[index_test]).reshape((len(index_test), 1))
                nse_1 = r2_score(Y[index_test], pred_1)
            else:
                nse_1 = 0

            if counter_2 < max_pen:
                svm_regressor.fit(S[index_train], Y[index_train].ravel())
                pred_2 = svm_regressor.predict(S[index_test]).reshape((len(index_test), 1))
                nse_2 = r2_score(Y[index_test], pred_2)
            else:
                nse_2 = 0

            if counter_4 < max_pen:
                gaussian_regressor_2.fit(S[index_train], Y[index_train].ravel())
                pred_4 = gaussian_regressor_2.predict(S[index_test]).reshape((len(index_test), 1))
                nse_4 = r2_score(Y[index_test], pred_4)
            else:
                nse_4 = 0

            if counter_5 < max_pen:
                forest_regressor.fit(S[index_train], Y[index_train].ravel())
                pred_5 = forest_regressor.predict(S[index_test]).reshape((len(index_test), 1))
                nse_5 = r2_score(Y[index_test], pred_5)
            else:
                nse_5 = 0

            try:
                if counter_3 < max_pen:
                    coef2 = MyRBFcreate(S[index_train], Y[index_train])
                    pred_3 = MyRBFpred(S[index_train], S[index_test], coef2)[0]
                    nse_3 = r2_score(Y[index_test], pred_3)
                else:
                    nse_3 = 0

                if nse_1 < 0:  # surrogates in which the NSE is negative are expected to have a bad fitting in the
                    nse_1 = 0  # external archive S and thus are given NSE = 0 so that the corresponding probabilities
                if nse_2 < 0:  # in the roulette are 0 as well. That way when a surrogate has a bad fitting in the
                    nse_2 = 0  # external archive ( NSE < 0 ) we don't even let it participate in the roulette so that
                if nse_3 < 0:  # there is no chance for it to be chosen and possibly return a bad prediction and thus
                    nse_3 = 0  # get penalized.
                if nse_4 < 0:
                    nse_4 = 0
                if nse_5 < 0:
                    nse_5 = 0

                sum_nse = nse_1 + nse_2 + nse_3 + nse_4 + nse_5  # sum_nse >= 0
                # print(f"nse_1 = {nse_1}  nse_2 = {nse_2}  nse_3 = {nse_3}  nse_4 = {nse_4}  nse_5 = {nse_5}")

                if sum_nse != 0:
                    p1 = nse_1 / sum_nse  # we could think of the possibility of not having counters for the bad predictions of the surrogates and permanently deactivating them when they reach a threshold, but instead have a threshold for the NSE in order to activate any surrogate in each iteration, which threshold starts with NSE = 0, but when we come closer to the MFE it could grow larger, for example NSE = 0.9
                    p2 = nse_2 / sum_nse
                    p3 = nse_3 / sum_nse
                    p4 = nse_4 / sum_nse
                    p5 = nse_5 / sum_nse
                else:
                    p1 = 0
                    p2 = 0
                    p3 = 0
                    p4 = 0
                    p5 = 0

                # print(f"p1={p1:.3f}  p2={p2:.3f}  p3={p3:.3f}  p4={p4:.3f}  p5={p5:.3f}\n")

                num = np.random.uniform(0, p1 + p2 + p3 + p4 + p5)

                # print(f"NUMBER = {num:.3f}\n")

                if num < p1:
                    # print("1  GAUSSIAN PROCESS (kernel=MATERN)\n")
                    gaussian_regressor_1.fit(S, Y.ravel())

                    model1 = dual_annealing(GaussianProcess_EAS_1, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model1.x

                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_1 += 1

                elif num < p1 + p2:
                    # print("2  SUPPORT VECTOR MACHINE\n")
                    svm_regressor.fit(S, Y.ravel())

                    model2 = dual_annealing(SVM_EAS, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model2.x
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_2 += 1

                elif num < p1 + p2 + p3:
                    # print("3  RBF")

                    model3 = dual_annealing(AF_help_EAS, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model3.x

                    record_dist = np.array([])
                    record_Ypred = np.array([])
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_3 += 1

                elif num < p1 + p2 + p3 + p4:
                    # print("4  GAUSSIAN PROCESS (kernel=RATIONAL QUADRATIC)\n")
                    gaussian_regressor_2.fit(S, Y.ravel())

                    model4 = dual_annealing(GaussianProcess_EAS_2, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model4.x
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_4 += 1

                elif num < p1 + p2 + p3 + p4 + p5:
                    # print("5  RANDOM FOREST\n")
                    forest_regressor.fit(S, Y.ravel())

                    model5 = dual_annealing(RandomForest_EAS, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model5.x
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_5 += 1

                if counter_1 + counter_2 + counter_3 + counter_4 + counter_5 == 5 * max_pen:
                    term_surrogates = True

            except np.linalg.LinAlgError:

                if nse_1 < 0:
                    nse_1 = 0
                if nse_2 < 0:
                    nse_2 = 0
                if nse_4 < 0:
                    nse_4 = 0
                if nse_5 < 0:
                    nse_5 = 0

                sum_nse = nse_1 + nse_2 + nse_4 + nse_5

                # print(f"nse_1 = {nse_1}  nse_2 = {nse_2}  nse_4 = {nse_4}  nse_5 = {nse_5}")

                if sum_nse != 0:
                    p1 = nse_1 / sum_nse
                    p2 = nse_2 / sum_nse
                    p4 = nse_4 / sum_nse
                    p5 = nse_5 / sum_nse
                else:
                    p1 = 0
                    p2 = 0
                    p4 = 0
                    p5 = 0

                # print(f"p1={p1:.3f}  p2={p2:.3f}  p4={p4:.3f}  p5={p5:.3f}\n")

                num = np.random.uniform(0, p1 + p2 + p4 + p5)
                # print(f"NUMBER = {num:.3f}\n")

                if num < p1:
                    # print("1  GAUSSIAN PROCESS (kernel=MATERN)\n")
                    gaussian_regressor_1.fit(S, Y.ravel())

                    model1 = dual_annealing(GaussianProcess_EAS_1, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model1.x

                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_1 += 1

                elif num < p1 + p2:
                    # print("2  SUPPORT VECTOR MACHINE\n")
                    svm_regressor.fit(S, Y.ravel())

                    model2 = dual_annealing(SVM_EAS, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model2.x
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_2 += 1

                elif num < p1 + p2 + p4:
                    # print("4  GAUSSIAN PROCESS (kernel=RATIONAL QUADRATIC)\n")
                    gaussian_regressor_2.fit(S, Y.ravel())

                    model4 = dual_annealing(GaussianProcess_EAS_2, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model4.x
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_4 += 1

                elif num < p1 + p2 + p4 + p5:
                    # print("5  RANDOM FOREST\n")
                    forest_regressor.fit(S, Y.ravel())

                    model5 = dual_annealing(RandomForest_EAS, variable_bounds, maxiter=1000, maxfun=10000)

                    BestPar = model5.x
                    final_f = fn(BestPar)

                    final_f_ar = np.array([final_f])

                    neval += 1

                    S = np.concatenate((S, BestPar.reshape(1, -1)), axis=0)
                    Y = np.concatenate((Y, final_f_ar.reshape(1, -1)), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    # Replace the Worst of population

                    maxpopf = fpop.max()
                    # print(f"\nfn(BestPar)={final_f:.5f}  maxpopf={maxpopf:.5f}\n")

                    if final_f < maxpopf:
                        index = fpop.argmax()
                        pop[index, :] = BestPar
                        fpop[index, 0] = final_f
                    else:
                        counter_5 += 1

        if neval >= maxeval:
            break

        # Generate a simplex selecting its vertices randomly from the actual population

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
        fns = np.reshape(fns, (n + 1, 1))

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
            gsfunction[i, 0] = fns1[i, 0] + (np.random.rand() * temper)

        wposition = gsfunction.argmax()
        sn1 = copy.deepcopy(sn)
        sn1 = np.delete(sn1, s1pos)
        s1 = copy.deepcopy(s)
        s1 = np.delete(s1, s1pos, axis=0)
        wpopposition = sn1[wposition]

        w = pop[wpopposition, :]  # the randomized worst point
        w = np.reshape(w, (n, 1))
        fnw = fpop[wpopposition, 0]  # the function value of the randomized worst point

        # Compute the centroid of the simplex

        sum_s = sum(s)
        sum_s = np.reshape(sum_s, (n, 1))

        g = (sum_s - w) / n  # to g einai n x 1

        # make a reflection step

        r0 = np.zeros((trials, n))
        linsp = np.linspace(0, 1, trials).reshape((trials, 1))
        linsp += 0.5
        r0 += (g - w).flatten()
        r0 *= linsp
        r0 += g.flatten()
        xmin_ar = np.zeros((trials, n))
        xmin_ar += np.array(xmin)
        xmax_ar = np.zeros((trials, n))
        xmax_ar += np.array(xmax)
        xx = r0 > xmax_ar
        r0[xx] = xmax_ar[xx]
        yy = r0 < xmin_ar
        r0[yy] = xmin_ar[yy]

        # Finding the best reflection point r0 of the above that gives the lowest AF

        best_r0 = AF(S, r0, coef, we)

        final_r0 = r0[best_r0, :]

        # function value of reflection step

        fnr0 = fn(final_r0)
        fnr0_ar = np.array([fnr0])
        neval += 1

        # check if the number of function evaluations exceeded the maximum value

        if neval >= maxeval:
            if fnr0 < fnw:
                w = final_r0
                pop[wpopposition, :] = w
                fpop[wpopposition, 0] = fnr0

            ftolpop = CheckTolerance(fpop.min(), fpop.max())
            break

        S = np.append(S, final_r0.reshape(1, -1), axis=0)

        Y = np.append(Y, fnr0_ar.reshape(1, -1), axis=0)

        try:
            coef = MyRBFcreate(S, Y)
        except np.linalg.LinAlgError:
            errorcoeff = coef.mean()
            coef = np.append(coef, errorcoeff)

        g = g.flatten()

        # check if the reflection point is either not accepted (no move) or fr<fw (downhill move) the method follows
        # the modified (quasi-stochastic) Nelder-Mead procedure, making contraction and expansion moves respectively

        if fnr0 < fnw:  # check if the reflected point is better than the randomized worst point of simplex (w)
            if fnr0 < fns[s1pos, 0]:  # the reflected point is better than the lowest (best) point of simplex (s1)

                # line minimization employing subsequent random expansion steps

                r01 = copy.deepcopy(final_r0)

                fnr01 = copy.deepcopy(fnr0)
                ns0 = 1
                linsp2 = np.linspace(0, 1, trials).reshape((trials, 1))

                for i in range(maxeval):  # multiple expansion

                    if neval >= maxeval:
                        break

                    rs = np.zeros((trials, n))

                    ns = ns0 + linsp2
                    rs += r01 - g
                    rs *= ns
                    rs += g

                    xx = rs > xmax_ar
                    rs[xx] = xmax_ar[xx]
                    yy = rs < xmin_ar
                    rs[yy] = xmin_ar[yy]

                    ns0 += np.random.rand()

                    # Finding the best reflection point rs of the above that gives the lowest AF

                    best_rs = AF(S, rs, coef, we)

                    final_rs = rs[best_rs, :]

                    # function value of reflection step

                    fnrs = fn(final_rs)
                    fnrs_ar = np.array([fnrs])
                    neval += 1

                    S = np.append(S, final_rs.reshape(1, -1), axis=0)

                    Y = np.append(Y, fnrs_ar.reshape(1, -1), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    if fnrs < fnr01:
                        if neval >= maxeval:
                            w = final_rs
                            fnw = fnrs
                            break
                        else:
                            r01 = final_rs
                            fnr01 = fnrs
                    else:
                        w = r01
                        fnw = fnr01
                        break

                pop[wpopposition, :] = w
                fpop[wpopposition, 0] = fnw

            else:  # the reflected point is not better than the lowest (best) point of simplex (s1)
                linsp3 = np.linspace(0, 1, trials).reshape((trials, 1))

                r1 = np.zeros((trials, n))

                linsp3 *= 0.5
                linsp3 += 0.25
                r1 += final_r0 - g
                r1 *= linsp3
                r1 += g

                xx = r1 > xmax_ar
                r1[xx] = xmax_ar[xx]
                yy = r1 < xmin_ar
                r1[yy] = xmin_ar[yy]

                # Finding the best reflection point r1 of the above that gives the lowest AF

                best_r1 = AF(S, r1, coef, we)

                final_r1 = r1[best_r1, :]

                fnr1 = fn(final_r1)
                fnr1_ar = np.array([fnr1])
                neval += 1

                S = np.append(S, final_r1.reshape(1, -1), axis=0)

                Y = np.append(Y, fnr1_ar.reshape(1, -1), axis=0)

                try:
                    coef = MyRBFcreate(S, Y)
                except np.linalg.LinAlgError:
                    errorcoeff = coef.mean()
                    coef = np.append(coef, errorcoeff)

                if fnr1 < fnr0:
                    w = final_r1
                    fnw = fnr1
                else:
                    w = final_r0
                    fnw = fnr0

                pop[wpopposition, :] = w
                fpop[wpopposition, 0] = fnw

        else:  # the reflected point is not better than the randomized worst point of simplex (w)

            if fnr0 - np.random.rand() * temper > fnw + np.random.rand() * temper:
                # don't accept the reflection step and try an inside contraction step
                ratio = max(1 - (log(neval - m) / log(maxeval)), 0.50)
                # Reduce temperature
                temper = ratio * temper0

                linsp4 = np.linspace(0, 1, trials).reshape((trials, 1))

                r1 = np.zeros((trials, n))

                linsp4 *= 0.5
                linsp4 += 0.25
                r1 += g - w.flatten()
                r1 *= linsp4
                r1 = - r1
                r1 += g
                xx = r1 > xmax_ar
                r1[xx] = xmax_ar[xx]
                yy = r1 < xmin_ar
                r1[yy] = xmin_ar[yy]

                # Finding the best reflection point r1 of the above that gives the lowest AF

                best_r1 = AF(S, r1, coef, we)

                final_r1 = r1[best_r1, :]

                fnr1 = fn(final_r1)
                fnr1_ar = np.array([fnr1])
                neval += 1

                S = np.append(S, final_r1.reshape(1, -1), axis=0)

                Y = np.append(Y, fnr1_ar.reshape(1, -1), axis=0)

                try:
                    coef = MyRBFcreate(S, Y)
                except np.linalg.LinAlgError:
                    errorcoeff = coef.mean()
                    coef = np.append(coef, errorcoeff)

                if fnr1 < fnw:  # successful inside contraction

                    w = final_r1
                    fnw = fnr1
                    pop[wpopposition, :] = w
                    fpop[wpopposition, 0] = fnw

                else:  # unsuccessful inside contraction - multiple contraction step - Shrinkage

                    if neval >= maxeval:
                        break
                    elif neval > maxeval - n:
                        continue

                    # new points
                    mat_s1pos = s[s1pos, :]
                    mat_s1pos = list(mat_s1pos)
                    mat_s1pos = mat_s1pos * n
                    mat_s1pos = np.array(mat_s1pos)
                    mat_s1pos = mat_s1pos.reshape((n, n))
                    mcon = (s1 + mat_s1pos) * 0.5
                    for r in range(n):
                        mcon[r, :] = CheckBounds(n, xlow, xup, mcon[r, :].reshape((n, 1))).flatten()

                    # fitness of new points

                    fmcon = np.zeros((n, 1))
                    for it in range(n):
                        fmcon[it, 0] = fn(mcon[it, :])

                    neval += n

                    S = np.append(S, mcon, axis=0)

                    Y = np.append(Y, fmcon, axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        for _ in range(n):
                            errorcoeff = coef.mean()
                            coef = np.append(coef, errorcoeff)

                    for p in range(n):
                        pop[sn1[p], :] = mcon[p, :]
                        fpop[sn1[p], 0] = fmcon[p, 0]

            else:  # accept the reflection point and try some random uphill steps along the reflection direction

                ns0 = 1
                fnrs = 10 ** 50
                r01 = copy.deepcopy(final_r0)

                fnr01 = copy.deepcopy(fnr0)
                linsp5 = np.linspace(0, 2, trials).reshape((trials, 1))
                for il in range(maxclimbs):  # uphill steps
                    rs = np.zeros((trials, n))

                    ns = ns0 + linsp5
                    rs += r01 - g
                    rs *= ns
                    rs += g

                    xx = rs > xmax_ar
                    rs[xx] = xmax_ar[xx]
                    yy = rs < xmin_ar
                    rs[yy] = xmin_ar[yy]

                    ns0 += np.random.rand()

                    # Finding the best reflection point rs of the above that gives the lowest AF

                    best_rs = AF(S, rs, coef, we)

                    final_rs = rs[best_rs, :]

                    fnrs = fn(final_rs)
                    fnrs_array = np.array([fnrs])
                    neval += 1

                    S = np.append(S, final_rs.reshape(1, -1), axis=0)

                    Y = np.append(Y, fnrs_array.reshape(1, -1), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    if fnrs < fnr01:
                        w = final_rs
                        fnw = fnrs
                        break
                    elif neval >= maxeval:
                        break
                    else:
                        r01 = final_rs
                        fnr01 = fnrs
                if fnrs < fnr01:
                    pop[wpopposition, :] = w
                    fpop[wpopposition, 0] = fnw

                else:  # try a mutation step by generating a random point out of range (xmean-xstdev, xmean+xstdev)
                    # Increase the number of function evaluations

                    if neval >= maxeval:
                        break

                    # Compute the statistics of the population

                    meanpop = pop.mean(axis=0).reshape((n, 1))
                    sdpop = pop.std(axis=0).reshape((n, 1))

                    Cmut += 1  # Mutation Counter
                    # mutation
                    samplen = np.random.choice([-1, 1], size=n)
                    newpoint = np.zeros(n)
                    for k in range(n):
                        if samplen[k] == 1:
                            if meanpop[k, 0] + sdpop[k, 0] >= xup[k]:
                                newpoint[k] = xup[k]
                            else:
                                newpoint[k] = meanpop[k, 0] + sdpop[k, 0] + (
                                        xup[k] - meanpop[k, 0] - sdpop[k, 0]) * np.random.rand()
                        else:
                            if meanpop[k, 0] - sdpop[k, 0] <= xlow[k]:
                                newpoint[k] = xlow[k]
                            else:
                                newpoint[k] = meanpop[k, 0] - sdpop[k, 0] - (
                                        meanpop[k, 0] - sdpop[k, 0] - xlow[k]) * np.random.rand()

                    newpoint = CheckBounds(n, xlow, xup, newpoint.reshape((n, 1))).flatten()

                    fnnewpoint = fn(newpoint)
                    neval += 1
                    fnnewpoint_ar = np.array([fnnewpoint])

                    S = np.append(S, newpoint.reshape(1, -1), axis=0)

                    Y = np.append(Y, fnnewpoint_ar.reshape(1, -1), axis=0)

                    try:
                        coef = MyRBFcreate(S, Y)
                    except np.linalg.LinAlgError:
                        errorcoeff = coef.mean()
                        coef = np.append(coef, errorcoeff)

                    if fnnewpoint < fnr0:
                        w = newpoint
                        fnw = fnnewpoint
                    else:
                        if np.random.rand() <= pmut:
                            w = newpoint
                            fnw = fnnewpoint
                        else:
                            w = final_r0
                            fnw = fnr0
                    pop[wpopposition, :] = w
                    fpop[wpopposition, 0] = fnw

        if neval >= maxeval:
            break

        # Refinement step

        bestpos = fpop.argmin()

        paramREf = pop[bestpos, :]

        xnewD1 = perfo_perturbEAS(paramREf, neval, maxeval, m, 0.2, xlow, xup)  # ( RefTrials x n )
        xmin_ar2 = np.zeros((RefTrials, n))
        xmin_ar2 += np.array(xmin)
        xmax_ar2 = np.zeros((RefTrials, n))
        xmax_ar2 += np.array(xmax)
        xx = xnewD1 > xmax_ar2
        xnewD1[xx] = xmax_ar2[xx]
        yy = xnewD1 < xmin_ar2
        xnewD1[yy] = xmin_ar2[yy]

        ypred = MyRBFpred(S, xnewD1, coef)[0]

        index = ypred.argmin()

        final_xnewD1 = xnewD1[index, :]

        fnewD1 = fn(final_xnewD1)
        neval += 1
        fnewD1_ar = np.array([fnewD1])
        S = np.append(S, final_xnewD1.reshape(1, -1), axis=0)

        Y = np.append(Y, fnewD1_ar.reshape(1, -1), axis=0)

        try:
            coef = MyRBFcreate(S, Y)
        except np.linalg.LinAlgError:
            errorcoeff = coef.mean()
            coef = np.append(coef, errorcoeff)

        ind = fpop.argmax()

        if fnewD1 < fpop[ind, 0]:
            pop[ind, :] = final_xnewD1
            fpop[ind, 0] = fnewD1

        fmin = np.append(fmin, fpop.min())
        fmax = np.append(fmax, fpop.max())
        fanaiter = np.append(fanaiter, neval)

        if isinstance(temper, float):
            temptemper = np.append(temptemper, temper)
        else:
            temptemper = np.append(temptemper, float(temper[0]))

        # convergence criteria

        ftolpop = CheckTolerance(fpop.min(), fpop.max())

        # check convergence criteria

        if ftolpop <= ftol or neval >= maxeval:
            break

        # Display Option
        print(f"FE = {neval}, BestValue = {fpop.min()}")

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
        'fmin': fmin,
        'fmax': fmax,
        'fanaiter': fanaiter,
        'temptemper': temptemper,
        'S': S,
        'Y': Y,
        'duration': duration
        }

    return results
