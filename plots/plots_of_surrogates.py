import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pyDOE import lhs
from math import log, inf, sqrt, exp, sin, cos, pi
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern
from sklearn.model_selection import train_test_split
import copy
import statistics


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

    eta = sqrt(10**(-16) * np.linalg.norm(a, ord=1) * np.linalg.norm(a, ord=np.inf))

    coefs = np.dot(np.linalg.inv(a + eta * np.identity(m + PolyDim)), Ftransform)
    return coefs


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


# I CHECKED IT AND IT IS CORRECT
def MyRBFpred_EAS(ext_S, candidate, coefs):  # opou S o population ( m x n ) kai candidate to neo shmeio (1 x n )
    m = ext_S.shape[0]  # px. MyRBFpred(S, np.array([[1, 2, 3]]), coef)
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


def spherical(x):  # opou x mia python list h 1d array
    fx = 0
    for l in range(0, len(x)):
        fx += x[l] ** 2
    return fx


def ackley(x):
    dd = len(x)
    sum1 = 0
    sum2 = 0
    for ii in range(dd):
        sum1 += x[ii] ** 2
        sum2 += cos(2 * pi * x[ii])
    return -20 * exp(-0.2 * sqrt(sum1 / dd)) - exp(sum2 / dd) + 20 + exp(1)


def griewank(x):
    s = 0
    prod = 1
    for l in range(len(x)):
        s += x[l] ** 2
        prod *= cos(x[l] / sqrt(l + 1))
    return (s / 4000) - prod + 1


def zakharov(x):
    sum1 = 0
    sum2 = 0
    for l in range(len(x)):
        sum1 += x[l] ** 2
        sum2 += 0.5 * (l + 1) * x[l]
    return sum1 + sum2 ** 2 + sum2 ** 4


def rastrigin(x):
    s = 0
    for l in range(len(x)):
        s += x[l] ** 2 - 10 * cos(2 * pi * x[l])
    return 10 * len(x) + s


def levy(x):
    w1 = 1 + (x[0] - 1) / 4
    wd = 1 + (x[len(x) - 1] - 1) / 4
    s = 0
    for l in range(len(x) - 1):
        w = 1 + (x[l] - 1) / 4
        s += ((w - 1) ** 2) * (1 + 10 * (sin(pi * w + 1) ** 2))
    return (sin(pi * w1)) ** 2 + s + (((wd - 1) ** 2) * (1 + (sin(2 * pi * wd)) ** 2))


# Lets test the levy function with two variables in the domain [-10, 10]
lower_boundary = -10
upper_boundary = 10
step_inc = 0.5 # total samples to depict the actual levy function
n_samples = 200 # population
n = 2 # control variables

x_low = [lower_boundary, lower_boundary] * n_samples
x_low = np.array(x_low, dtype=float)
x_low = np.reshape(x_low, (n_samples, n))

x_up = [upper_boundary, upper_boundary] * n_samples
x_up = np.array(x_up, dtype=float)
x_up = np.reshape(x_up, (n_samples, n))

latin_hs = lhs(n, samples=n_samples, criterion="cm")  # criterion="c"/"cm"/"m"
pop = x_low + ((x_up - x_low) * latin_hs)
fpop = np.zeros((n_samples, 1), dtype=float) # real solutions
for i in range(n_samples):
    fpop[i, 0] = levy(pop[i, :])
# fpop = fpop.flatten()

S = copy.deepcopy(pop)
Y = copy.deepcopy(fpop)


svm_regressor = SVR(kernel='rbf')  # Gaussian RBF
gaussian_regressor_1 = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(10 ** (-13), 10 ** 13)))
forest_regressor = RandomForestRegressor(n_estimators=int(n_samples / 2))
gaussian_regressor_2 = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds=(10 ** (-13), 10 ** 13)))

coef = MyRBFcreate(S, Y)

index = np.array([])
for z in range(S.shape[0]):
    index = np.append(index, z)
index = index.astype("int")
index = index.reshape((len(index), 1))

index_train, index_test = train_test_split(index, test_size=0.2)
index_train = index_train.flatten()
index_test = index_test.flatten()

gaussian_regressor_1.fit(S[index_train], Y[index_train].ravel())
pred_1 = gaussian_regressor_1.predict(S[index_test]).reshape((len(index_test), 1))
nse_1 = r2_score(Y[index_test], pred_1)


svm_regressor.fit(S[index_train], Y[index_train].ravel())
pred_2 = svm_regressor.predict(S[index_test]).reshape((len(index_test), 1))
nse_2 = r2_score(Y[index_test], pred_2)


gaussian_regressor_2.fit(S[index_train], Y[index_train].ravel())
pred_4 = gaussian_regressor_2.predict(S[index_test]).reshape((len(index_test), 1))
nse_4 = r2_score(Y[index_test], pred_4)


forest_regressor.fit(S[index_train], Y[index_train].ravel())
pred_5 = forest_regressor.predict(S[index_test]).reshape((len(index_test), 1))
nse_5 = r2_score(Y[index_test], pred_5)


coef2 = MyRBFcreate(S[index_train], Y[index_train])
pred_3 = MyRBFpred(S[index_train], S[index_test], coef2)[0]
nse_3 = r2_score(Y[index_test], pred_3)

print(f"nse_1 = {nse_1}  nse_2 = {nse_2}  nse_3 = {nse_3}  nse_4 = {nse_4}  nse_5 = {nse_5}")


x_data = np.arange(lower_boundary, upper_boundary, step_inc)
y_data = np.arange(lower_boundary, upper_boundary, step_inc)
x_data, y_data = np.meshgrid(x_data, y_data)

z_data_real = np.zeros((x_data.shape[0], x_data.shape[1]))
z_data_gaussian_1 = np.zeros((x_data.shape[0], x_data.shape[1]))
z_data_gaussian_2 = np.zeros((x_data.shape[0], x_data.shape[1]))
z_data_svm = np.zeros((x_data.shape[0], x_data.shape[1]))
z_data_rf = np.zeros((x_data.shape[0], x_data.shape[1]))
z_data_rbf = np.zeros((x_data.shape[0], x_data.shape[1]))


for i in range(x_data.shape[0]):
    for j in range(x_data.shape[1]):
        z_data_real[i, j] = levy(np.array([x_data[i, j], y_data[i, j]]))
        z_data_gaussian_1[i, j] = GaussianProcess_EAS_1(np.array([x_data[i, j], y_data[i, j]]))
        z_data_gaussian_2[i, j] = GaussianProcess_EAS_2(np.array([x_data[i, j], y_data[i, j]]))
        z_data_svm[i, j] = SVM_EAS(np.array([x_data[i, j], y_data[i, j]]))
        z_data_rf[i, j] = RandomForest_EAS(np.array([x_data[i, j], y_data[i, j]]))
        z_data_rbf[i, j] = MyRBFpred_EAS(S[index_train], np.array([x_data[i, j], y_data[i, j]]).reshape(1, -1), coef2)[0]

n_rows = 2
n_columns = 3
colormap = cm.inferno
fontsize = 9

fig = plt.figure(figsize=plt.figaspect(1))
ax_real = fig.add_subplot(n_rows,n_columns,1, projection="3d")
ax_real.set_title("Response Surface of Levy Function", y=1.13, fontdict={'fontsize': fontsize})
surf_real = ax_real.plot_surface(x_data, y_data, z_data_real, cmap=colormap, linewidth=0)
fig.colorbar(surf_real, fraction=0.046, pad=0.1)


ax_gaussian_1 = fig.add_subplot(n_rows,n_columns,2, projection="3d")
ax_gaussian_1.set_title(f"Response Surface of Gaussian Process with Matérn Κernel\nNSE={nse_1:.3f}", y=1.09, fontdict={'fontsize': fontsize})
surf_gaussian_1 = ax_gaussian_1.plot_surface(x_data, y_data, z_data_gaussian_1, cmap=colormap, linewidth=0)
fig.colorbar(surf_gaussian_1, fraction=0.046, pad=0.1)


ax_svm = fig.add_subplot(n_rows,n_columns,3, projection="3d")
ax_svm.set_title(f"Response Surface of Support Vector Machine\nNSE={nse_2:.3f}", y=1.09, fontdict={'fontsize': fontsize})
surf_svm = ax_svm.plot_surface(x_data, y_data, z_data_svm, cmap=colormap, linewidth=0)
fig.colorbar(surf_svm, fraction=0.046, pad=0.1)

ax_rbf = fig.add_subplot(n_rows,n_columns,4, projection="3d")
ax_rbf.set_title(f"Response Surface of Cubic RBF\nwith Linear Polynomial Tail\nNSE={nse_3:.3f}", y=0.94, fontdict={'fontsize': fontsize})
surf_rbf = ax_rbf.plot_surface(x_data, y_data, z_data_rbf, cmap=colormap, linewidth=0)
fig.colorbar(surf_rbf, fraction=0.046, pad=0.1)


ax_gaussian_2 = fig.add_subplot(n_rows,n_columns,5, projection="3d")
ax_gaussian_2.set_title(f"Response Surface of Gaussian Process\nwith Rational Quadratic Kernel\nNSE={nse_4:.3f}", y=0.94, fontdict={'fontsize': fontsize})
surf_gaussian_2 = ax_gaussian_2.plot_surface(x_data, y_data, z_data_gaussian_2, cmap=colormap, linewidth=0)
fig.colorbar(surf_gaussian_2, fraction=0.046, pad=0.1)



ax_rf = fig.add_subplot(n_rows,n_columns,6, projection="3d")
ax_rf.set_title(f"Response Surface of Random Forest\nNSE={nse_5:.3f}", y=0.98, fontdict={'fontsize': fontsize})
surf_rf = ax_rf.plot_surface(x_data, y_data, z_data_rf, cmap=colormap, linewidth=0)
fig.colorbar(surf_rf, fraction=0.046, pad=0.1)

plt.tight_layout()
plt.show()
