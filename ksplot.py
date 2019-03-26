import matplotlib.pyplot as plt
import numpy as np

def KS(g, rho=50.0):
    """
    Kreisselmeier-Steinhauser constraint aggregation function.
    """
    g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
    g_diff = g - g_max
    exponents = np.exp(rho * g_diff)
    summation = np.sum(exponents, axis=-1)[:, np.newaxis]

    KS = g_max + 1.0 / rho * np.log(summation)

    dsum_dg = rho * exponents
    dKS_dsum = 1.0 / (rho * summation)
    dKS_dg = dKS_dsum * dsum_dg

    dsum_drho = np.sum(g_diff * exponents, axis=-1)[:, np.newaxis]
    dKS_drho = dKS_dsum * dsum_drho

    return KS, dKS_dg.flatten()

def RePU(g, p = 2):
    """
    RePU (Rectified Polynomial Unit) aggregator.
    """
    y = np.zeros(g.shape)
    dy = np.zeros(g.shape)

    # select infeasible
    idx = np.where(g > 0)

    # apply RePU
    y[idx] = g[idx]**p
    dy[idx] = p * g[idx] ** (p - 1)

    # to aggregate: use sum(y)
    return y, dy

n = 101
k = 5
Z = np.zeros((n, n, 2))
X = np.linspace(-k, k, n)
Y = np.linspace(-k, k, n)

plt.subplot(121)
for i in range(n):
    for k in range(n):
        pt = np.array([X[i], Y[k]])
        result, deriv = KS(pt, 1)
        Z[i, k] = result


plt.quiver(Z[::5, ::5, 0], Z[::5, ::5, 1])
n = 101
k = 5
Z = np.zeros((n, n, 2))
X = np.linspace(-k, k, n)
Y = np.linspace(-k, k, n)


plt.subplot(122)
for i in range(n):
    for k in range(n):
        pt = np.array([X[i], Y[k]])
        result, deriv = RePU(pt, 2)
        Z[i, k] = result

plt.quiver(Z[::5,::5,0], Z[::5,::5,1])
plt.show()