# Minimise L1 Loss: f(x; (a, b)) = 1/2 | <a,x> - b |
# b_i = <a_i, x_true> + \xi_i, a_i ~ N(0, 1), \xi_i ~ N(0, 1)
# 1. non-negative least squares - \cX = \bbR_+^2
# x_true = (1, -1)
# 2. ridge regression - \cX = {x \in \bbR^2: ||x||^2 \leq \lambda}, \lambda > 0
# x_true = (1, 1), \lambda = 1

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def objective(x, a, b):
    # return np.abs(np.dot(a, x) - b)
    return 0.5 * (np.dot(a, x) - b) ** 2



# def sto_grad(x, a, b):
#     return a[(np.dot(a, x) - b >= 0)].sum(axis=0) - a[(np.dot(a, x) - b < 0)].sum(axis=0)


def sto_grad_2(x, a, b):
    return a * (np.dot(a, x) - b)

def sto_grad_LASSO(x, a, b, lam):
    return a[(np.dot(a, x) - b >= 0)].sum(axis=0) - a[(np.dot(a, x) - b < 0)].sum(axis=0) + lam * np.linalg.norm(x)

def sto_grad_obj(y, x, B):
    a = np.random.normal(0, 1, size=(B, 2))
    b = np.dot(a, x) + np.random.normal(0, 1, size=B)
    grad = a * (np.dot(a, x) - b)
    return np.dot(grad, y)

def cons1(x, C):
    return C ** 2 - (x[0] ** 2 + x[1] ** 2)

def cons2(x):
    return x[0]

def cons3(x):
    return x[1]

# step size parameter
a = 1
b = 1
gamma = 1

# number of iterations
T = 100000

# number of realisations
R = 20
B = 1
lam = 0.9

y = np.array([1., -1.])
ref = np.array([lam ** 0.5, 0.])

# y = np.array([1, 1])
# ref = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
cons = [{'type': 'ineq', 'fun': cons1, 'args': (lam ** 0.5,)},
        {'type': 'ineq', 'fun': cons2},
        {'type': 'ineq', 'fun': cons3}]

rates = []
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)
for alg in range(1,3):
    err = np.zeros((R, T))

    for i in range(R):

        print('Realisation:', i)
        print('------------------------------')

        # x = init_gen_circle(C, cX)
        # x = np.random.uniform(0, 1, size=(2,))
        # x = xfix
        x = np.array([2., 2.])
        for j in range(T):

            print('Iteration:', j)
            alphat = a / ((b + j) ** gamma)

            # 1. non-negative least square

            if alg == 0:
                # Frank-Wolfe
                # vt = np.array([0.5*h/(g[0][0] + g[0][0]*g[0][1]**2 + g[0][0]**3),
                #       0.5*h/(g[0][1] + g[0][1]*g[0][0]**2 + g[0][1]**3)])
                # vt = np.array(fw(x, g, h, lam))
                vt = sp.optimize.minimize(sto_grad_obj, x, args=(x, B), constraints=cons, method='COBYLA')
                print(vt['x'])
                x = (1 - alphat) * x + alphat * vt['x']
            elif alg == 1:
                # PSGD
                g = np.random.normal(0, 1, size=(B, 2))
                h = np.dot(g, y) + np.random.normal(0, 1, size=B)
                ct = sto_grad_2(x, g, h)
                x -= alphat * ct[0]
            elif alg == 2:
                # Keifer-Wolfowitz
                v = 1
                g1 = np.random.normal(0, 1, size=(B, 2))
                h1 = np.dot(g1, y) + np.random.normal(0, 1, size=B)
                g2 = np.random.normal(0, 1, size=(B, 2))
                h2 = np.dot(g2, y) + np.random.normal(0, 1, size=B)
                ct = np.zeros(np.shape(x))
                ct[0] = (objective(x + [v, 0], g1, h1) - objective(x - [v, 0], g2, h2))/(2 * v)
                ct[1] = (objective(x + [0, v], g1, h1) - objective(x - [0, v], g2, h2))/(2 * v)
                # print(ct)
                x -= alphat * ct
            # print(x)
            # projection
            # if x[0] < 0:
            #     x[0] = 0
            # if x[1] < 0:
            #     x[1] = 0

            # # 2. LASSO
            # g = np.random.normal(0, 1, size=(B, 2))
            # h = np.dot(g, y) + np.random.normal(0, 1, size=B)
            # ct = sto_grad_LASSO(x, g, h, lam)
            # x -= alphat * ct

            # projection
            if alg == 1 or alg == 2:
                if x[0] < 0:
                    x[0] = 0
                if x[1] < 0:
                    x[1] = 0
                if x[0] ** 2 + x[1] ** 2 > lam:
                    unit = np.linalg.norm(x)
                    x[0] = (lam ** .5) * (x[0]) / unit
                    x[1] = (lam ** .5) * (x[1]) / unit
            # print(x)

            err[i][j] = np.linalg.norm(x - ref)

        print('------------------------------')

    errL1 = np.sum(err, axis=0) / R
    idx = np.nonzero(errL1)
    rangeT = np.array([i for i in range(1, T + 1)])
# fit the rate

    rate = np.polyfit(np.log(rangeT[idx]), np.log(errL1[idx]), 1)[0]
    rates.append(rate)
# plot and save the figure

    ax.loglog(rangeT, errL1, 'k', linewidth=3, alpha=1.-(alg-1)/2)

ax.plot([1, 1e5], [5e-1, 5e-6], 'k--')
# ax.legend(["Frank-Wolfe", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.legend(["PSGD", "Kiefer-Wolfowitz", "$\mathcal{O}(t^{-1})$"], fontsize=20)
# ax.legend(["Kiefer-Wolfowitz", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.grid()
ax.set_xlabel('t', fontsize=20)
ax.set_ylabel(r'$\mathbb{E}[|\ell(x_t) - \ell(x^*)|]$', fontsize=20)
ax.tick_params(axis='both', labelsize=15)
# plt.savefig('./Figures/FWB{}Rate{:.2f}.pdf'.format(B, rate), format='pdf')
# plt.savefig('./Figures/PSGDRidgeB{}Rate{:.2f}.pdf'.format(B, rate), format='pdf')
# plt.savefig('./Figures/KWv{}Rate{:.2f}.pdf'.format(v, rate), format='pdf')
plt.savefig('./Figures/Regression.pdf', format='pdf')
plt.show()

print(rates)

