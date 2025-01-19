import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# l(x) = ||x - x^*||
# \nable l(x) = (x - x^*)/||x - x^*||


# step size parameter

a = 1
b = 1
gamma = 1

# number of iterations

T = 10000

# number of realisations
R = 10

# radius of circular constraint
C = 15

# center of the circle
cX = np.array([0., 0.])

# optimal solution
x_star = np.array([1., 1.])

# plot the constraints
# circle = plt.Circle((0, 0), C, color='grey', alpha=0.5)
# fig, ax = plt.subplots()
# ax.add_patch(circle)
# ax.scatter(1, 1, marker='o', color='k', s=20)
# ax.set_xlim((-20, 20))
# ax.set_ylim((-20, 20))
# ax.set_aspect('equal')
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.savefig('./Figures/CircleCons.pdf', format='pdf')
# plt.show()

# compute the objective
def objective(x, x_star):
    return np.linalg.norm(x - x_star)


# compute the gradient of l(x)
# def sto_grad_obj(y, x, x_star, alphat):
#     # batch size for cost
#     B = int(1000*(3*1/alphat))
#     mean_val = (x - x_star) / objective(x, x_star)
#     ct = np.random.normal(mean_val, np.ones(np.shape(mean_val)), size=(B, 2))
#     ct = np.sum(ct, axis=0)/B
#     return np.dot(ct, y)


def argmaxx(x, x_star, alphat, B, C):
    # batch size for cost
    # B = max(int(1 / alphat), 10)
    mean_val = (x - x_star) / objective(x, x_star)
    ct = np.random.normal(mean_val, np.ones(np.shape(mean_val)), size=(B, 2))
    ct = np.sum(ct, axis=0) / B
    return -C * ct / np.linalg.norm(ct)


# constraints

def cons(x, C):
    return C ** 2 - (x[0] ** 2 + x[1] ** 2)

def KW_grad(x, v, x_star):
    ct = np.zeros(np.shape(x))
    w1 = np.random.normal(0, 0.01)
    w2 = np.random.normal(0, 0.01)
    ct[0] = (objective(x + [v, 0], x_star) + w1
             - objective(x - [v, 0], x_star) - w2) / (2 * v)
    ct[1] = (objective(x + [0, v], x_star) + w1
             - objective(x - [0, v], x_star) - w2) / (2 * v)
    return ct

v = 0.1
cons = {'type': 'ineq', 'fun': cons, 'args': (C,)}


# initial value generator

def init_gen_circle(C, cX):
    r = C * np.sqrt(np.random.uniform())
    theta = 2 * np.pi * np.random.uniform()
    x1 = cX[0] + r * np.cos(theta)
    x2 = cX[1] + r * np.sin(theta)
    return np.array([x1, x2])

B = 10
rates = np.zeros(3)
# plot and save the figure
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)

for alg in range(3):
    xfix = init_gen_circle(C, cX)
    err = np.zeros((R, T))
    for i in range(R):

        print('Realisation:', i)
        print('------------------------------')
        # x = init_gen_circle(C, cX)
        # x = np.random.uniform(0, 1, size=(2,))
        # x = xfix
        x = np.array([0., 0.])

        for j in range(T):
            print('Iteration:', j)
            alphat = a / ((b + j) ** gamma)
            if alg == 0:
                # Frank-Wolfe
                #     vt = sp.optimize.minimize(sto_grad_obj, init_gen_circle(C, cX), args=(x, x_star, alphat), constraints=cons, method='L-BFGS-B')
                #     print(vt['x'])
                #     x = (1 - alphat) * x + alphat * vt['x']
                vt = argmaxx(x, x_star, alphat, B, C)
                x = (1 - alphat) * x + alphat * vt
            elif alg == 1:
                # unconstrained SGD
                # B = max(int(1 / alphat), 10)
                mean_val = (x - x_star) / objective(x, x_star)
                ct = np.random.normal(mean_val, np.ones(np.shape(mean_val)), size=(B, 2))
                ct = np.sum(ct, axis=0) / B
                x -= alphat * ct
            else:
                # Keifer-Wolfowitz
                x -= alphat * KW_grad(x, v, x_star)
            # print(x)
            err[i][j] = np.abs(objective(x, x_star))
        print('------------------------------')

    errL1 = np.sum(err, axis=0) / R

    idx = np.nonzero(errL1)
    rangeT = np.array([i for i in range(1, T + 1)])

    # fit the rate
    rate = np.polyfit(np.log(rangeT[idx]), np.log(errL1[idx]), 1)[0]
    rates[alg] = rate

    ax.loglog(rangeT, errL1, 'k',linewidth=3, alpha=1. - (alg/3))
    # ax.legend(["Frank-Wolfe", "$\mathcal{O}(t^{-1})$"], fontsize=20)
    # ax.legend(["PSGD", "$\mathcal{O}(t^{-1})$"], fontsize=20)
    # ax.legend(["Kiefer-Wolfowitz", "$\mathcal{O}(t^{-1})$"], fontsize=20)
    ax.grid()
    ax.set_xlabel('t', fontsize=20)
    ax.set_ylabel(r'$\mathbb{E}[|\ell(x_t) - \ell(x^*)|]$', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
# plt.savefig('./Figures/FWCircleB{}Rate{:.2f}.pdf'.format(B, rate), format='pdf')
# plt.savefig('./Figures/PSGDCircleB{}Rate{:.2f}.pdf'.format(B, rate), format='pdf')
# plt.savefig('./Figures/KWCirclev{}Rate{:.2f}.pdf'.format(v, rate), format='pdf')
ax.plot([1, 1e4], [1e-1, 1e-5], 'k--')
ax.legend(["Frank-Wolfe", "PSGD", "Kiefer-Wolfowitz", "$\mathcal{O}(t^{-1})$"], fontsize=20)
# plt.savefig('./Figures/CircleEp.pdf', format='pdf')
plt.show()

print(rates)