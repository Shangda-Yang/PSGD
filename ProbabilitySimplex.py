from Projections import proj_prob_simplex
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(9)

rel = 100
T = 10000

n = 50

pStar = np.zeros(n)
pStar[0] = 1.

cMean = 5.
cVar = 1.

errL1 = np.zeros(T-1)

factor = 1.

# --------------------------------------------------------------------------------------- #
# simulations
for i in range(rel):
    p = np.random.rand(n)
    for t in range(1, T):
        alpha = 1/(factor*t)
        c = np.random.normal(cMean, cVar, n) + np.arange(n)
        # c = np.sort(c)
        p -= alpha*c
        p = proj_prob_simplex(p)
        eps = np.abs(np.dot(np.arange(n) + 5, (p-pStar)))
        errL1[t-1] += eps

errL1 /= rel

print('Finished')

# --------------------------------------------------------------------------------------- #
idx = np.nonzero(errL1)
rangeT = np.array([i for i in range(1, T+1)])
# fit the rate
print(np.polyfit(np.log(rangeT[idx]), np.log(errL1rangeT[idx]), 1)[0])
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)
ax.loglog(rangeT, errL1, 'k', linewidth=3)
ax.loglog([1, 1e4], [6e-2, 6e-6], 'k--')
ax.legend(["PSGD", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.grid()
ax.set_xlabel('t', fontsize=20)
ax.set_ylabel(r'$\mathbb{E}[|\bar{c}^Tp^* - \bar{c}^Tp_t|]$', fontsize=20)
ax.tick_params(axis='both', labelsize=15)
plt.savefig('./Figures/PeobabilitySimplex.pdf', format='pdf')
plt.show()
