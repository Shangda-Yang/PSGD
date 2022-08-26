import numpy as np
from pypoman.projection import project_point_to_polytope
import matplotlib.pyplot as plt

# model
"""
       A
     /   \
    B  -  C
    States: s_1 (A), s_2 (B), s_3 (C) 
    Actions: a_1 (anti-clockwise), a_2 (clockwise) (1/3 probability go random)
            -> e.g. A go anti-clockwise 2/3 probability go to B, 
                    1/3 probability go to B, C or stay in A
    Costs: independent of actions 
           c(s_1, a_1) ~ N(1,1), c(s_1, a_2) ~ N(1,1)
           c(s_2, a_1) ~ N(2,1), c(s_2, a_2) ~ N(2,1)
           c(s_3, a_1) ~ N(5,1), c(s_3, a_2) ~ N(5,1)
"""
# number of state and actions
nState = 3
nAction = 2

# discount factor
beta = 0.02
# xi
xi = 1/3
# parameters of step factor
a = 0.1
b = 1
gamma = 1

# cost mean and variance
cM = [[1., 1.], [2., 2.], [5., 5.]]
cV = [[1., 1.], [1., 1.], [1., 1.]]

# movement matrix (deterministic)
pLeft = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])

pRight = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]])

pStay = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

# transition matrix
p = 1/3  # probability of random movement
PRandom = 1/3 * (pLeft + pRight + pStay)
PLeft = (1-p)*pLeft + p*PRandom
PRight = (1-p)*pRight + p*PRandom

# transpose and discount
PLeftTD = beta*np.transpose(PLeft)
PRightTD = beta*np.transpose(PRight)
# A and b matrix for projection
A = np.hstack(( (np.identity(3) - PLeftTD), (np.identity(3) - PRightTD)) )
A = np.vstack((A, -A))
pb = np.squeeze(np.vstack( (np.full((nState, 1), xi), np.full((nState, 1), -xi)) ) )
ineq = (A, pb)


x = np.random.random((nState, nAction))
B = 200 # batch size to evaluate c_t
c = np.zeros((nState, nAction))
piSA = 1/6
R = 20
# --------------------------------------------------------------------------------------- #
# Reference solution
# iterations
T_ref = 100000
x_ref = 0
for _ in range(R):
    for i in range(1, T_ref):
        alpha = a/((b+i)**gamma)
        SA = np.random.randint(low=[0, 0], high=[3, 2], size=(B, 2)) # iid (s, a) pairs
        for si, ai in SA:
            c[si][ai] += np.random.normal(cM[si][ai], cV[si][ai])/piSA
        c /= B
        x -= alpha*c
        # Ax = b
        x = np.squeeze(x.reshape((-1, nState*nAction)))
        x = project_point_to_polytope(x, ineq)
        # print(i)
        x = x.reshape((nState, nAction), order='F')
    x_ref += x/R
# --------------------------------------------------------------------------------------- #
# simulations
T = 10000
error = np.zeros((R, T))
xM = np.zeros((R, T))
# temp = 0

for j in range(R):
    x = np.random.random((nState, nAction))
    for i in range(1, T+1):
        alpha = a/((b+i)**gamma)
        SA = np.random.randint(low=[0, 0], high=[3, 2], size=(B, 2))  # iid (s, a) pairs
        for si, ai in SA:
            c[si][ai] += np.random.normal(cM[si][ai], cV[si][ai]) / piSA
        c /= B
        x -= alpha * c
        # Ax = b
        x = np.squeeze(x.reshape((-1, nState*nAction)))
        x = project_point_to_polytope(x, ineq)
        # print(j, i)
        x = x.reshape((nState, nAction), order='F')
        error[j][i-1] = np.abs(np.sum(np.multiply(cM, (x - x_ref))))
        # xM[j][i-1] = np.sum(np.multiply(cM, x))
        # temp = x

errL1 = np.sum(error, axis=0)/R
# errL1 = np.abs(np.sum(np.transpose(xM) - xM[:, -1], axis=1))/R

# --------------------------------------------------------------------------------------- #
idx = np.nonzero(errL1)
rangeT = np.array([i for i in range(1, T+1)])
# fit the rate
print(np.polyfit(np.log(rangeT[idx]), np.log(errL1[idx]), 1)[0])
# plot and save the figure
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)
ax.loglog(rangeT, errL1, 'k', linewidth=3)
ax.plot([1, 1e4], [1e-3, 1e-7], 'k--')
ax.legend(["PSGD", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.grid()
ax.set_xlabel('t', fontsize=20)
ax.set_ylabel('$\mathbb{E}[|c^Tx_t - c^Tx^*|]$', fontsize=20)
ax.tick_params(axis='both', labelsize=15)
plt.savefig("./Figures/3StateMDP.pdf", format="pdf")
plt.show()




