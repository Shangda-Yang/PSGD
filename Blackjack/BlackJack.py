import gym
import numpy as np
from gym.envs.toy_text.blackjack import draw_card, sum_hand, usable_ace, cmp, is_bust, deck
from DealerPlayerFunctions import P
from pypoman.projection import project_point_to_polytope
import matplotlib.pyplot as plt

# create an environment,
# natural means if giving an extra 0.5 point for a natural
# sab means if following the exact rules by Sutton and Barto
env = gym.make('Blackjack-v1', natural=False, sab=True)

# number of states and actions
nState = len(P)
nAction = len(P[0])

# discount factor
beta = 1
# xi
xi = 0.3
# parameters of step factor
a = 0.1
b = 1
gamma = 1

# reward and variance
cM = np.zeros((nState, nAction))
cV = np.ones((nState, nAction))

for i in range(len(P)):
    p = P[i][0]
    for tp in p:
        cM[i][0] += tp[0]*tp[2]/len(p)

for i in range(len(P)):
    p = P[i][1]
    for tp in p:
        cM[i][1] += tp[0]*tp[2]/len(p)

cM = -cM

# transition matrix
Ph = np.zeros((len(P), len(P)))
Ps = np.zeros((len(P), len(P)))

for i in range(len(P)):
    for j in range(len(P[i][1])):
        Ph[i][P[i][1][j][1]] = P[i][1][j][0]

for i in range(len(P)):
    for j in range(len(P[i][0])):
        Ps[i][P[i][0][j][1]] = P[i][0][j][0]

# A and b matrix for projection
A = np.hstack(( (np.identity(nState) - Ph), (np.identity(nState) - Ps)) )
A = np.vstack((A, -A))
pb = np.squeeze(np.vstack( (np.full((nState, 1), xi), np.full((nState, 1), -xi)) ) )
ineq = (A, pb)

B = 200 # batch size to evaluate c_t
c = np.zeros((nState, nAction))
piSA = 1/580
R = 10
# --------------------------------------------------------------------------------------- #
# Reference solution
# iterations
T_ref = 10000
x_ref = 0
for j in range(R):
    x = np.random.random((nState, nAction))
    for i in range(1, T_ref):
        alpha = a/((b+i)**gamma)
        SA = np.random.randint(low=[0, 0], high=[nState, nAction], size=(B, 2)) # iid (s, a) pairs
        for si, ai in SA:
            c[si][ai] += np.random.normal(cM[si][ai], cV[si][ai])/piSA
        c /= B
        x -= alpha*c
        # Ax = b
        x = np.squeeze(x.reshape((-1, nState*nAction)))
        x = project_point_to_polytope(x, ineq)
        # print(i)
        x = x.reshape((nState, nAction), order='F')
        print("Realisations:{}, Step:{}".format(j, i))
    x_ref += x/R

print("Reference finished.")
# --------------------------------------------------------------------------------------- #
# simulations
T = 1000
error = np.zeros((R, T))
xM = np.zeros((R, T))
temp = 0

for j in range(R):
    x = np.random.random((nState, nAction))
    for i in range(1, T+1):
        alpha = a/((b+i)**gamma)
        SA = np.random.randint(low=[0, 0], high=[nState, nAction], size=(B, 2))  # iid (s, a) pairs
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
        print("Realisations: {}, Iterations: {}".format(j, i))

print("Finished.")

errL1 = np.sum(error, axis=0)/R
# errL1 = np.abs(np.sum(np.transpose(xM) - np.sum(xM[:, -1])/R, axis=1))/R
# --------------------------------------------------------------------------------------- #
# t = 1000
# errL1 = errL1[:t]
idx = np.nonzero(errL1)
rangeT = np.array([i for i in range(1, T+1)])
# fit the rate
print(np.polyfit(np.log(rangeT[idx]), np.log(errL1[idx]), 1)[0])
# plot and save the figure
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)
ax.loglog(rangeT, errL1, 'k', linewidth=3)
ax.plot([1, 1e3], [1e-2, 1e-5], 'k--')
ax.legend(["PSGD", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.grid()
ax.set_xlabel('t', fontsize=20)
ax.set_ylabel('$\mathbb{E}[|c^Tx_t - c^Tx^*|]$', fontsize=20)
ax.tick_params(axis='both', labelsize=15)
plt.savefig("../Figures/Blackjack.pdf", format="pdf")
plt.show()