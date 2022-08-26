import numpy as np
import pypoman
from pypoman.projection import project_point_to_polytope
import matplotlib.pyplot as plt
from numpy import arange, array, cos, pi, sin
import pylab

# vertices
vertices = np.array([(1., 3.), (2., 1.),
                     (2., 5.), (4., 6.),
                     (6., 5.), (6., 3.),
                     (5., 1.5)])

# generate A matrix and b vector
A, bv = pypoman.compute_polytope_halfspaces(vertices)
ineq = (A, bv)
# mean cost
C = np.array([4, 6])

# minimum x
vCost = vertices.dot(C)
minX = vertices[np.argmin(vCost)]

# step size parameter
a = 0.5
b = 1
gamma = 1

# number of iterations
T = 1000
# batch size for costs
B = 5
# number of realisations
R = 1000

err = np.zeros((R, T))
for j in range(R):
    # starting point
    x = np.random.normal([3, 3], [1, 1], size=2)
    for i in range(T):
        alphat = a/((b+i)**gamma)
        ct = np.transpose(np.random.normal([C[0], C[1]], [5, 5], size=(B, 2)))
        ct = np.sum(ct, axis=1)/B

        x -= alphat * ct

        # plot the polytope and projections

        # pylab.ion()
        # pylab.figure(figsize=(7, 7))
        # pylab.gca().set_aspect("equal")
        # pypoman.plot_polygon(vertices,color='grey')
        # pylab.plot([x[0]], [x[1]], marker='x', markersize=6, color='k')
        # point = x
        x = project_point_to_polytope(x, ineq)

        # pylab.plot([x[0]], [x[1]], marker='o', markersize=5, color='k')
        # pylab.plot([point[0], x[0]], [point[1], x[1]], 'k--')
        # pylab.xlim([0, 8])
        # pylab.ylim([0, 8])
        # pylab.xticks(list(range(9)))
        # pylab.yticks(list(range(9)))
        # pylab.tick_params(axis='both', labelsize=15)
        # pylab.xlabel("$x_1$", fontsize=20)
        # pylab.ylabel("$x_2$", fontsize=20)
        # # pylab.savefig('./Figures/LPPolytope.pdf', format='pdf')
        # pylab.show()

        err[j][i] = np.abs(np.dot(C, x)-np.dot(C, minX))

errL1 = np.sum(err, axis=0)/R

# --------------------------------------------------------------------------------------- #
idx = np.nonzero(errL1)
rangeT = np.array([i for i in range(1, T+1)])
# fit the rate
print(np.polyfit(np.log(rangeT[idx]), np.log(errL1[idx]), 1)[0])
# plot and save the figure
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)
ax.loglog(rangeT, errL1, 'k', linewidth=3)
ax.plot([1, 1e3], [5e-2, 5e-5], 'k--')
ax.legend(["PSGD", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.grid()
ax.set_xlabel('t', fontsize=20)
ax.set_ylabel(r'$\mathbb{E}[|c^Tx_t - c^Tx^*|]$', fontsize=20)
ax.tick_params(axis='both', labelsize=15)
plt.savefig('./Figures/LinearProgramsPSGDB{}.pdf'.format(B), format='pdf')
plt.show()







