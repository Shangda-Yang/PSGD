import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_toolkits.mplot3d as Axes3D


# R^3 example
# l(x) = E_g[-x_1 + <g, x>], g ~ N(0,I)
# Constraints: intersection of three balls

# radius of the balls
r1 = 2
r2 = 2
r3 = 2

# center of the balls
c1 = np.array([1., 0., 0.])
c2 = np.array([-1., 0., 0.])
c3 = np.array([0., 1., 0.])

R = [r1, r2, r3]
C = [c1, c2, c3]
list_center = [c1, c2, c3]
list_radius = [r1, r2, r3]
#
#
# def plt_sphere(list_center, list_radius):
#     ax = fig.add_subplot(projection='3d')
#     for c, r in zip(list_center, list_radius):
#
#         # draw sphere
#         u = np.linspace(0, 2 * np.pi, 100)
#         v = np.linspace(0, 2 * np.pi, 100)
#         x = r * np.outer(np.cos(u), np.sin(v))
#         y = r * np.outer(np.sin(u), np.sin(v))
#         z = r * np.outer(np.ones(np.size(u)), np.cos(v))
#         ax.plot_surface(x-c[0], y-c[1], z-c[2], color='gray', alpha=0.1)
#
#     ax.scatter(0, 0, np.sqrt(3), marker='o', color='k', s=50)
#
#     start, end = ax.get_xlim()
#     ax.xaxis.set_ticks(np.arange(start, end, 1))
#     ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
#     ax.yaxis.set_ticks(np.arange(start, end, 1))
#     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
#     ax.zaxis.set_ticks(np.arange(start, end, 1))
#     ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
#     ax.set_xlabel(r'$x_1$')
#     ax.set_ylabel(r'$x_2$')
#     ax.zaxis.set_rotate_label(False)
#     ax.set_zlabel(r'$x_3$', rotation=0)
#     ax.set_aspect('equal')
#
# #
# #
# fig = plt.figure()
# plt_sphere(list_center, list_radius)
# plt.savefig('Figures/ThreeBallsCon.pdf', format='pdf')
# plt.show()
from numpy import sqrt, dot, cross
from numpy.linalg import norm

# Find the intersection of three spheres
# P1,P2,P3 are the centers, r1,r2,r3 are the radii
# def trilaterate(P1, P2, P3, r1, r2, r3):
#     temp1 = P2-P1
#     e_x = temp1/norm(temp1)
#     temp2 = P3-P1
#     i = dot(e_x,temp2)
#     temp3 = temp2 - i*e_x
#     e_y = temp3/norm(temp3)
#     e_z = cross(e_x,e_y)
#     d = norm(P2-P1)
#     j = dot(e_y,temp2)
#     x = (r1*r1 - r2*r2 + d*d) / (2*d)
#     y = (r1*r1 - r3*r3 -2*i*x + i*i + j*j) / (2*j)
#     temp4 = r1*r1 - x*x - y*y
#     if temp4<0:
#         raise Exception("The three spheres do not intersect!");
#     z = sqrt(temp4)
#     p_12_a = P1 + x*e_x + y*e_y + z*e_z
#     p_12_b = P1 + x*e_x + y*e_y - z*e_z
#     return p_12_a, p_12_b
#
# x, y = trilaterate(c1, c2, c3, r1, r2, r3)

# projection using Bregman’s cyclic algorithm
def bregman(f0, R, C):
    check = True
    for r, c in zip(R, C):
        if (f0[0] - c[0]) ** 2 + (f0[1] - c[1]) ** 2 + (f0[2] - c[2]) ** 2 > r ** 2:
            check = False
    if check == False:
        for r, c in zip(R, C):
            unit = np.linalg.norm(f0 - c)
            f0[0] = c[0] + r*(f0[0] - c[0])/unit
            f0[1] = c[1] + r*(f0[1] - c[1])/unit
            f0[2] = c[2] + r*(f0[2] - c[2])/unit
    return f0


def objective(x):
    return -x[0] + np.dot(np.random.normal(0, 1, size=np.size(x)), x)

v = 10
def KW_grad(x, v):
    ct = np.zeros(np.shape(x))
    ct[0] = (objective(x + [v, 0., 0.]) - objective(x - [v, 0., 0.])) / (2 * v)
    ct[1] = (objective(x + [0., v, 0.]) - objective(x - [0., v, 0.])) / (2 * v)
    ct[2] = (objective(x + [0., 0., v]) - objective(x - [0., 0., v])) / (2 * v)
    return ct

def sto_grad(x, B):
    # B = max(int(1 / alphat), 10)
    ct = np.random.normal(0, 1, size=(B, np.size(x)))
    ct = np.sum(ct, axis=0) / B
    ct[0] -= 1
    return ct

# FW grad
def argmaxx(x, B, C):
    ct = np.random.normal(0, 1, size=(B, np.size(x)))
    ct = np.sum(ct, axis=0) / B
    ct[0] -= 1
    return -C * ct / np.linalg.norm(ct)

T = 800
M = 20

a = 1
b = 1
gamma = 1
B = 10

ref = np.array([0., 0., np.sqrt(3)])
err = np.zeros((M, T))

t = 20
rates = np.zeros(2)
fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=6)
for alg in range(2):
    for i in range(M):
        print('Realisation:', i)
        print('------------------------------')
        alphat = 0.01
        k = 1
        # x = np.random.normal(0, 10, size=3)
        x = np.array([1., 1., 1.])
        for j in range(T):
            print('Iteration:', j)
            # alphat = a / ((b + j) ** gamma)
            if k == t:
                alphat /= 2
                k = 1
            else:
                k += 1

            if alg == 0:
                # SGD
                ct = sto_grad(x, B)
                x -= alphat * ct
            else:
                # Keifer-Wolfowitz
                x -= alphat * KW_grad(x, v)
                # print(KW_grad(x, v))
            # projection
            x = bregman(x, R, C)
            # print(x)
            err[i][j] = np.linalg.norm(x[0] - ref[0])
        print('------------------------------')

    errL1 = np.sum(err, axis=0) / M

    idx = np.nonzero(errL1)
    rangeT = np.array([i for i in range(1, T + 1)])

    # fit the rate
    # rate = np.polyfit(np.log(rangeT[idx]), np.log(errL1[idx]), 1)[0]
    rate = np.polyfit(rangeT[idx], np.log(errL1[idx]), 1)[0]
    rates[alg] = rate
    print(rate)

    # plot and save the figure
    # plt.yscale("log")

    ax.plot(rangeT, errL1, 'k', linewidth=3, alpha=1. - (alg/2))
    ax.set_yscale('log')
    ax.set_xscale('linear')
    # ax.loglog(rangeT, errL1, 'k', linewidth=3, alpha=1. - (alg/2))
    # ax.legend(["Kiefer-Wolfowitz", "$\mathcal{O}(t^{-1})$"], fontsize=20)
    # ax.legend(["PSGD", "$\mathcal{O}(t^{-1})$"], fontsize=20)
    # ax.legend(["PSGD", "Linear Convergence"], fontsize=20)

ax.plot([1, 800], [np.e**(-0.035 * 1), np.e**(-0.035 * 800)], 'k--')
# ax.plot([1, 1e4], [1e-1, 1e-5], 'k--')
ax.grid()
ax.set_xlabel('t', fontsize=20)
ax.set_ylabel(r'$\mathbb{E}[|\ell(x_t) - \ell(x^*)|]$', fontsize=20)
ax.tick_params(axis='both', labelsize=15)
# ax.legend(["PSGD", "Kiefer-Wolfowitz", "$\mathcal{O}(t^{-1})$"], fontsize=20)
ax.legend(["PSGD", "Kiefer-Wolfowitz", "$\mathcal{O}(e^{-0.034*t})$"], fontsize=20)
# plt.savefig('./Figures/KWThreeBallsV{}Rate{:.2f}.pdf'.format(v, rate), format='pdf')
# plt.savefig('./Figures/PSGDThreeBalls{}Rate{:.2f}.pdf'.format(B, rate), format='pdf')
# plt.savefig('./Figures/ThreeBallsAlgs.pdf', format='pdf')
# plt.savefig('./Figures/PSGDHalfAlphaThreeBalls{}Rate{:.2f}.pdf'.format(B, rate), format='pdf')
plt.savefig('./Figures/HalfAlphaThreeBalls.pdf', format='pdf')
plt.show()

print(rates)