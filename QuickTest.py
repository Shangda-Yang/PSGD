import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt

# test for exponential decay when constraints applied
T = 100000
alpha = 0.3
eps = 2.

initial = 1
x = [initial]
xp = [initial]

for i in range(1, T):
    x.append(x[i-1] + np.random.normal(-2*alpha*(x[i-1] + 1), alpha**2 * eps**2))

for i in range(1, T):
    temp = xp[i-1] + np.random.normal(-2*alpha*(xp[i-1] + 1), alpha**2 * eps**2)
    if temp < 0:
        temp = 0
        xp.append(temp)
    else:
        xp.append(temp)

x = np.array(x)
xp = np.array(xp)

# normal
fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
fig1.tight_layout(pad=6)
ax2.hist(x, bins=60, density=True, color='grey', edgecolor='k')
mean1, std1 = norm.fit(x)
xmin1, xmax1 = plt.xlim()
xnorm = np.linspace(xmin1, xmax1, 100)
y1 = norm.pdf(xnorm, mean1, std1)
ax2.plot(xnorm, y1, 'k', linewidth=2, label=r'$N(-1,\sigma^2)$')
ax2.grid()
ax2.set_xlabel('x', fontsize=20)
ax2.set_ylabel('Density', fontsize=20)
ax2.tick_params(axis='both', labelsize=15)
ax2.legend(fontsize=20, loc='upper right')

ax1.plot(xnorm, (xnorm + 1)**2, 'k', linewidth=2, label='$y=(x+1)^2$')
ax1.grid()
ax1.set_xlabel('x', fontsize=20)
ax1.set_ylabel('y', fontsize=20)
ax1.tick_params(axis='both', labelsize=15)
ax1.legend(fontsize=20, loc='upper right')
plt.savefig('./Figures/Normal.pdf', format='pdf')
plt.show()

# exponential

# fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
# fig2.tight_layout(pad=6)
# ax2[1].hist(xp[xp != 0], bins=60, density=True, color='grey', edgecolor='k')
# mean2, std2 = expon.fit(xp[xp!=0])
# xmin2, xmax2 = plt.xlim()
# xmin2 = 1e-3
# xexp = np.linspace(xmin2, xmax2, 100)
# y2 = expon.pdf(xexp, scale=std2)
# ax2[1].plot(xexp, y2, 'k', linewidth=2, label=r'$\lambda\exp(-\lambda x)$')
# ax2[1].grid()
# ax2[1].set_xlabel('x', fontsize=20)
# ax2[1].set_xlim([-0.2, 2.7])
# ax2[1].set_ylabel('Density', fontsize=20)
# ax2[1].tick_params(axis='both', labelsize=15)
# ax2[1].legend(fontsize=20, loc='upper right')

# xleft = np.linspace(-1, xmin2, 100)
# ax2[0].plot(xexp, (xexp + 1)**2, '-k', linewidth=2, label='$y=(x+1)^2$')
# # ax2[0].plot(xleft, (xleft + 1)**2, '--k', linewidth=2, label='$y=(x+1)^2$')
# ax2[0].axvline(x=0, color='dimgray', linewidth=3)
# ax2[0].grid()
# ax2[0].set_xlabel('x', fontsize=20)
# ax2[0].set_xlim([-0.2, 2.7])
# ax2[0].set_ylabel('y', fontsize=20)
# ax2[0].tick_params(axis='both', labelsize=15)
# ax2[0].legend(fontsize=20, loc='upper right')
# # plt.savefig('./Figures/ExpStart0.pdf', format='pdf')
# plt.show()

fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
fig2.tight_layout(pad=6)
ax2[1].hist(xp[xp != 0], bins=60, density=True, color='grey', edgecolor='k')
mean2, std2 = expon.fit(xp[xp!=0])
xmin2, xmax2 = plt.xlim()
xmin2 = 1e-3
xexp = np.linspace(xmin2, xmax2, 100)
y2 = expon.pdf(xexp, scale=std2)
ax2[1].plot(xexp, y2, 'k', linewidth=2, label=r'$\lambda\exp(-\lambda x)$')
ax2[1].grid()
ax2[1].set_xlabel('x', fontsize=20)
ax2[1].set_xlim([-1.2, 1.6])
ax2[1].set_ylabel('Density', fontsize=20)
ax2[1].tick_params(axis='both', labelsize=15)
ax2[1].legend(fontsize=20, loc='upper right')

xleft = np.linspace(-1, xmin2, 100)
ax2[0].plot(xexp, (xexp + 1)**2, '-k', linewidth=2, label='$y=(x+1)^2$')
ax2[0].plot(xleft, (xleft + 1)**2, '--k', linewidth=2, label='$y=(x+1)^2$')
ax2[0].axvline(x=0, color='dimgray', linewidth=3)
ax2[0].grid()
ax2[0].set_xlabel('x', fontsize=20)
ax2[0].set_xlim([-1.2, 1.6])
ax2[0].set_ylabel('y', fontsize=20)
ax2[0].tick_params(axis='both', labelsize=15)
ax2[0].legend(fontsize=20, loc='upper right')
plt.savefig('./Figures/Exp.pdf', format='pdf')
plt.show()

