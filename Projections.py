import numpy as np


def proj_prob_simplex(q):
    ''' Find the L_2 projection onto the probability simplex

    # Input -- numpy array
    # Output -- numpy array on prob. simplex
    '''
    q = np.array(q)
    n = len(q)
    mu = -np.sort(-q)
    sum_mu = 0
    for j in range(len(mu)):
        sum_mu += mu[j]
        if (j + 1) * mu[j] - sum_mu + 1 > 0:
            rho = (j + 1)
            sum_rho = sum_mu
    theta = (sum_rho - 1) / rho

    proj_q = np.max((q - theta * np.ones(n), np.zeros(n)), axis=0)

    return proj_q
