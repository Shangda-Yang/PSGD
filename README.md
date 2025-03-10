# Exponential Concentration in Stochastic Approximation

This repository contains the tests for exponential decay when constraints are applied and the implementation of the examples in the paper "Exponential Concentration in Stochastic Approximation" (https://arxiv.org/abs/2208.07243).

Examples include a circle constraint problem, a three-spherical constraints problem, a non-negative ridge regression, a 2 variables linear programming problem, a 50 variables probability simplex problem, a multi-armed bandit problem, a three-state two-action Markov decision process, and a more complicated Markov decision process - the Blackjack problem introduced in Sutton and Barto (2018).

Before runing any code, create a folder under the current position named 'Figures' to save figures.

Projections applied:

For the probability simplex problem and the multi-armed bandit problem, the L1-projection algorithm (https://stanford.edu/~jduchi/projects/DuchiShSiCh08.html) is applied. This algorithm has been rewritten in Python and is involved in this repository.

For the linear programming, multi-armed bandit and MDP problems, Polyhedron Manipulation in Python (pypoman) package (https://scaron.info/doc/pypoman/) and cvxopt (http://cvxopt.org/) are applied. Install the packages first.

In particular, the Blackjack problem is constructed based on the OpenAI gym. Install the gym package before running the files in Blackjack.
