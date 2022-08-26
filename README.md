# Convergence Rates for Stochastic Approximation on a Boundary

This repository contains the implementation for the examples in the paper "Convergence Rates for Stochastic Approximation on a Boundary" (https://arxiv.org/abs/2208.07243).

Examples include a 2 variables linear programming problem, a 50 variables probability simplex problem, a multi-armed bandit problem, a three-state two-action Markov decision process and a more complicated Markov decision process - the Blackjack problem introduced in Sutton and Barto (2018).

Projections applied:

For the probability simplex problem, L1-projection algorithm (https://stanford.edu/~jduchi/projects/DuchiShSiCh08.html) is applied. This algorithm is rewritten in Python and involved in this repository.

For the linear programming, multi-armed bandit and MDP problems, Polyhedron Manipulation in Python (pypoman) package (https://scaron.info/doc/pypoman/) and cvxopt (http://cvxopt.org/) are applied. Install the packages first.

In particular, the Blackjack problem is constructed based on the OpenAI gym. Install the gym package before running the files in Blackjack.
