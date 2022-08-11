# Approximate Probabilistic Reachability Calculation in Large γ-discounted MDP Using Regressor and Tree Search

Steps to reproduce the result:
1. Run appr_reachability_soccer_prepare.py to generate P(s, a, s') for all actions. The generated files are PR_a[n].npz
2. Run appr_reachability_soccer_V_truth.py to generate the V*(s), the ground truth of reachability from initial state s. It should take about 23 iterations (~2 minutes)
3. Run appr_reachability_soccer_lr.py to obtain a Linear Regressor of the reachability value function, V^k(s). 500 iterations takes about 14 minutes
4. Run appr_reachability_soccer_aqts.py to see how Adaptive Q-value Tree Search V^(k+n) can reduce the error in V^k(s). 500 AQTS (each has 50000 search, ~10 levels of tree depth) takes about 160 minutes.
