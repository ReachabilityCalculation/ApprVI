# Approximate Probabilistic Reachability Calculation in Large γ-discounted MDP Using Regressor and Tree Search

Steps to reproduce the result:
1. Run appr_reachability_soccer_prepare.py to generate P(s, a, s') for all actions. The generated files are PR_a[n].npz
2. Run appr_reachability_soccer_V_truth.py to generate the V*(s), the ground truth of reachability from initial state s. It should take about 23 iterations (~2 minutes)
