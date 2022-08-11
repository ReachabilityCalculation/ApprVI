from scipy import sparse
import numpy as np

p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = 0,0,0,0,0,0,0,0,0

def state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball):
    return p1_x * 1200500 + p1_y * 171500 + p2_x * 34300 + p2_y * 4900 \
           + p3_x * 980 + p3_y * 140 + p4_x * 28 + p4_y * 4 + ball * 1 + 2

def id_to_state(id):
    id = id -2
    p1_x = id // 1200500
    id = id % 1200500
    p1_y = id // 171500
    id = id % 171500
    p2_x = id // 34300
    id = id % 34300
    p2_y = id // 4900
    id = id % 4900
    p3_x = id // 980
    id = id % 980
    p3_y = id // 140
    id = id % 140
    p4_x = id // 28
    id = id % 28
    p4_y = id // 4
    ball = id % 4
    return p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball

N = 6002500 # total N non-final states
DIFF_THRESHOLD = 1e-6

# list of actions: shoot, p1_down, p1_right, p2_down, p2_right, p3_down, p3_right, p4_down, p4_right, pass_1, pass_2, pass_3
PR_a0 = sparse.load_npz('PR_a0.npz')
PR_a1 = sparse.load_npz('PR_a1.npz')
PR_a2 = sparse.load_npz('PR_a2.npz')
PR_a3 = sparse.load_npz('PR_a3.npz')
PR_a4 = sparse.load_npz('PR_a4.npz')
PR_a5 = sparse.load_npz('PR_a5.npz')
PR_a6 = sparse.load_npz('PR_a6.npz')
PR_a7 = sparse.load_npz('PR_a7.npz')
PR_a8 = sparse.load_npz('PR_a8.npz')
PR_a9 = sparse.load_npz('PR_a9.npz')
PR_a10 = sparse.load_npz('PR_a10.npz')
PR_a11 = sparse.load_npz('PR_a11.npz')

V_prev = np.zeros(shape=(N+2))
V_prev[1] = 1.0  # state 0 = losing, 1 = winning, there V[1]=1.0
V = V_prev
i = 0
while True:  # keep iterating till max diff is small enough (< DIFF_THRESHOLD)
    # V = max_a[ P(s,a,s') * V(s') ]
    V = np.max(np.concatenate([PR_a0.dot(V_prev).reshape(N+2, 1), PR_a1.dot(V_prev).reshape(N+2, 1),
                        PR_a2.dot(V_prev).reshape(N+2, 1), PR_a3.dot(V_prev).reshape(N+2, 1),
                        PR_a4.dot(V_prev).reshape(N+2, 1), PR_a5.dot(V_prev).reshape(N+2, 1),
                        PR_a6.dot(V_prev).reshape(N+2, 1), PR_a7.dot(V_prev).reshape(N+2, 1),
                        PR_a8.dot(V_prev).reshape(N+2, 1), PR_a9.dot(V_prev).reshape(N+2, 1),
                        PR_a10.dot(V_prev).reshape(N+2, 1), PR_a11.dot(V_prev).reshape(N+2, 1)], axis=1), axis=1)
    diff = np.max(np.abs(V-V_prev))  # max difference between iterations
    print(i, diff)
    if diff < DIFF_THRESHOLD:
        break
    V_prev = V
    i += 1

np.savez('appr_reachability_soccer_V_truth', V)
print('Soccer experiment reachability ground truth is stored at appr_reachability_soccer_V_truth.npz')