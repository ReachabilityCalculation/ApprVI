from scipy import sparse
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from datetime import datetime

np.random.seed(20210727)
N = 6002500 # total N non-final states
NS = 3000 # number of states to sample at each iteration
K = 500 # number of iterations

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

def h(i, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball):
    # convert state variables to 14 features, 1 constant term, 5 player/ball to goal distances, 8 (x,y) player positions
    if i == 0:
        return 1.0
    elif i == 1:
        distance = math.sqrt((p1_x-4.5) * (p1_x-4.5) + (p1_y-6.5) * (p1_y-6.5))
        return 1/(1+math.exp(distance-2))
    elif i == 2:
        distance = math.sqrt((p2_x-4.5) * (p2_x-4.5) + (p2_y-6.5) * (p2_y-6.5))
        return 1/(1+math.exp(distance-2))
    elif i == 3:
        distance = math.sqrt((p3_x-4.5) * (p3_x-4.5) + (p3_y-6.5) * (p3_y-6.5))
        return 1/(1+math.exp(distance-2))
    elif i == 4:
        distance = math.sqrt((p4_x-4.5) * (p4_x-4.5) + (p4_y-6.5) * (p4_y-6.5))
        return 1/(1+math.exp(distance-2))
    elif i == 5:
        return h(ball+1, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball)
    elif i == 6:
        return p1_x/4
    elif i == 7:
        return p1_y/6
    elif i == 8:
        return p2_x/4
    elif i == 9:
        return p2_y/6
    elif i == 10:
        return p3_x/4
    elif i == 11:
        return p3_y/6
    elif i == 12:
        return p4_x/4
    elif i == 13:
        return p4_y/6

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

V_true = np.load('appr_reachability_soccer_V_truth.npz')['arr_0']

print("Current Time = ", datetime.now().strftime("%H:%M:%S"))
H = []
for id in np.arange(N):
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = id_to_state(id + 2)
    for i in range(1, 14):
        H.append(h(i, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
H = np.array(H).reshape(N, 13)  # To convert a state to a feature vector
print('prepare H is done.')
print("Current Time = ", datetime.now().strftime("%H:%M:%S"))

V_prev = np.zeros(shape=(N+2))
V_prev[1] = 1.0
V = V_prev
for i in range(1, K+1):
    V = np.max(np.concatenate([PR_a0.dot(V_prev).reshape(N+2, 1), PR_a1.dot(V_prev).reshape(N+2, 1),
                        PR_a2.dot(V_prev).reshape(N+2, 1), PR_a3.dot(V_prev).reshape(N+2, 1),
                        PR_a4.dot(V_prev).reshape(N+2, 1), PR_a5.dot(V_prev).reshape(N+2, 1),
                        PR_a6.dot(V_prev).reshape(N+2, 1), PR_a7.dot(V_prev).reshape(N+2, 1),
                        PR_a8.dot(V_prev).reshape(N+2, 1), PR_a9.dot(V_prev).reshape(N+2, 1),
                        PR_a10.dot(V_prev).reshape(N+2, 1), PR_a11.dot(V_prev).reshape(N+2, 1)], axis=1), axis=1)
    lr = max(0.3/(i//20+1), 0.05)  # decaying learning rate, from 0.3 gradually down to 0.05
    V_target = V_prev + lr * (V - V_prev)

    samples = np.random.choice(np.arange(0, N), NS, replace=False)
    sample_data = H[samples]
    sample_target = V_target[samples+2]
    reg = LinearRegression().fit(sample_data, sample_target)
    V = np.concatenate([np.array([0.0, 1.0]), reg.predict(H).reshape(N)])

    samples = np.random.choice(np.arange(0, N), 60000, replace=False)+2
    diff = np.max(np.abs(V[samples]-V_prev[samples]))
    print('%d, %0.7f, %0.4f'%(i, diff, np.max(np.abs(V - V_true))))
    V_prev = V
print('LR coef = ', reg.coef_)
print('LR intercept = ', reg.intercept_)
print("Current Time = ", datetime.now().strftime("%H:%M:%S"))