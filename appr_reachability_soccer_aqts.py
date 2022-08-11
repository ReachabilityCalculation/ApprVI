from scipy import sparse
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from anytree import NodeMixin
import scipy
from datetime import datetime

np.random.seed(20210727)
N = 6002500 # total N non-final states
NS = 60000 # number of samples to estimate max and 90% errors

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
H = np.load('appr_reachability_soccer_lr_features.npz')['arr_0']
# Use the trained Linear Regressor
reg = LinearRegression()
reg.coef_ = np.array([0.07892341, -0.03036688,  0.04394309, -0.04207733,  0.6185226,   0.04268554,
  0.06763494,  0.03975894,  0.07196814,  0.00210926,  0.00082727,  0.00906666,  0.01856762])
reg.intercept_ = 0.05219726510709791

def T(V):
    return np.max(np.concatenate([PR_a0.dot(V).reshape(N + 2, 1), PR_a1.dot(V).reshape(N + 2, 1),
                           PR_a2.dot(V).reshape(N + 2, 1), PR_a3.dot(V).reshape(N + 2, 1),
                           PR_a4.dot(V).reshape(N + 2, 1), PR_a5.dot(V).reshape(N + 2, 1),
                           PR_a6.dot(V).reshape(N + 2, 1), PR_a7.dot(V).reshape(N + 2, 1),
                           PR_a8.dot(V).reshape(N + 2, 1), PR_a9.dot(V).reshape(N + 2, 1),
                           PR_a10.dot(V).reshape(N + 2, 1), PR_a11.dot(V).reshape(N + 2, 1)], axis=1), axis=1)

def show_error_distribution(V1, V2=V_true):
    errors = np.random.choice(np.abs(V1 - V2)[2:], 10000, replace=False)
    # errors = np.abs(V1 - V2)[2:]
    plt.hist(errors, bins=np.arange(start=0, stop=0.4, step=0.01))
    plt.title('error distribution')
    a, b, loc, scale = scipy.stats.beta.fit(errors)
    print('a=%0.4f,b=%0.4f,loc=%0.4f,scale=%0.4f' % (a, b, loc, scale))
    print(scipy.stats.kstest(errors, 'beta', (a, b, loc, scale)))
    x = np.linspace(scipy.stats.beta.ppf(0.01, a, b, loc, scale),
                    scipy.stats.beta.ppf(0.999, a, b, loc, scale), 999)
    points = np.linspace(1, 999, 999).reshape(-1, 1)
    points = np.concatenate([points, scipy.stats.beta.pdf(x, a, b, loc, scale).reshape(-1, 1)], axis=1)
    points[:, 0] = x
    points[:, 1] = points[:, 1] * 100
    plt.plot(points[:, 0], points[:, 1])
    plt.show()

V = np.concatenate([np.array([0.0, 1.0]), reg.predict(H).reshape(N)])
for i in [90, 100]:
    print('percential %d of |V^k-V_true| = %0.4f'%(i, np.percentile(np.abs(V - V_true)[2:], i)))
show_error_distribution(V)

samples = np.random.choice(np.arange(0, N), NS, replace=False) + 2
V1 = V[samples]
V2 = T(V)[samples]
for i in [90, 100]:
    print('percential %d of |V^k-V^(k+1)| = %0.4f'%(i, np.percentile(np.abs(V1 - V2)[2:], i)))

class MyBaseClass(object):
    foo = 4

class QVNode(MyBaseClass, NodeMixin):
    def __init__(self, name, state, action, value, prob, count, parent=None, children=None):
        super(QVNode, self).__init__()
        self.name = name
        self.state = state
        self.action = action
        self.value = value
        self.prob = prob
        self.count = count
        self.parent = parent
        if children:
            self.children = children

def init_Q_value(s, a):
    sum = 0.0
    for next_s in (eval('(PR_a%d[%d].indices)'%(a,s))):
        sum += eval('(PR_a%d[%d, next_s])'%(a,s)) * V[next_s]
    return sum

def AQTS(state, margin=0.03, MAX_N=5000, c=1.4, debug=False):
    s0 = QVNode('s0', state, -1, V[state], 1.0, 0)
    next_q = [QVNode('', state, a, init_Q_value(state, a), -1, 0) for a in range(12)]
    avg_q = np.sum([n.value for n in next_q])/12
    for a in range(12):
        if next_q[a].value > avg_q-margin:
            next_q[a].name = '%d,a%d'%(state, a)
            next_q[a].parent = s0
            # print(next_q[a].name, next_q[a].value, next_q[a].count, next_q[a].parent.name)

    max_level = 1
    while s0.count < MAX_N:
        # selection
        scores = [q.value + c*math.log(s0.count + 1e-8)/(q.count+1e-8) for q in s0.children]
        best_i = np.argmax(scores)
        s = s0.children[best_i]
        level = 1
        while (len(s.children) >0):
            next_states = [x for x in s.children if x.state>=2]  # all unknown states
            if len(next_states) == 0:
                s.count = 100000000
                break
            next_q = []
            for x in next_states:
                [next_q.append(q) for q in x.children]  # all q nodes for all unknown states
            scores = [q.value + c * math.log(s0.count + 1e-8) / (q.count + 1e-8) for q in next_q]
            best_i = np.argmax(scores)
            s = next_q[best_i]
            level += 1
        max_level = max(max_level, level)

        if s.count < 100000000:
            # grow, (1) next state first, then (2) next q
            next_states = [QVNode('%d'%x, x, -1, 0, 0, 0, parent=s) for x in eval('(PR_a%d[%d].indices)' % (s.action, s.state)) if x>0]
            for x in next_states:
                x.prob = eval('(PR_a%d[%d, %d])'%(s.action, s.state, x.state))
                if x.state > 1:
                    next_q = [QVNode('%d,a%d'%(x.state,a), x.state, a, init_Q_value(x.state, a), -1, 1) for a in range(12)]
                    avg_q = np.sum([n.value for n in next_q]) / 12
                    for a in range(12):
                        if next_q[a].value > avg_q - margin:
                            next_q[a].parent = x
                    x.value = np.max([q.value for q in x.children])
                    x.count = 1
                else:
                    x.value = V[x.state]

        # backpropogate
        while True:
            sum = 0
            for x in s.children:
                if len(x.children) > 0:
                    x.value = np.max([q.value for q in x.children])
                    sum += x.prob * x.value
                else:
                    sum += x.prob * x.value

            s.value = sum
            s.count = s.count + 1
            # print(s.name, s.value, s.count)
            s = s.parent
            if s != s0:
                s = s.parent
            else:
                s.value = np.max([q.value for q in s.children])
                s.count = s.count+ 1
                if debug:
                    print(s.value, s.count, level, max_level, V_true[s.state])
                break
    return s.value

samples = np.random.choice(np.arange(0, N), 500, replace=False)
errors = []
print("Current Time = ", datetime.now().strftime("%H:%M:%S"))
print('state_id,V_true,V^k,V^(k+n),ErrorV^(k+n)')
for x in samples:
    V_refined = AQTS(x, margin=0.00, MAX_N=50000, c=0.01, debug=False)
    e = abs(V_refined-V_true[x])
    print('%d,%0.4f,%0.4f,%0.4f,%0.4f'%(x, V_true[x], V[x], V_refined, e))
    errors.append(e)
print("Current Time = ", datetime.now().strftime("%H:%M:%S"))
for i in [90, 100]:
    print('percential %d of errors after AQTS = %0.4f'%(i, np.percentile(errors, i)))




