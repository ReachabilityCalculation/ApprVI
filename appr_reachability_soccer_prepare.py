from scipy import sparse
import numpy as np
import math

p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = 0,0,0,0,0,0,0,0,0

# above are the state variables, (x,y) for 4 player, ball=[0..3]
# below are the functions to map between state variables and an integer ID
# 2 final states are: 0=losing, 1=winning

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
Death = 0.1
Survive = 1 - Death

# print(state_to_id(4,6,0,0,0,0,0,0,0))
# print(id_to_state(5831002))

# list of actions: shoot, p1_down, p1_right, p2_down, p2_right, p3_down, p3_right, p4_down, p4_right, pass_1, pass_2, pass_3

def get_PR_a0():
    RI = [0,   1]  # row indexes, from which state
    CI = [0,   1]  # column indexes, to which state
    PR = [1.0, 1.0]  # 0->0 has 1.0 probability, so is 1->1
    SHT_P = np.array([
        [[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 20, 30],
        [0, 0, 0, 0, 20, 50, 60],
        [0, 0, 0, 10, 30, 70, 80]],

        [[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 10, 20],
        [0, 0, 0, 0, 10, 40, 50],
        [0, 0, 0, 0, 20, 60, 70]],

        [[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 10, 20],
        [0, 0, 0, 0, 10, 40, 50],
        [0, 0, 0, 0, 20, 60, 70]],

        [[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 30, 40],
        [0, 0, 0, 0, 10, 50, 60]]
    ])/100
    for id in range(2, N+2):
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = id_to_state(id)
        # success, failure
        RI.append(id)
        RI.append(id)
        CI.append(1)
        CI.append(0)
        if ball == 0:
            PR.append(SHT_P[ball, p1_x, p1_y])  # Pr(id, 1), is the shooting success prob
            PR.append(1.0 - SHT_P[ball, p1_x, p1_y])
        elif ball == 1:
            PR.append(SHT_P[ball, p2_x, p2_y])
            PR.append(1.0 - SHT_P[ball, p2_x, p2_y])
        elif ball == 2:
            PR.append(SHT_P[ball, p3_x, p3_y])
            PR.append(1.0 - SHT_P[ball, p3_x, p3_y])
        else:
            PR.append(SHT_P[ball, p4_x, p4_y])
            PR.append(1.0 - SHT_P[ball, p4_x, p4_y])

    RI = np.array(RI)
    CI = np.array(CI)
    PR = np.array(PR)
    return sparse.coo_matrix((PR,(RI,CI)),shape=(N+2,N+2)).tocsr()

def get_PR_down(p_i):
    RI = [0, 1]
    CI = [0, 1]
    PR = [1.0, 1.0]
    DRB_P = np.array([80, 90, 70, 70])/100  # different player has different dribble success rate
    for id in range(2, N + 2):
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = id_to_state(id)
        if p_i == 0:
            if p1_x >= 4: # can't move down
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i: # dribble down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x+1, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive) # Prob of success dribble to next state
                PR.append(1.0 - DRB_P[ball] * Survive)
            else: # move down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x + 1, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(Survive) # Prob of success running to next state (there is always some chance the opponent steals the ball at each step)
                PR.append(1.0 - Survive)
        elif p_i == 1:
            if p2_x >= 4: # can't move down
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i: # dribble down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x+1, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else: # move down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x+1, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)
        elif p_i == 2:
            if p3_x >= 4: # can't move down
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i: # dribble down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x+1, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else: # move down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x+1, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)
        else:
            if p4_x >= 4:  # can't move down
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i:  # dribble down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x+1, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else:  # move down
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x+1, p4_y, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)

    RI = np.array(RI)
    CI = np.array(CI)
    PR = np.array(PR)
    return sparse.coo_matrix((PR, (RI, CI)), shape=(N + 2, N + 2)).tocsr()

def get_PR_right(p_i):
    RI = [0, 1]
    CI = [0, 1]
    PR = [1.0, 1.0]
    DRB_P = np.array([80, 90, 70, 70])/100
    for id in range(2, N + 2):
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = id_to_state(id)
        if p_i == 0:
            if p1_y >= 6: # can't move right
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i: # dribble right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y+1, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else: # move right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y+1, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)
        elif p_i == 1:
            if p2_y >= 6: # can't move right
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i: # dribble right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y+1, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else: # move right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y+1, p3_x, p3_y, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)
        elif p_i == 2:
            if p3_y >= 6: # can't move right
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i: # dribble right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y+1, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else: # move right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y+1, p4_x, p4_y, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)
        else:
            if p4_y >= 6:  # can't move right
                RI.append(id)
                CI.append(0)
                PR.append(1.0)
            elif ball == p_i:  # dribble right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y+1, ball))
                CI.append(0)
                PR.append(DRB_P[ball] * Survive)
                PR.append(1.0 - DRB_P[ball] * Survive)
            else:  # move right
                # success, failure
                RI.append(id)
                RI.append(id)
                CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y+1, ball))
                CI.append(0)
                PR.append(Survive)
                PR.append(1.0 - Survive)

    RI = np.array(RI)
    CI = np.array(CI)
    PR = np.array(PR)
    return sparse.coo_matrix((PR, (RI, CI)), shape=(N + 2, N + 2)).tocsr()

def get_PR_pass(next_i):
    RI = [0, 1]
    CI = [0, 1]
    PR = [1.0, 1.0]
    SP, MP, LP = 0.95, 0.7, 0.1
    for id in range(2, N + 2):
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, ball = id_to_state(id)
        if ball == 0 and next_i == 1:
            distance = math.sqrt((p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 0 and next_i == 2:
            distance = math.sqrt((p1_x-p3_x)*(p1_x-p3_x) + (p1_y-p3_y)*(p1_y-p3_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 0 and next_i == 3:
            distance = math.sqrt((p1_x-p4_x)*(p1_x-p4_x) + (p1_y-p4_y)*(p1_y-p4_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 1 and next_i == 1:
            distance = math.sqrt((p2_x-p3_x)*(p2_x-p3_x) + (p2_y-p3_y)*(p2_y-p3_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 1 and next_i == 2:
            distance = math.sqrt((p2_x-p4_x)*(p2_x-p4_x) + (p2_y-p4_y)*(p2_y-p4_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 1 and next_i == 3:
            distance = math.sqrt((p2_x-p1_x)*(p2_x-p1_x) + (p2_y-p1_y)*(p2_y-p1_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 2 and next_i == 1:
            distance = math.sqrt((p3_x-p4_x)*(p3_x-p4_x) + (p3_y-p4_y)*(p3_y-p4_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 2 and next_i == 2:
            distance = math.sqrt((p3_x-p1_x)*(p3_x-p1_x) + (p3_y-p1_y)*(p3_y-p1_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 2 and next_i == 3:
            distance = math.sqrt((p3_x-p2_x)*(p3_x-p2_x) + (p3_y-p2_y)*(p3_y-p2_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 3 and next_i == 1:
            distance = math.sqrt((p4_x-p1_x)*(p4_x-p1_x) + (p4_y-p1_y)*(p4_y-p1_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        elif ball == 3 and next_i == 2:
            distance = math.sqrt((p4_x-p2_x)*(p4_x-p2_x) + (p4_y-p2_y)*(p4_y-p2_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)
        else:
            distance = math.sqrt((p4_x-p3_x)*(p4_x-p3_x) + (p4_y-p3_y)*(p4_y-p3_y))
            if distance < 1.5:
                pass_success = SP
            elif distance < 2.01:
                pass_success = MP
            else:
                pass_success = LP
            # success, failure
            RI.append(id)
            RI.append(id)
            CI.append(state_to_id(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, (ball+next_i)%4))
            CI.append(0)
            PR.append(pass_success * Survive)
            PR.append(1.0 - pass_success * Survive)

    RI = np.array(RI)
    CI = np.array(CI)
    PR = np.array(PR)
    return sparse.coo_matrix((PR, (RI, CI)), shape=(N + 2, N + 2)).tocsr()

PR_a0 = get_PR_a0()
PR_a1 = get_PR_down(0)
PR_a2 = get_PR_right(0)
PR_a3 = get_PR_down(1)
PR_a4 = get_PR_right(1)
PR_a5 = get_PR_down(2)
PR_a6 = get_PR_right(2)
PR_a7 = get_PR_down(3)
PR_a8 = get_PR_right(3)
PR_a9 = get_PR_pass(1)
PR_a10 = get_PR_pass(2)
PR_a11 = get_PR_pass(3)

sparse.save_npz('PR_a0', PR_a0)
sparse.save_npz('PR_a1', PR_a1)
sparse.save_npz('PR_a2', PR_a2)
sparse.save_npz('PR_a3', PR_a3)
sparse.save_npz('PR_a4', PR_a4)
sparse.save_npz('PR_a5', PR_a5)
sparse.save_npz('PR_a6', PR_a6)
sparse.save_npz('PR_a7', PR_a7)
sparse.save_npz('PR_a8', PR_a8)
sparse.save_npz('PR_a9', PR_a9)
sparse.save_npz('PR_a10', PR_a10)
sparse.save_npz('PR_a11', PR_a11)

print('Preparing P(s,a,s) is done. For each of the 12 actions, there is a PR_a?.npz holding the P(s, a?, s).')