# import numpy as np
# import scipy
# import sklearn
# import pandas as pd
# import pickle
# import os

from parameters import *

def compute_L():
    # TODO sth is wrong with EQ. 8, yL len shoud be no more than r < m
    """
    L = S - F as in eq. 8
    """
    # F_top_left [m, m]
    # F_bottom_right [r, r]
    # F_top_right [m,r]
    # F_bottom_left [r,m]
    def gij(i, j):
        return W[i][j] * (c[i][0] * (1 - 0.5 * (1 + y[j][0])) + (1 - c[i][0]) * (0.5 * (1 + y[j][0])))
    # compute F
    for i in range(m+r):
        for j in range(m+r):
            if 1 <= i+1 <= m and m+1 <= j+1 <= m+r:
                F[i][j] = gij(i, j-m)
            if m+1 <= i+1 <= m+r and 1 <= j+1 <= m:
                F[i][j] = gij(i-m, j)  # sth is wrong

    # compute S
    for i in range(m+r):
        S[i][i] = np.sum(F[i])

    return S - F


def update_D():
    def pos(x):
        return (np.abs(x) + x) * 0.5
    def neg(x):
        return (np.abs(x) - x) * 0.5
    D_caret_1 = B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)
    D_caret_2 = yita * neg(DL.dot(p).dot(p.T)) + yita * pos(yL.dot(p.T)) + beta * neg(L21.dot(U)) + beta * neg(L22.dot(DL))

    D_caret = X.dot(V) + \
              gamma * pos(B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)) + \
              gamma * neg(B_bar.T.dot(E.T).dot(E).dot(B_bar).dot(D).dot(q).dot(q.T)) + \
              np.pad(D_caret_2, ((D_caret_1.shape[0] - D_caret_2.shape[0],0),(0,0)))

    D_tilde_1 = beta*pos(L21.dot(U)) + beta*pos(L22.dot(DL)) + yita*pos(DL.dot(p).dot(p.T)) + yita*neg(yL.dot(p.T))
    D_tilde = D.dot(V.T.dot(V)) + \
              lumda*D + \
              gamma*pos(B.T.dot(E.T).dot(E).dot(B_bar).dot(D).dot(q).dot(q.T)) + \
              gamma*neg(B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)) + \
              np.pad(D_tilde_1, ((D.shape[0] - D_tilde_1.shape[0],0), (0, 0))
                     )
    return D*np.sqrt(D_caret/D_tilde)

def update_U():
    def pos(x):
        return (np.abs(x) + x) * 0.5
    def neg(x):
        return (np.abs(x) - x) * 0.5
    U_caret = alpha*(Y*A).dot(U).dot(T.T) + alpha*(Y*A).T.dot(U).dot(T) + beta*neg(L11.dot(U)) + beta*neg(L12.dot(DL))
    U_tilde = alpha*(Y*(U.dot(T).dot(U.T))).dot(U).dot(T.T) + alpha*(Y*(U.dot(T).dot(U.T))).T.dot(U).dot(T) + lumda*U + beta*pos(L11.dot(U)) + beta*pos(L12.dot(DL))
    return U*np.sqrt(U_caret/U_tilde)

def update_V():
    return V * np.sqrt(X.T.dot(D)/(V.dot(D.T).dot(D) + lumda * V))

def update_T():
    return T * np.sqrt(alpha * U.T.dot(Y * A).dot(U) / ( alpha * U.T.dot(Y * (U.dot(T).dot(U.T))).dot(U) + lumda * T))

def update_p():
    return (yita / (yita * DL.T.dot(DL) + lumda * I)).dot(DL.T).dot(yL)

def update_q():
    return (yita / (yita * D.T.dot(B_bar.T).dot(E).dot(B_bar).dot(D) + lumda * I)).dot(D.T).dot(B_bar.T).dot(E).dot(o)

def compute_yU():
    return np.sign(DU.dot(p))

L[:,:] = compute_L()
steps = 1000
for _ in range(steps):

    D[:,:] = update_D()
    U[:,:] = update_U()
    V[:,:] = update_V()
    T[:,:] = update_T()
    p[:,:] = update_p()
    q[:,:] = update_q()
    if _ % 10 == 0: print(_)

yU_pred = compute_yU()
print(yU_pred)


