#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

__author__ = 'Florents Tselai'

import bisect
from copy import copy
from math import floor

from numpy import log2, float64

from .computations import *
from .utils import *


def ApproxMaxMI(D, x, y, k_hat):
    assert x > 1 and y > 1 and k_hat > 1

    Q = EquipartitionYAxis(sort_D_increasing_by(D, increasing_by='y'), y)
    D = sort_D_increasing_by(D, increasing_by='x')
    return OptimizeXAxis(D, Q, x, k_hat)


def ApproxCharacteristicMatrix(D, B, c):
    assert B > 3 and c > 0

    D_orth = [tuple(reversed(p)) for p in D]

    s = int(floor(B / 2)) + 1

    I = np.zeros(shape=(s, s), dtype=float64)
    I_orth = np.zeros(shape=(s, s), dtype=float64)
    M = np.zeros(shape=(s, s), dtype=float64)

    '''
    Lines 2-6
    '''
    for y in range(2, s):
        x = int(floor(B / y))

        for i, v in enumerate(ApproxMaxMI(D, x, y, c * x)): I[i + 2][y] = v

        for i, v in enumerate(ApproxMaxMI(D_orth, x, y, c * x)): I_orth[i + 2][y] = v

    '''
    Lines 7-10
    '''

    def characteristic_value(x, y):
        return max(I[x][y], I_orth[y][x]) if (x * y) <= B and x != 0 and y != 0 else np.nan

    I = np.fromfunction(np.vectorize(characteristic_value), (s, s), dtype=np.float64)

    def normalize(x, y):
        return I[x][y] / min(log2(x), log2(y)) if (x * y) <= B and x != 0 and y != 0 else np.nan

    M = np.fromfunction(np.vectorize(normalize), (s, s), dtype=np.float64)

    return M


def EquipartitionYAxis(D, y):
    assert is_sorted_increasing_by(D, 'y')

    n = len(D)

    desiredRowSize = float64(n) / float64(y)

    i = 0
    sharp = 0
    currRow = 0

    Q = {}
    while (i < n):
        # s = len([p for p in D if p[1] == D[i][1]])
        # s = sum(imap(lambda p: p[1] == D[i][1], D))

        s = sum(1 for p in D if p[1] == D[i][1])

        lhs = abs(float64(sharp) + float64(s) - desiredRowSize)
        rhs = abs(float64(sharp) - desiredRowSize)

        if (sharp != 0 and lhs >= rhs):
            sharp = 0
            currRow += 1
            temp1 = float64(n) - float64(i)
            temp2 = float64(y) - float64(currRow)
            desiredRowSize = temp1 / temp2

        for j in range(s):
            Q[D[i + j]] = currRow

        i += s
        sharp += s

    return Q


def GetClumpsPartition(D, Q):
    assert is_sorted_increasing_by(D, 'x')

    n = len(D)

    Q_tilde = copy(Q)

    i = 0
    c = -1

    while (i < n):
        s = 1
        flag = False
        for j in range(i + 1, n):
            if p_x(D[i]) == p_x(D[j]):
                s += 1
                if Q_tilde[D[i]] != Q_tilde[D[j]]:
                    flag = True
            else:
                break

        if s > 1 and flag:
            for j in range(s):
                Q_tilde[D[i + j]] = c
            c -= 1
        i += s

    i = 0
    P = {}
    P[D[0]] = 0
    for j in range(1, n):
        if Q_tilde[D[j]] != Q_tilde[D[j - 1]]:
            i += 1
        P[D[j]] = i

    return P


def GetSuperclumpsPartition(D, Q, k_hat):
    assert is_sorted_increasing_by(D, 'x')

    P_tilde = GetClumpsPartition(D, Q)
    k = len(set(P_tilde.values()))
    if k > k_hat:
        D_P_tilde = [(0, P_tilde[p]) for p in D]
        P_hat = EquipartitionYAxis(D_P_tilde, k_hat)
        P = {p: P_hat[(0, P_tilde[p])] for p in D}
        return P
    else:
        return P_tilde


def OptimizeXAxis(D, Q, x, k_hat):
    assert is_sorted_increasing_by(D, 'x')

    super_clumps_partition = GetSuperclumpsPartition(D, Q, k_hat)
    c = GetPartitionOrdinalsFromMap(D, super_clumps_partition, axis='x')

    # Total number of clumps
    k = len(set(super_clumps_partition.values()))
    assert k == len(c) - 1

    # Find the optimal partitions of size 2 
    I = np.zeros(shape=(k + 1, x + 1))

    for t in range(2, k + 1):
        s = max(range(1, t + 1),
                key=lambda s: HP([c[0], c[s], c[t]]) - HPQ([c[0], c[s], c[t]], Q))

        # Optimal partition of size 2 on the first t clumps
        P_t_2 = [c[0], c[s], c[t]]
        I[t][2] = HQ(Q) + HP(P_t_2) - HPQ(P_t_2, Q)

    # Inductively build the rest of the table of optimal partitions
    for l in range(3, x + 1):
        for t in range(l, k + 1):
            s = max(range(l - 1, t + 1),
                    key=lambda s: float64((c[s] / c[t])) * (I[s][l - 1] - HQ(Q)) - float64(
                        ((c[t] - c[s]) / c[t])) * HPQ([c[s], c[t]], Q)
                    )
            P_t_l = c[1:l - 1]
            bisect.insort(P_t_l, c[t])
            # Optimal partition of size l on the first t clumps of D
            I[t][l] = HQ(Q) + HP(P_t_l) - HPQ(P_t_l, Q)

    for l in range(k + 1, x + 1): I[k][l] = I[k][k]

    return I[k][2:x + 1]


def mine(cm, B, e=1):
    mic = max(value for (x, y), value in np.ndenumerate(cm) if x * y < B and (x, y) != (0, 0) and not np.isnan(value))
    mas = max(np.abs(value - cm[y][x]) for (x, y), value in np.ndenumerate(cm) if
              x * y < B and (x, y) != (0, 0) and not np.isnan(value))
    mev = max(value for (x, y), value in np.ndenumerate(cm) if
              x * y < B and (x, y) != (0, 0) and not np.isnan(value) and (x is 2 or y is 2))
    mcn = min(np.log2(x * y) for (x, y), value in np.ndenumerate(cm) if
              x * y < B and (x, y) != (0, 0) and not np.isnan(value) and value >= (1 - e) * mic)

    return {'MIC': mic,
            'MAS': mas,
            'MEV': mev,
            'MCN': mcn
            }
