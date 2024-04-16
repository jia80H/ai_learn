# coding=utf-8

import numpy as np
from random import random
import datetime


A = [[0.5, 0.4, 0.1], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]]
B = [[0.4, 0.6], [0.8, 0.2], [0.5, 0.5]]
pi = [0.2, 0.5, 0.3]
output = [0, 1, 0, 0, 1]

epslon = 1e-10

print 'A='
print np.array(A)
print 'B='
print np.array(B)
print 'pi=', pi
print 'output=', output


def _random_choose(s_ary):
    v = random()
    s = 0
    for i in range(len(s_ary)):
        s += s_ary[i]
        if v <= s:
            return i
    return -1


def build_output(A, B, pi, n):
    pi = np.array(pi)
    result = []
    for _ in range(n):
        i = _random_choose(pi)
        j = _random_choose(B[i])
        result.append(j)
        pi = np.matmul(pi, A)
    return result


################################
#           前向算法
################################
def markov_forward(A, B, pi, output):
    ran = range(len(pi))
    temp = [e for e in pi]
    for out in output:
        alpha = temp
        for i in ran:
            alpha[i] *= B[i][out]
        temp = np.matmul(alpha, A)
    print 'alpha=', alpha
    return sum(alpha)


################################
#           后向算法
################################
def markov_backward(A, B, pi, output):
    ran = range(len(pi))
    beta = [1 for _ in range(len(pi))]
    for out in reversed(output[1:]):
        for i in ran:
            beta[i] *= B[i][out]
        beta = np.matmul(A, beta)

    for i in ran:
        beta[i] *= B[i][output[0]]

    beta = np.matmul(pi, beta)
    return beta


################################
#           Viterbi
################################
def markov_viterbi(A, B, pi, output):
    ran = range(len(pi))
    alpha = pi
    result = []
    for out in output:
        for i in ran:
            alpha[i] *= B[i][out]
        result.append(np.argmax(alpha))
        alpha = np.matmul(alpha, A)
    return result


def baum_welch(states, observs, output):
    A = _get_rand_matrix(states, states)
    B = _get_rand_matrix(states, observs)
    pi = _get_rand_matrix(1, states)[0]

    need_more = True
    while need_more:
        alpha = _baum_welch_forward(A, B, pi, output)
        beta  = _baum_welch_backward(A, B, pi, output)
        _adjust(alpha, beta)

        gamma = _get_gamma(alpha, beta)
        kexi = _get_kexi(alpha, beta, A, B, output)

        need_more = _update_A_B_pi(A, B, pi, gamma, kexi, states, observs, output)

    return A, B, pi


def _update_A_B_pi(A, B, pi, gamma, kexi, states, observs, output):
    need_more = False
    T = len(output)
    for i in range(states):
        sum_gamma = 0
        for t in range(T - 1):
            sum_gamma += gamma[t][i]

        for j in range(states):
            sum_i_j = 0
            for t in range(T - 1):
                sum_i_j += kexi[t][i][j]
            e = sum_i_j / sum_gamma
            if abs(e - A[i][j]) > epslon:
                need_more = True
            A[i][j] = _zero(e)
    for i in range(states):
        sum_observs = [0] * observs
        sum_gamma = 0
        for t in range(T):
            g = gamma[t][i]
            sum_gamma += g
            sum_observs[output[t]] += g
        for j in range(observs):
            e = sum_observs[j] / sum_gamma
            if abs(e - B[i][j]) > epslon:
                need_more = True
            B[i][j] = _zero(e)
    for i in range(states):
        e = gamma[0][i]
        if abs(e - pi[i]) > epslon:
            need_more = True
        pi[i] = _zero(e)
    return need_more


def _zero(e):
    # return e
    return e if abs(e) > epslon else 0


def _adjust(alpha, beta):
    for t in range(len(alpha)):
        total = sum(alpha[t])
        alpha[t] = [e/total for e in alpha[t]]
        beta[t] = [e/total for e in beta[t]]


def _get_gamma(alpha, beta):
    states = len(alpha[0])
    gamma = []
    for t in range(len(alpha)):
        r = []
        total = 0.0
        for i in range(states):
            ab = alpha[t][i] * beta[t][i]
            r.append(ab)
            total += ab
        gamma.append([e/total for e in r])
    return gamma


def _get_kexi(alpha, beta, A, B, output):
    states = len(A)
    kexi = []
    for t in range(len(alpha)-1):
        row_s = []
        total = 0.0
        for i in range(states):
            row = []
            for j in range(states):
                v = alpha[t][i] * A[i][j] * B[j][output[t+1]] * beta[t+1][j]
                total += v
                row.append(v)
            row_s.append(row)
        kexi.append([[e/total for e in row] for row in row_s])
    return kexi


def _get_rand_matrix(rows, cols):
    result = []
    for _ in range(rows):
        row = []
        value = 1.0
        current = 0.0
        n = cols
        while n > 1:
            sep = value / n
            d = 2 * sep * random()
            current += d
            row.append(d)
            value -= d
            n -= 1
        row.append(value)
        result.append(row)

    return result


def _baum_welch_forward(A, B, pi, output):
    ran = range(len(pi))
    result = []

    alpha = [e for e in pi]
    for out in output:
        for i in ran:
            alpha[i] *= B[i][out]
        result.append(alpha)
        alpha = np.matmul(alpha, A)
    return result


def _baum_welch_backward(A, B, pi, output):
    ran = range(len(pi))
    result = []
    beta = [1 for _ in range(len(pi))]
    result.insert(0, beta)
    for out in reversed(output[1:]):
        beta = [e for e in beta]
        for i in ran:
            beta[i] *= B[i][out]
        beta = np.matmul(A, beta)
        result.insert(0, beta)

    # for i in ran:
    #     beta[i] *= B[i][output[0]] * pi[i]
    # result.insert(0, beta)
    #
    return result


if __name__ == '__main__':
    print '----------------------- Forward --------------------------'
    print markov_forward(A, B, pi, output)
    print '----------------------- Backward --------------------------'
    print markov_backward(A, B, pi, output)
    print '----------------------- Viterbi --------------------------'
    print markov_viterbi(A, B, pi, output)

    print '----------------------- Baum Welch --------------------------'
    # output = build_output(A, B, pi, 100)
    # output = [0, 1, 0, 0, 1, 1, 0, 1]
    print datetime.datetime.now().strftime("Start: %H:%M:%S"),
    A, B, pi = baum_welch(3, 2, output)
    print datetime.datetime.now().strftime("\tEnd: %H:%M:%S")
    print "A=\n", np.array(A)
    print "B=\n", np.array(B)
    print "pi=", pi
    print

    pred_out = []
    for _ in range(len(output)):
        pred_out.append(np.argmax(np.matmul(pi, B), 0))
        pi = np.matmul(pi, A)
    print 'real output:\t', output
    print 'pred output:\t', pred_out
    print 'error rate:\t', np.average(abs(np.array(output) - pred_out))
