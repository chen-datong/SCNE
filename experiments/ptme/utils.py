import numpy as np
from XXZ import partial_transpose_rho
import sympy as sp

def ppt3(p2, p3):
    return p2 * p2 - p3

def D3(p2, p3):
    return -1/2 + 3/2 * p2 - p3

def D4(p2, p3, p4):
    return p4 - 1/2 * (1 - p2) * (1 - p2) + 1/3 - 4/3 * p3

def D5(p2, p3, p4, p5):
    return -(1/24 - 5/12 * p2 + 5/8 * p2 ** 2 + 5/6 * p3 - 5/6 * p2 * p3 - 5/4 * p4 + p5)

def B2(p2, p3, p4, p5):
    det = (-p2*p2*p5 + 2*p2*p3*p4 - p3*p3*p3 + p3*p5 - p4*p4)
    return -det

def tr_depol_t(trA, t, alpha, beta):
    expr = 0
    for k in range(t + 1):
        expr += sp.binomial(t, k) * alpha**k * beta**(t - k) * trA[k]
    return sp.simplify(expr)

def D3ToD5_tfim_dprob(file='./heisenberg/tfim_N14_gs_J1.0_h1.0.npy', dprob=[0.2 + 0.01 * i for i in range(50)]):
    psi = np.load(file)
    rho = np.outer(psi, psi.conj())
    d = rho.shape[0]
    dA = d // 2
    dB = 2
    
    # Calculate Partial Transpose   of the pure state rho
    rho_pt = partial_transpose_rho(rho, dA, dB, sys="B")
    
    # rho2 = rho_pt @ rho_pt
    # tr2 = np.trace(rho2)
    # rho3 = rho2 @ rho_pt
    # tr3 = np.trace(rho3)
    # rho4 = rho3 @ rho_pt
    # tr4 = np.trace(rho4)
    # rho5 = rho4 @ rho_pt
    # tr5 = np.trace(rho5)
    tr2 = 1
    tr3 = 0.72090833759654648905
    tr4 = 0.66242685043620008756
    tr5 = 0.57813060099128232849

    trace = {0: d, 1: 1, 2: tr2, 3: tr3, 4: tr4, 5: tr5}
    D3s = []
    D4s = []
    D5s = []
    ppt3s = []
    for p in dprob:
        p2 = tr_depol_t(trace, 2, 1 - p, p / d)
        p3 = tr_depol_t(trace, 3, 1 - p, p / d)
        p4 = tr_depol_t(trace, 4, 1 - p, p / d)
        p5 = tr_depol_t(trace, 5, 1 - p, p / d)
        D3s.append(float(D3(p2, p3)))
        D4s.append(float(D4(p2, p3, p4)))
        D5s.append(float(D5(p2, p3, p4, p5)))
        ppt3s.append(float(ppt3(p2, p3)))
    # find the first index where D3, D4, D5 are all negative
    for i in range(len(dprob)):
        # if ppt3s[i] < 0:
        #     print(f"First index where PPT3 is negative: {i}, p: {dprob[i]}")
        #     break
        # if D3s[i] < 0:
        #     print(f"First index where D3 is negative: {i}, p: {dprob[i]}")
        # if D3s[i] < 0 and D4s[i] < 0:
        #     print(f"First index where D3, D4 are negative: {i}, p: {dprob[i]}")
        
        if D3s[i] < 0 and D4s[i] < 0 and D5s[i] < 0:
            print(f"First index where D3, D4, D5 are all negative: {i}, p: {dprob[i]}")
            break
    return D3s, D4s, D5s
# 0.27, 0.39, 0.475
if __name__ == "__main__":
    D3ToD5_tfim_dprob()
