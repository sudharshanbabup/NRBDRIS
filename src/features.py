"""N agnostic sufficient statistic of a scenario (Theorem 3): Gram coordinates of the channel span."""
import numpy as np

def qr_pos(V):
    U, R = np.linalg.qr(V)
    ph = np.diag(R) / np.maximum(np.abs(np.diag(R)), 1e-300)
    return U * ph, (R.T * ph.conj()).T

def features(ch):
    G, a, C, H = ch['G'], ch['a'], ch['C'], ch['H']
    sG = np.linalg.norm(G, 2)
    cols = [G / sG, a[:, None] / np.linalg.norm(a)] + [C[:, [q]] / np.linalg.norm(C[:, q]) for q in range(C.shape[1])] \
           + [H[:, [k]] / np.linalg.norm(H[:, k]) for k in range(H.shape[1])]
    U, R = qr_pos(np.hstack(cols))
    M = G.shape[1]; Q = C.shape[1]; K = H.shape[1]
    P, s2 = ch['P'], ch['s2r']
    rho0 = ch['st2'] * P * sG ** 4 * np.linalg.norm(a) ** 4 / s2
    chi = ch['sc2'] * P * sG ** 4 * np.linalg.norm(C, axis=0) ** 4 / s2
    e = np.sqrt(P / ch['s2u']) * sG * np.linalg.norm(H, axis=0)
    dv = np.sqrt(P / ch['s2u']) * ch['Hd']
    return dict(U=U, R=R, M=M, Q=Q, K=K, rho0=rho0, chi=chi, e=e, dv=dv, Gam=ch['Gam'])

def split(R, M, Q, K):
    Gn = R[:, :M]; an = R[:, M]; Cn = R[:, M + 1:M + 1 + Q]; Hn = R[:, M + 1 + Q:M + 1 + Q + K]
    return Gn, an, Cn, Hn
