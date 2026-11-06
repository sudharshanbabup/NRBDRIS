"""Core model for non reciprocal BD-RIS monostatic ISAC.
Conventions: Theta is the N x N scattering matrix. Forward (illumination) row a^T Theta G,
collection column G^T Theta a. All metrics depend on Theta through x^T Theta y only.
"""
import numpy as np

def ula(n, th):
    return np.exp(1j * np.pi * np.arange(n) * np.sin(th))

def crandn(*s, rng):
    return (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)

# ---------------------------------------------------------------- projections
def proj_unitary(Y):
    U, _, Vh = np.linalg.svd(Y)
    return U @ Vh

def sym(Y):
    return 0.5 * (Y + Y.T)

def proj_symunitary(Y):
    # polar factor of the symmetric part is symmetric unitary (Lemma 3)
    return proj_unitary(sym(Y))

def proj_diag(Y):
    d = np.diag(Y)
    ph = np.where(np.abs(d) > 0, d / np.maximum(np.abs(d), 1e-300), 1.0)
    return np.diag(ph)

def proj_group(Y, gsize, symmetric):
    N = Y.shape[0]; X = np.zeros_like(Y)
    for s in range(0, N, gsize):
        B = Y[s:s + gsize, s:s + gsize]
        X[s:s + gsize, s:s + gsize] = proj_symunitary(B) if symmetric else proj_unitary(B)
    return X

def proj_ball(Y):
    U, s, Vh = np.linalg.svd(Y)
    return (U * np.minimum(s, 1.0)) @ Vh

def proj_symball(Y):
    return sym(proj_ball(sym(Y)))  # outer sym only removes round off

def projector(kind, gsize=None):
    return {'nr': proj_unitary, 'r': proj_symunitary, 'd': proj_diag,
            'nrball': proj_ball, 'rball': proj_symball,
            'nrg': lambda Y: proj_group(Y, gsize, False),
            'rg': lambda Y: proj_group(Y, gsize, True)}[kind]

# ---------------------------------------------------------------- dilation
def psd_sqrt(A):
    w, V = np.linalg.eigh(0.5 * (A + A.conj().T))
    return (V * np.sqrt(np.maximum(w, 0))) @ V.conj().T

def halmos(Z):
    d = Z.shape[0]; I = np.eye(d)
    S1 = psd_sqrt(I - Z @ Z.conj().T); S2 = psd_sqrt(I - Z.conj().T @ Z)
    return np.block([[Z, S1], [S2, -Z.conj().T]])

def lift(Z, U):
    """Return Theta in U(N) with U^T Theta U = Z (Theorem 3). Symmetric Z gives symmetric Theta."""
    N, d = U.shape
    Q, _ = np.linalg.qr(np.hstack([U, np.random.default_rng(0).standard_normal((N, N - d))]))
    Q[:, :d] = U  # complete basis B = [U, U_perp]
    Psi = np.eye(N, dtype=complex); Psi[:2 * d, :2 * d] = halmos(Z)
    return Q.conj() @ Psi @ Q.conj().T

# ---------------------------------------------------------------- reduction
def reduce(ch):
    """QR of V = [G, a, c_q, h_k]; returns U and reduced channel dict."""
    G, a, C, H = ch['G'], ch['a'], ch['C'], ch['H']
    V = np.hstack([G, a[:, None], C, H])
    U, _ = np.linalg.qr(V)
    r = dict(ch)
    r['G'] = U.conj().T @ G; r['a'] = U.conj().T @ a
    r['C'] = U.conj().T @ C; r['H'] = U.conj().T @ H
    return U, r

# ---------------------------------------------------------------- metrics
def effective(T, ch):
    """g_k (M x K), f (M), b (M), F_q (M x Q), B_q (M x Q)."""
    G, a, C, H = ch['G'], ch['a'], ch['C'], ch['H']
    TG = T @ G
    Gk = ch['Hd'] + (H.T @ TG).T   # columns g_k: h_dk + (h_k^T Theta G)^T
    f = (a @ TG)                  # a^T Theta G  (illumination row)
    b = G.T @ (T @ a)             # G^T Theta a (collection)
    Fq = (C.T @ TG).T             # columns f_q
    Bq = G.T @ (T @ C)            # columns b_q
    return Gk, f, b, Fq, Bq

def sinr(T, W, ch):
    K = ch['H'].shape[1]
    Gk = effective(T, ch)[0]
    S = np.abs(Gk.T @ W) ** 2      # K x (K+Ms)
    sig = np.array([S[k, k] for k in range(K)])
    return sig / (S.sum(1) - sig + ch['s2u'])

def rin(T, W, ch):
    _, f, b, Fq, Bq = effective(T, ch)
    R = W @ W.conj().T
    Pq = np.real(np.einsum('mq,mn,nq->q', Fq, R, Fq.conj()))
    M = b.size
    Rin = (Bq * (ch['sc2'] * Pq)) @ Bq.conj().T + ch['s2r'] * np.eye(M)
    P0 = np.real(f @ R @ f.conj())
    return Rin, P0, b

def scnr_mvdr(T, W, ch):
    Rin, P0, b = rin(T, W, ch)
    return ch['st2'] * P0 * np.real(b.conj() @ np.linalg.solve(Rin, b))

def mvdr(T, W, ch):
    Rin, P0, b = rin(T, W, ch)
    return np.linalg.solve(Rin, b)

def scnr_u(T, W, u, ch):
    Rin, P0, b = rin(T, W, ch)
    return ch['st2'] * P0 * np.abs(u.conj() @ b) ** 2 / np.real(u.conj() @ Rin @ u)

# ---------------------------------------------------------------- gradients wrt conj(Theta)
def obj_grad(T, W, u, ch):
    """f = N/D with fixed u, W. Returns f and df/dTheta^*."""
    G, a, C = ch['G'], ch['a'], ch['C']
    p = G.conj() @ u
    Q = G @ (W @ W.conj().T) @ G.conj().T
    TQ = T @ Q
    x0 = p.conj() @ T @ a
    y0 = np.real(a @ TQ @ T.conj().T @ a.conj())
    Num = ch['st2'] * abs(x0) ** 2 * y0
    dNum = ch['st2'] * (y0 * x0 * np.outer(p, a.conj()) + abs(x0) ** 2 * np.outer(a.conj(), a @ TQ))
    Den = ch['s2r'] * np.real(u.conj() @ u); dDen = np.zeros_like(T)
    for q in range(C.shape[1]):
        c = C[:, q]; s2 = ch['sc2'][q]
        xq = p.conj() @ T @ c
        yq = np.real(c @ TQ @ T.conj().T @ c.conj())
        Den += s2 * abs(xq) ** 2 * yq
        dDen += s2 * (yq * xq * np.outer(p, c.conj()) + abs(xq) ** 2 * np.outer(c.conj(), c @ TQ))
    f = Num / Den
    return f, (dNum - f * dDen) / Den

def slack_grad(T, W, ch, Gam):
    """normalised SINR slack s_k = (|z_kk|^2 - Gam_k(I_k + s2u))/(Gam_k s2u) and gradient."""
    G, H = ch['G'], ch['H']; K = H.shape[1]
    V = G @ W                      # N x (K+Ms)
    Zm = ch['Hd'].T @ W + H.T @ T @ V   # K x (K+Ms)
    s = np.zeros(K); A = []
    for k in range(K):
        z = Zm[k]; wgt = -Gam[k] * np.ones(z.size); wgt[k] = 1.0
        s[k] = (np.sum(wgt * np.abs(z) ** 2) - Gam[k] * ch['s2u']) / (Gam[k] * ch['s2u'])
        A.append(np.outer(H[:, k].conj(), (wgt * z) @ V.conj().T) / (Gam[k] * ch['s2u']))
    return s, A

# ---------------------------------------------------------------- fast diagonal (D-RIS) versions
def obj_grad_d(th, W, u, ch):
    G, a, C = ch['G'], ch['a'], ch['C']
    p = G.conj() @ u
    V = G @ W                               # Q = V V^H
    def parts(c):
        ct = c * th
        x = np.vdot(p, ct)                  # p^H Theta c
        e = ct @ V                          # (c o th)^T V
        v = V.conj() @ e                    # (c o th)^T Q
        y = np.real(np.vdot(e, e))
        g = y * x * p * c.conj() + abs(x) ** 2 * c.conj() * v
        return abs(x) ** 2 * y, g
    n0, g0 = parts(a); Num = ch['st2'] * n0; dNum = ch['st2'] * g0
    Den = ch['s2r'] * np.real(np.vdot(u, u)); dDen = np.zeros_like(th)
    for q in range(C.shape[1]):
        nq, gq = parts(C[:, q]); Den += ch['sc2'][q] * nq; dDen += ch['sc2'][q] * gq
    f = Num / Den
    return f, (dNum - f * dDen) / Den

def slack_grad_d(th, W, ch, Gam):
    G, H = ch['G'], ch['H']; K = H.shape[1]
    V = G @ W
    Zm = ch['Hd'].T @ W + (H * th[:, None]).T @ V
    s = np.zeros(K); A = []
    for k in range(K):
        z = Zm[k]; wgt = -Gam[k] * np.ones(z.size); wgt[k] = 1.0
        s[k] = (np.sum(wgt * np.abs(z) ** 2) - Gam[k] * ch['s2u']) / (Gam[k] * ch['s2u'])
        A.append(H[:, k].conj() * ((wgt * z) @ V.conj().T) / (Gam[k] * ch['s2u']))
    return s, A
