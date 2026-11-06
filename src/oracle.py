"""Rank one (LoS) oracle used for (i) theory curves and (ii) theory guided initialisation.
Unit wavefronts: t (illumination, t = Theta u1) and r (collection, r = Theta^T u1).
J(t, r) = |a^T t|^2 |r^T a|^2 / (1 + sum_q chi_q |c_q^T t|^2 |r^T c_q|^2), a, c_q unit norm."""
import numpy as np

def unit(v): return v / np.linalg.norm(v)

def mvdr_dir(a, C, w, extra=None):
    n = a.size; R = np.eye(n, dtype=complex)
    if C.shape[1]: R = R + (C.conj() * w) @ C.T
    return unit(np.linalg.solve(R, a.conj()))

def J(a, C, chi, t, r):
    return np.abs(a @ t) ** 2 * np.abs(r @ a) ** 2 / (1 + np.sum(chi * np.abs(C.T @ t) ** 2 * np.abs(C.T @ r) ** 2))

def null_dir(a, C, S):
    """unit x maximising |x^T a| subject to x^T c_q = 0, q in S"""
    if len(S) == 0: return unit(a.conj())
    Q, _ = np.linalg.qr(C[:, S].conj())
    x = a.conj() - Q @ (Q.conj().T @ a.conj())
    return unit(x)

def oracle_nr(a, C, chi, iters=200):
    """block coordinate ascent; both updates are exact (generalised Rayleigh quotients), so J is monotone.
    Starts: every split S (tx nulls) / S^c (rx nulls)."""
    a = unit(a); Q = C.shape[1]; C = C / np.linalg.norm(C, axis=0) if Q else C
    best = (-1, None, None)
    for m in range(2 ** Q):
        S = [q for q in range(Q) if m >> q & 1]; Sc = [q for q in range(Q) if not m >> q & 1]
        t = null_dir(a, C, S); r = null_dir(a, C, Sc); v = J(a, C, chi, t, r)
        for it in range(iters):
            r = mvdr_dir(a, C, chi * np.abs(C.T @ t) ** 2)
            t = mvdr_dir(a, C, chi * np.abs(C.T @ r) ** 2)
            vn = J(a, C, chi, t, r)
            if vn - v < 1e-10 * v: v = vn; break
            v = vn
        if v > best[0]: best = (v, t, r)
    return best

def oracle_r(a, C, chi, starts=64, iters=400, rng=None):
    """reciprocal: max_t J(t, t). Riemannian gradient ascent with backtracking on the unit sphere of
    span{a*, C*} (the optimum lies there), many starts incl. all null patterns."""
    rng = rng or np.random.default_rng(0)
    a = unit(a); Q = C.shape[1]; C = C / np.linalg.norm(C, axis=0) if Q else C
    B, _ = np.linalg.qr(np.column_stack([a.conj()] + [C[:, q].conj() for q in range(Q)]))
    ab = B.T @ a; Cb = B.T @ C      # x^T a = (B y)^T a = y^T (B^T a)
    def f(y):
        return np.abs(ab @ y) ** 4 / (1 + np.sum(chi * np.abs(Cb.T @ y) ** 4))
    def grad(y):  # d f / d y^*
        s = ab @ y; cs = Cb.T @ y; D = 1 + np.sum(chi * np.abs(cs) ** 4)
        dN = 2 * np.abs(s) ** 2 * s * ab.conj()
        dD = (Cb.conj() * (2 * chi * np.abs(cs) ** 2 * cs)).sum(1)
        return (dN * D - np.abs(s) ** 4 * dD) / D ** 2
    inits = []
    for m in range(2 ** Q):
        S = [q for q in range(Q) if m >> q & 1]
        inits.append(B.conj().T @ null_dir(a, C, S))
    while len(inits) < starts:
        y = rng.standard_normal(B.shape[1]) + 1j * rng.standard_normal(B.shape[1]); inits.append(y)
    best = (-1, None)
    for y in inits:
        y = unit(y); v = f(y); st = 1.0 / max(v, 1e-12)
        for it in range(iters):
            g = grad(y); g = g - y * np.real(np.vdot(y, g))
            acc = False
            for bt in range(30):
                yn = unit(y + st * g); vn = f(yn)
                if vn > v: acc = True; break
                st *= 0.5
            if not acc: break
            if vn - v < 1e-11 * v: y, v = yn, vn; break
            y, v = yn, vn; st *= 2
        if v > best[0]: best = (v, B @ y)
    return best

def xi(a, C):
    if C.shape[1] == 0: return 0.0
    Q, _ = np.linalg.qr(C); a = unit(a)
    return float(np.linalg.norm(Q.conj().T @ a) ** 2)

def nr_highcnr(a, C):
    Q = C.shape[1]; best = 0
    for m in range(2 ** Q):
        S = [q for q in range(Q) if m >> q & 1]; Sc = [q for q in range(Q) if not m >> q & 1]
        best = max(best, (1 - xi(a, C[:, S])) * (1 - xi(a, C[:, Sc])))
    return best

def r_highcnr(a, C):
    return (1 - xi(a, C)) ** 2
