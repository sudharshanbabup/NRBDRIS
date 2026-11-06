"""Rank one (LoS) BS-RIS oracle: illumination t and collection r wavefronts (unit norm).
SCNR = rho0 * |a^T t|^2 |r^T a|^2 / (1 + sum_q chi_q |c_q^T t|^2 |r^T c_q|^2), all vectors unit norm.
Reciprocal: r = t."""
import numpy as np
from core import ula

def unit(v): return v / np.linalg.norm(v)

def xi(a, C):
    """fraction of target steering inside the clutter span: ||P_C a||^2 / ||a||^2"""
    if C.shape[1] == 0: return 0.0
    Q, _ = np.linalg.qr(C); a = unit(a)
    return float(np.linalg.norm(Q.conj().T @ a) ** 2)

def best_null(a, C):
    """max |x^T a|^2 over unit x with x^T c = 0 for c in C  ( = 1 - xi )"""
    return 1.0 - xi(a, C)

def nr_highcnr(a, C):
    """exact NR high CNR value: max over splits S of (1 - xi_S)(1 - xi_{S^c})"""
    Q = C.shape[1]; best = 0
    for m in range(2 ** Q):
        S = [q for q in range(Q) if m >> q & 1]; Sc = [q for q in range(Q) if not m >> q & 1]
        best = max(best, (1 - xi(a, C[:, S])) * (1 - xi(a, C[:, Sc])))
    return best

def r_highcnr(a, C):
    return (1 - xi(a, C)) ** 2

def _mvdr_dir(a, C, w):
    """unit x maximising |x^T a|^2/(1+sum w_q |x^T c_q|^2): x* = R^{-1} a^* direction (conj domain)."""
    n = a.size; R = np.eye(n, dtype=complex)
    for q in range(C.shape[1]): R += w[q] * np.outer(C[:, q].conj(), C[:, q])
    return unit(np.linalg.solve(R, a.conj()))

def oracle_nr(a, C, chi, iters=300, starts=8, rng=None):
    a = unit(a); C = np.array([unit(C[:, q]) for q in range(C.shape[1])]).T if C.shape[1] else C
    rng = rng or np.random.default_rng(0); best = 0
    for s in range(starts):
        t = unit(a.conj() + (0.3 * s) * (rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size)) / np.sqrt(a.size))
        for it in range(iters):
            r = _mvdr_dir(a, C, chi * np.abs(C.T @ t) ** 2)
            t = _mvdr_dir(a, C, chi * np.abs(r @ C) ** 2)
        v = np.abs(a @ t) ** 2 * np.abs(r @ a) ** 2 / (1 + np.sum(chi * np.abs(C.T @ t) ** 2 * np.abs(r @ C) ** 2))
        best = max(best, v)
    return best

def oracle_r(a, C, chi, iters=2000, starts=8, rng=None):
    """reciprocal: max |a^T t|^4/(1+sum chi |c^T t|^4) on the unit sphere of span{a*,C*}; fixed point on MVDR form."""
    a = unit(a); C = np.array([unit(C[:, q]) for q in range(C.shape[1])]).T if C.shape[1] else C
    rng = rng or np.random.default_rng(0); best = 0
    f = lambda t: np.abs(a @ t) ** 4 / (1 + np.sum(chi * np.abs(C.T @ t) ** 4))
    for s in range(starts):
        t = unit(a.conj() + (0.3 * s) * (rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size)) / np.sqrt(a.size))
        for it in range(iters):
            # stationarity: t ∝ (I + sum chi_q |c_q^T t|^2 c_q^* c_q^T * |a^Tt|^2/(... ) ) ...; use the same MVDR fixed point
            # (it is the KKT condition of max |a^T t|^2 |a^T t|^2 / D with weights chi|c^T t|^2)
            tn = _mvdr_dir(a, C, 2 * chi * np.abs(C.T @ t) ** 2 * 1.0 / (1 + np.sum(chi * np.abs(C.T @ t) ** 4)) * 1.0)
            # damped update keeps monotone behaviour in practice
            cand = [unit(0.5 * t + 0.5 * tn), tn]
            vals = [f(x) for x in cand]
            if max(vals) >= f(t): t = cand[int(np.argmax(vals))]
            else: break
        best = max(best, f(t))
    return best
