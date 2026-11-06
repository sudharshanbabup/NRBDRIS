import numpy as np, cvxpy as cp, time
from core import *

class WStep:
    """Exact W step: Charnes-Cooper SDR of the fractional program, tight by rank one construction."""
    def __init__(self, M, K):
        self.M, self.K = M, K
        self.F = cp.Parameter((M, M), hermitian=True); self.C = cp.Parameter((M, M), hermitian=True)
        self.nu = cp.Parameter(nonneg=True)
        self.Hk = [cp.Parameter((M, M), hermitian=True) for _ in range(K)]
        self.Hc = [cp.Parameter((M, M), hermitian=True) for _ in range(K)]
        self.Rk = [cp.Variable((M, M), hermitian=True) for _ in range(K + 1)]
        self.tau = cp.Variable(nonneg=True)
        R = sum(self.Rk)
        cons = [X >> 0 for X in self.Rk]
        cons += [cp.real(cp.trace(self.C @ R)) + self.nu * self.tau == 1, cp.real(cp.trace(R)) <= self.tau]
        for k in range(K):
            cons += [cp.real(cp.trace(self.Hc[k] @ self.Rk[k])) >= cp.real(cp.trace(self.Hk[k] @ R)) + self.tau]
        self.prob = cp.Problem(cp.Maximize(cp.real(cp.trace(self.F @ R))), cons)

    def solve(self, T, u, ch):
        M, K = self.M, self.K; P = ch['P']
        Gk, f, b, Fq, Bq = effective(T, ch)
        Fm = np.outer(f.conj(), f) * ch['st2'] * abs(u.conj() @ b) ** 2
        Cm = np.zeros((M, M), complex)
        for q in range(Fq.shape[1]):
            Cm += ch['sc2'][q] * abs(u.conj() @ Bq[:, q]) ** 2 * np.outer(Fq[:, q].conj(), Fq[:, q])
        nu = ch['s2r'] * np.real(u.conj() @ u) / P       # power normalised R' = R/P
        sF = np.linalg.norm(Fm) + 1e-300; sC = max(np.linalg.norm(Cm), nu)
        Fm = 0.5 * (Fm + Fm.conj().T); Cm = 0.5 * (Cm + Cm.conj().T)
        self.F.value = Fm / sF; self.C.value = Cm / sC; self.nu.value = nu / sC
        for k in range(K):
            Hm = np.outer(Gk[:, k].conj(), Gk[:, k]) * P / ch['s2u']; Hm = 0.5 * (Hm + Hm.conj().T)
            self.Hk[k].value = Hm; self.Hc[k].value = Hm * (1 + 1 / ch['Gam'][k])
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            try: self.prob.solve(solver=cp.SCS, eps=1e-7)
            except Exception: return None
        if self.prob.status not in ('optimal', 'optimal_inaccurate') or self.tau.value is None or self.tau.value <= 0:
            return None
        Rk = [X.value / self.tau.value * P for X in self.Rk]
        Rt = sum(Rk); W = []
        for k in range(K):
            g = Gk[:, k]; den = np.real(g @ Rk[k] @ g.conj())
            W.append(Rk[k] @ g.conj() / np.sqrt(max(den, 1e-300)))
        W = np.array(W).T if K else np.zeros((M, 0))
        Rs = Rt - W @ W.conj().T
        w, V = np.linalg.eigh(0.5 * (Rs + Rs.conj().T))
        Ws = V * np.sqrt(np.maximum(w, 0))
        Wf = np.hstack([W, Ws])
        pw = np.real(np.trace(Wf @ Wf.conj().T))
        if pw > P: Wf *= np.sqrt(P / pw)
        return Wf

def curv(W, ch, Gam):
    """c_k = Gam_k ||h_k||^2 ||G W_{-k}||_2^2 / (Gam_k s2u): majoriser curvature of the normalised SINR slack."""
    K = len(Gam); G, H = ch['G'], ch['H']; out = np.zeros(K)
    for k in range(K):
        Wm = np.delete(W, k, axis=1)
        out[k] = np.linalg.norm(H[:, k]) ** 2 * np.linalg.norm(G @ Wm, 2) ** 2 / ch['s2u']
    return out

def theta_step_d(T, W, u, ch, iters=20, L=None, tol=1e-7):
    """same MM step for the diagonal architecture, O(N) per evaluation (vector form)."""
    Gam = ch['Gam']; K = len(Gam); th = np.diag(T).copy()
    ph = lambda v: v / np.maximum(np.abs(v), 1e-300)
    f, D = obj_grad_d(th, W, u, ch)
    if L is None: L = 4 * np.linalg.norm(D) / np.sqrt(th.size)
    mu = np.zeros(K)
    for it in range(iters):
        f, D = obj_grad_d(th, W, u, ch)
        s, A = slack_grad_d(th, W, ch, Gam)
        cq = curv(W, ch, Gam)
        acc = False
        for bt in range(40):
            mu_ = mu.copy()
            for di in range(30):
                tn = ph(th + (D + sum(m * a for m, a in zip(mu_, A))) / (L + float(np.dot(mu_, cq))))
                dt = tn - th; nd = np.real(np.vdot(dt, dt))
                lin = np.array([s[k] + 2 * np.real(np.vdot(A[k], dt)) - cq[k] * nd for k in range(K)])
                if np.all(lin >= -1e-9) and di > 0: break
                step = np.array([L / (2 * np.vdot(A[k], A[k]).real + 1e-300) for k in range(K)])
                mu_ = np.maximum(0, mu_ - step * np.minimum(lin, 0) * 1.5)
            fn = obj_grad_d(tn, W, u, ch)[0]
            sn = slack_grad_d(tn, W, ch, Gam)[0]
            if fn >= f * (1 + 1e-12) and np.all(sn >= -1e-9):
                acc = True; break
            L *= 2
        if not acc: break
        th = tn; mu = mu_; L /= 1.6
        if (fn - f) / f < tol: break
    return np.diag(th), L

def theta_step(T, W, u, ch, proj, iters=20, L=None, tol=1e-7, curvature=True):
    """MM with closed form projection and dual corrected direction; monotone and feasibility preserving."""
    Gam = ch['Gam']; K = len(Gam)
    f, D = obj_grad(T, W, u, ch)
    if L is None: L = 4 * np.linalg.norm(D) / np.linalg.norm(T)
    mu = np.zeros(K)
    for it in range(iters):
        f, D = obj_grad(T, W, u, ch)
        s, A = slack_grad(T, W, ch, Gam)
        cq = curv(W, ch, Gam) if curvature else np.zeros(K)   # curvature of the interference majoriser (Sec. IV-C)
        acc = False
        for bt in range(40):
            mu_ = mu.copy()
            for di in range(30):   # projected dual ascent on linearised constraints
                Tn = proj(T + (D + sum(m * a for m, a in zip(mu_, A))) / (L + float(np.dot(mu_, cq))))
                dT = Tn - T; nd = np.real(np.vdot(dT, dT))
                lin = np.array([s[k] + 2 * np.real(np.vdot(A[k], dT)) - cq[k] * nd for k in range(K)])
                if np.all(lin >= -1e-9) and di > 0: break
                step = np.array([L / (2 * np.vdot(A[k], A[k]).real + 1e-300) for k in range(K)])
                mu_ = np.maximum(0, mu_ - step * np.minimum(lin, 0) * 1.5 - step * (lin > 0) * 0.0)
            fn = obj_grad(Tn, W, u, ch)[0]
            sn = slack_grad(Tn, W, ch, Gam)[0]
            if fn >= f * (1 + 1e-12) and np.all(sn >= -1e-9):
                acc = True; break
            L *= 2
        if not acc: break
        T = Tn; mu = mu_; L /= 1.6
        if (fn - f) / f < tol: break
    return T, L

def _dirs(ch):
    G, a, C, H = ch['G'], ch['a'], ch['C'], ch['H']
    Ug = np.linalg.svd(G, full_matrices=False)[0]; g = Ug[:, 0]
    nz = lambda v: v / max(np.linalg.norm(v), 1e-300)
    return g, nz(a), [nz(C[:, q]) for q in range(C.shape[1])], [nz(H[:, k]) for k in range(H.shape[1])]

def init_family(ch, kind, proj, n_rand=0, rng=None):
    """Structured initial points: illumination t (target + users, clutter aware) and collection r."""
    g, a, cs, hs = _dirs(ch); n = a.size
    out = []
    for beta in (0.0, 3.0, 30.0, 300.0):
        Pc = np.eye(n, dtype=complex)
        for c in cs: Pc += beta * np.outer(c.conj(), c)
        rclean = np.linalg.solve(Pc, a.conj())          # clutter rejecting pattern (conj domain)
        for wu in (0.5, 2.0):
            t = a.conj() + wu * sum(h.conj() for h in hs) if hs else a.conj()
            if kind in ('nr', 'nrball', 'nrg'):
                Y = np.outer(t / np.linalg.norm(t), g.conj()) + np.outer(g.conj(), rclean / np.linalg.norm(rclean))
            elif kind == 'd':
                tt = rclean + wu * sum(h.conj() for h in hs) if hs else rclean
                Y = np.diag(tt * g.conj().conj() / np.maximum(np.abs(tt * g), 1e-300) ) if False else np.diag(np.exp(1j * np.angle(tt * g.conj().conj().conj())))
            else:
                tt = rclean / np.linalg.norm(rclean) + wu * sum(h.conj() for h in hs) if hs else rclean
                tt = tt / np.linalg.norm(tt)
                Y = np.outer(tt, g.conj()) + np.outer(g.conj(), tt)
            out.append(proj(Y + 1e-6 * np.eye(n)))
    if rng is not None:
        for _ in range(n_rand):
            out.append(proj(out[0] + 0.7 * crandn(n, n, rng=rng)))
    return out

def ao(ch, kind, gsize=None, T0=None, outer=40, inner=15, tol=1e-4, ws=None, verbose=False, hist=False):
    proj = projector(kind, gsize)
    M = ch['G'].shape[1]; K = ch['H'].shape[1]
    ws = ws or WStep(M, K)
    T = T0
    u = effective(T, ch)[2]; u = u / max(np.linalg.norm(u), 1e-300)
    W = ws.solve(T, u, ch)
    if W is None or (K and not np.all(sinr(T, W, ch) >= ch['Gam'] * (1 - 1e-5))): return None
    import time as _tm
    _t0 = _tm.time()
    u = mvdr(T, W, ch); L = None; val = scnr_mvdr(T, W, ch); H = [val]; HT = [_tm.time() - _t0]
    for o in range(outer):
        if kind == 'd':
            T, L = theta_step_d(T, W, u, ch, iters=inner, L=L)
        else:
            T, L = theta_step(T, W, u, ch, proj, iters=inner, L=L, curvature=kind in ('nrg', 'rg', 'nr', 'r'))
        u = mvdr(T, W, ch)
        Wn = ws.solve(T, u, ch)
        if Wn is not None and scnr_u(T, Wn, u, ch) >= scnr_u(T, W, u, ch) * (1 - 1e-6) and (K == 0 or np.all(sinr(T, Wn, ch) >= ch['Gam'] * (1 - 1e-5))):
            W = Wn
        u = mvdr(T, W, ch)
        nv = scnr_mvdr(T, W, ch); H.append(nv); HT.append(_tm.time() - _t0)
        if verbose: print(o, 10 * np.log10(nv))
        if (nv - val) / val < tol and o > 1: val = nv; break
        val = nv
    out = dict(T=T, W=W, u=u, scnr=val, sinr=sinr(T, W, ch) if K else np.zeros(0))
    if hist: out['hist'] = H; out['htime'] = HT
    return out

def solve(ch, kind, gsize=None, ws=None, n_rand=0, rng=None, extra=(), screen=3, **kw):
    """Multi start: run a short AO from every structured start, then refine the best `screen` ones."""
    proj = projector(kind, gsize)
    starts = list(extra) + init_family(ch, kind, proj, n_rand, rng)
    ws = ws or WStep(ch['G'].shape[1], ch['H'].shape[1])
    short = []
    for T0 in starts:
        o = ao(ch, kind, gsize, T0=T0, outer=3, inner=10, ws=ws)
        if o is not None: short.append(o)
    if not short: return None
    short.sort(key=lambda o: -o['scnr'])
    best = None
    for o in short[:screen]:
        r = ao(ch, kind, gsize, T0=o['T'], ws=ws, **kw)
        if r is not None and (best is None or r['scnr'] > best['scnr']): best = r
    return best

def solve_cont(ch, kind, gsize=None, ws=None, stages=(1e-3, 1e-2, 1e-1, 1.0), extra=(), **kw):
    """Clutter power continuation: track the solution from the matched filter regime to the true CNR."""
    proj = projector(kind, gsize)
    ws = ws or WStep(ch['G'].shape[1], ch['H'].shape[1])
    cands = list(extra)
    starts = init_family(ch, kind, proj)[:2]
    best = None
    for T0 in starts:
        T = T0
        for s in stages:
            c2 = dict(ch); c2['sc2'] = ch['sc2'] * s
            o = ao(c2, kind, gsize, T0=T, ws=ws, outer=kw.get('outer', 40), inner=kw.get('inner', 15))
            if o is None: break
            T = o['T']
        if o is not None: cands.append(T)
    for T0 in cands:
        o = ao(ch, kind, gsize, T0=T0, ws=ws, **kw)
        if o is not None and (best is None or o['scnr'] > best['scnr']): best = o
    return best

# ------------------------------------------------------------ theory guided starts (Sec. IV)
import oracle as orc

def rank1_params(ch):
    G = ch['G']; U1, s, Vh = np.linalg.svd(G, full_matrices=False)
    u1 = U1[:, 0]; s1 = s[0]
    P = ch['P']; s2 = ch['s2r']
    na = np.linalg.norm(ch['a']); nC = np.linalg.norm(ch['C'], axis=0) if ch['C'].shape[1] else np.zeros(0)
    rho0 = ch['st2'] * P * s1 ** 4 * na ** 4 / s2
    chi = ch['sc2'] * P * s1 ** 4 * nC ** 4 / s2
    return u1, rho0, chi

def oracle_starts(ch, kind, proj, wus=(0.0, 0.3)):
    u1, rho0, chi = rank1_params(ch)
    a = ch['a']; C = ch['C']; H = ch['H']
    out = []
    if kind in ('nr', 'nrball', 'nrg'):
        _, t, r = orc.oracle_nr(a, C, chi / max(rho0, 1e-300) * rho0)
    else:
        _, t = orc.oracle_r(a, C, chi); r = t
    for wu in wus:
        tt = t + wu * sum(H[:, k].conj() / np.linalg.norm(H[:, k]) for k in range(H.shape[1])) if H.shape[1] else t
        tt = tt / np.linalg.norm(tt)
        if kind == 'd':
            Y = np.diag(np.exp(1j * (np.angle(tt) - np.angle(u1))))
        elif kind in ('nr', 'nrball', 'nrg'):
            Y = np.outer(tt, u1.conj()) + np.outer(u1.conj(), r)
        else:
            Y = np.outer(tt, u1.conj()) + np.outer(u1.conj(), tt)
        out.append(proj(Y + 1e-6 * np.eye(Y.shape[0])))
    return out

def run_starts(ch, kind, starts, ws, gsize=None, **kw):
    best = None
    for T0 in starts:
        o = ao(ch, kind, gsize, T0=T0, ws=ws, **kw)
        if o is not None and (best is None or o['scnr'] > best['scnr']): best = o
    return best

def solve_all(ch, ws=None, do_d=True, timing=False):
    """D -> R -> NR chain of warm starts (each feasible set contains the previous one)."""
    import time
    ws = ws or WStep(ch['G'].shape[1], ch['H'].shape[1])
    U, r = reduce(ch); res = {}; tm = {}
    t0 = time.time()
    if do_d:
        res['d'] = run_starts(ch, 'd', oracle_starts(ch, 'd', projector('d')), ws, outer=30)
    tm['d'] = time.time() - t0; t0 = time.time()
    st = oracle_starts(r, 'rball', projector('rball'))
    if do_d and res['d'] is not None: st.append(U.T @ res['d']['T'] @ U)
    res['r'] = run_starts(r, 'rball', st, ws)
    tm['r'] = time.time() - t0; t0 = time.time()
    st = oracle_starts(r, 'nrball', projector('nrball'))
    if res['r'] is not None: st.append(res['r']['T'])
    res['nr'] = run_starts(r, 'nrball', st, ws)
    if res['nr'] is None and res['r'] is not None: res['nr'] = dict(res['r'])   # U_s is contained in U
    tm['nr'] = time.time() - t0
    if res['r'] is not None and res['nr'] is not None:   # symmetric polish from the NR point
        o = ao(r, 'rball', T0=projector('rball')(res['nr']['T']), ws=ws)
        if o is not None and o['scnr'] > res['r']['scnr']:
            res['r'] = o
            if o['scnr'] > res['nr']['scnr']:        # keep the nesting R in NR
                o2 = ao(r, 'nrball', T0=o['T'], ws=ws)
                res['nr'] = o2 if (o2 is not None and o2['scnr'] >= o['scnr']) else dict(o)
    res['U'] = U
    if timing: res['time'] = tm
    return res

# ------------------------------------------------------------ standard baselines (Sec. VI)
def sym_unitary_map(x, y):
    """symmetric unitary T with T x = y for unit x, y (Lemma 1 / closed form of Nerini et al.)."""
    n = x.size; x = x / np.linalg.norm(x); y = y / np.linalg.norm(y)
    c = y @ x; psi = np.angle(c); phi = 0.5 * np.arccos(min(abs(c), 1.0))
    z = np.zeros(n, complex); z[0] = np.cos(phi); z[1] = 1j * np.sin(phi); z *= np.exp(1j * psi / 2)
    A = np.column_stack([z, z.conj()]); B = np.column_stack([y, x.conj()])
    rng = np.random.default_rng(0)
    Qa, Ra = np.linalg.qr(np.column_stack([A, rng.standard_normal((n, n - 2))]))
    Qb, Rb = np.linalg.qr(np.column_stack([B, rng.standard_normal((n, n - 2))]))
    W2 = Rb[:2, :2] @ np.linalg.pinv(Ra[:2, :2])
    W2 = proj_unitary(W2)
    V = Qb @ np.block([[W2, np.zeros((2, n - 2))], [np.zeros((n - 2, 2)), np.eye(n - 2)]]) @ Qa.conj().T
    return V @ V.T

def fixed_theta(ch, T, ws, iters=4):
    """optimise W (SDR) and u (MVDR) for a fixed surface."""
    u = effective(T, ch)[2]; u = u / max(np.linalg.norm(u), 1e-300); W = None
    for _ in range(iters):
        Wn = ws.solve(T, u, ch)
        if Wn is None: break
        W = Wn; u = mvdr(T, W, ch)
    if W is None or (ch['H'].shape[1] and not np.all(sinr(T, W, ch) >= ch['Gam'] * (1 - 1e-5))): return None
    return dict(T=T, W=W, u=u, scnr=scnr_mvdr(T, W, ch))

def baselines(ch, res, ws):
    out = {}
    U, r = res['U'], None
    # (i) clutter unaware closed form BD-RIS: maximise the target round trip gain |a^T Theta g| [Nerini TWC 2024]
    g = np.linalg.svd(ch['G'], full_matrices=False)[0][:, 0]
    T = sym_unitary_map(g, ch['a'].conj())
    out['cf'] = fixed_theta(ch, T, ws)
    # (ii) two stage design: symmetric unitary projection of the NR solution [Fang and Mao CL 2024]
    if res.get('nr') is not None:
        Tnr = lift(res['nr']['T'], U)
        out['proj'] = fixed_theta(ch, proj_symunitary(Tnr), ws)
    # (iii) reciprocal BD-RIS by manifold optimisation from a generic start [Fidanovski TWC 2026]
    T0 = init_family(ch, 'r', projector('r'))[0]
    out['mo'] = ao(ch, 'r', T0=T0, ws=ws, outer=20)
    return out
