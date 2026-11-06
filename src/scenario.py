"""Geometry and channels (2D, RIS at the origin, angles w.r.t. RIS broadside).
Distances in m, powers in mW (linear). Path loss C0 (d/1m)^(-e)."""
import numpy as np
from core import ula, crandn

def db(x): return 10 ** (np.asarray(x, float) / 10)

DEFAULT = dict(M=4, N=64, K=2, Q=3, P_dBm=30.0, s2_dBm=-100.0, C0_dB=-30.0,
               d_BR=20.0, th_BR=-40.0, ph_BR=20.0, kap_G_dB=10.0, e_BR=2.2,
               d_t=10.0, th_t=(10.0, 40.0), e_t=2.0, rcs_t_dB=0.0,
               clut_span=3.0, clut_min=0.3, ctr_dB=30.0, clut_off=None,
               d_u=(60.0, 100.0), th_u=(-60.0, 0.0), kap_u_dB=3.0, e_u=2.6,
               d_BU=(70.0, 110.0), e_d=3.5, direct=True,
               user_clut_dB=None, Gam_dB=10.0)

def uni(rng, lo_hi):
    lo, hi = (lo_hi, lo_hi) if np.isscalar(lo_hi) else lo_hi
    return lo + (hi - lo) * rng.random()

def make(prm, rng):
    p = dict(DEFAULT); p.update(prm)
    M, N, K, Q = p['M'], p['N'], p['K'], p['Q']
    th = np.deg2rad
    PL = lambda d, e: db(p['C0_dB']) * d ** (-e)
    kG = db(p['kap_G_dB'])
    aB = np.exp(1j * np.pi * np.arange(M) * np.sin(th(p['ph_BR'])))
    los = np.outer(ula(N, th(p['th_BR'])), aB)
    if np.isinf(p['kap_G_dB']):
        G = np.sqrt(PL(p['d_BR'], p['e_BR'])) * los
    else:
        G = np.sqrt(PL(p['d_BR'], p['e_BR'])) * (np.sqrt(kG / (1 + kG)) * los + np.sqrt(1 / (1 + kG)) * crandn(N, M, rng=rng))
    ku = db(p['kap_u_dB']); H = np.zeros((N, K), complex); Hd = np.zeros((M, K), complex)
    thu = [uni(rng, p['th_u']) for _ in range(K)]; du = [uni(rng, p['d_u']) for _ in range(K)]
    for k in range(K):
        H[:, k] = np.sqrt(PL(du[k], p['e_u'])) * (np.sqrt(ku / (1 + ku)) * ula(N, th(thu[k])) + np.sqrt(1 / (1 + ku)) * crandn(N, rng=rng))
        if p['direct']:
            Hd[:, k] = np.sqrt(PL(uni(rng, p['d_BU']), p['e_d'])) * crandn(M, rng=rng)
    tht = uni(rng, p['th_t'])
    a = ula(N, th(tht))
    st2 = PL(p['d_t'], p['e_t']) ** 2 * db(p['rcs_t_dB'])
    if p['clut_off'] is not None:
        offs = list(p['clut_off'])
    else:
        offs = []
        while len(offs) < Q:
            o = uni(rng, (-p['clut_span'], p['clut_span']))
            if abs(o) >= p['clut_min']: offs.append(o)
    C = [ula(N, th(tht + o)) for o in offs]
    sc2 = [st2 * db(p['ctr_dB'])] * len(offs)
    if p['user_clut_dB'] is not None:          # scatterers around each user (same angle and range as the user)
        for k in range(K):
            C.append(ula(N, th(thu[k]))); sc2.append(PL(du[k], p['e_t']) ** 2 * db(p['user_clut_dB']))
    C = np.array(C).T if C else np.zeros((N, 0), complex)
    s2 = db(p['s2_dBm'])
    return dict(G=G, a=a, C=C, H=H, Hd=Hd, st2=st2, sc2=np.array(sc2, float), s2r=s2, s2u=s2,
                P=db(p['P_dBm']), Gam=db(p['Gam_dB']) * np.ones(K),
                geo=dict(th_t=tht, offs=offs, th_u=thu, d_u=du))
