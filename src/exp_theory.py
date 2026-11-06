"""Theory validation (rank one LoS BS-RIS link, pure sensing K = 0)."""
import sys, os, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scenario import make
from solvers import solve_all, WStep, rank1_params
import oracle as orc
from core import ula
os.makedirs('results', exist_ok=True)

def t1():
    base = dict(K=0, kap_G_dB=np.inf, th_t=20.0, clut_off=[-1.2, 1.0, 2.5], N=64)
    out = []
    ws = WStep(4, 0)
    for ctr in np.arange(-15, 76, 5):
        ch = make(dict(base, ctr_dB=float(ctr)), np.random.default_rng(0))
        u1, rho0, chi = rank1_params(ch)
        res = solve_all(ch, ws=ws)
        vn = orc.oracle_nr(ch['a'], ch['C'], chi)[0] * rho0
        vr = orc.oracle_r(ch['a'], ch['C'], chi)[0] * rho0
        out.append(dict(ctr=float(ctr), chi_dB=float(10 * np.log10(chi[0])), rho0_dB=float(10 * np.log10(rho0)),
                        ao_nr=res['nr']['scnr'], ao_r=res['r']['scnr'], ao_d=res['d']['scnr'], or_nr=vn, or_r=vr,
                        lim_nr=orc.nr_highcnr(ch['a'], ch['C']) * rho0, lim_r=orc.r_highcnr(ch['a'], ch['C']) * rho0))
        print(out[-1], flush=True)
    json.dump(out, open('results/theory_cnr.json', 'w'))

def t2():
    out = []
    ws = WStep(4, 0)
    for N in (32, 64, 128):
        for nu in np.round(np.arange(0.1, 1.51, 0.1), 2):
            th0 = np.deg2rad(20.0)
            s1 = np.sin(th0) + 2 * nu / N             # psi = pi (s1 - s0) = 2 pi nu / N
            off = np.rad2deg(np.arcsin(s1)) - 20.0
            ch = make(dict(K=0, kap_G_dB=np.inf, th_t=20.0, clut_off=[off], N=N, ctr_dB=60.0 - 40 * np.log10(N / 64)), np.random.default_rng(0))
            res = solve_all(ch, ws=ws)
            xi = orc.xi(ch['a'], ch['C'])
            out.append(dict(N=N, nu=float(nu), off=float(off), xi=xi, gain_ao=res['nr']['scnr'] / res['r']['scnr'],
                            gain_th=1 / (1 - xi), gain_d=res['nr']['scnr'] / res['d']['scnr']))
            print(out[-1], flush=True)
    json.dump(out, open('results/theory_offset.json', 'w'))

if __name__ == '__main__':
    {'t1': t1, 't2': t2}[sys.argv[1]]()
