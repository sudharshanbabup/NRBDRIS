"""Beampatterns, convergence and runtime, group connected architectures."""
import sys, os, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scenario import make
from solvers import *
from core import *
os.makedirs('results', exist_ok=True)

def beams():
    ch = make(dict(ctr_dB=30.0, th_t=20.0, clut_off=[-1.2, 1.0, 2.5], th_u=(-30.0, -30.0)), np.random.default_rng(3))
    res = solve_all(ch); U = res['U']; out = {}
    angs = np.linspace(10, 30, 801)
    for k in ('r', 'nr', 'd'):
        o = res[k]; T = lift(o['T'], U) if k != 'd' else o['T']
        W = o['W']; R = W @ W.conj().T; u = mvdr(T, W, ch) if k == 'd' else None
        G = ch['G']; N = G.shape[0]
        ill, col = [], []
        Tr = lift(o['T'], U) if k != 'd' else T
        u = mvdr(Tr, W, ch)
        for th in angs:
            a = ula(N, np.deg2rad(th))
            f = (a @ Tr @ G); ill.append(np.real(f @ R @ f.conj()))
            b = G.T @ (Tr @ a); col.append(np.abs(u.conj() @ b) ** 2)
        ill = np.array(ill); col = np.array(col)
        out[k] = dict(ill=(10 * np.log10(ill / ill.max())).tolist(), col=(10 * np.log10(col / col.max())).tolist(), scnr=o['scnr'])
    out['ang'] = angs.tolist(); out['th_t'] = ch['geo']['th_t']; out['offs'] = ch['geo']['offs']
    json.dump(out, open('results/beams.json', 'w')); print({k: 10*np.log10(out[k]['scnr']) for k in ('r','nr','d')})

def conv():
    """objective versus iterations: core (ball) versus full space, same starts."""
    ch = make(dict(ctr_dB=25.0), np.random.default_rng(11)); U, r = reduce(ch)
    ws = WStep(4, 2); out = {}
    for kind, chh, full in (('nrball', r, False), ('rball', r, False), ('nr', ch, True), ('r', ch, True)):
        proj = projector(kind)
        st = oracle_starts(chh, kind, proj)[0]
        o = ao(chh, kind, T0=st, ws=ws, outer=60, tol=0, hist=True)
        out[kind] = [float(x) for x in o['hist']]
        print(kind, 10 * np.log10(o['hist'][-1]))
    json.dump(out, open('results/conv.json', 'w'))

def runtime():
    out = []
    for N in (32, 64, 128, 256):
        for s in range(5):
            ch = make(dict(ctr_dB=30.0, N=N), np.random.default_rng(100 + s)); ws = WStep(4, 2)
            rec = dict(N=N)
            t = time.time(); U, r = reduce(ch); st = oracle_starts(r, 'nrball', projector('nrball'))[0]
            o = ao(r, 'nrball', T0=st, ws=ws); T = lift(o['T'], U); rec['core_nr'] = time.time() - t; rec['core_val'] = o['scnr']
            t = time.time(); st = oracle_starts(ch, 'nr', projector('nr'))[0]
            o2 = ao(ch, 'nr', T0=st, ws=ws, outer=40); rec['full_nr'] = time.time() - t; rec['full_val'] = o2['scnr'] if o2 else None
            t = time.time(); st = oracle_starts(ch, 'd', projector('d'))[0]
            o3 = ao(ch, 'd', T0=st, ws=ws, outer=30); rec['d'] = time.time() - t
            out.append(rec); print(rec, flush=True)
    json.dump(out, open('results2/runtime.json', 'w'))

def group():
    out = []
    N = 64; ws = WStep(4, 2)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    k, K = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    fn = f'results2/group_{k}.json'
    if os.path.exists(fn): out = json.load(open(fn))
    done = set(r['seed'] for r in out)
    for s in range(k, n, K):
        if s in done: continue
        ch = make(dict(ctr_dB=30.0), np.random.default_rng(5000 + s))
        res = solve_all(ch, ws=ws)
        rec = dict(seed=s, full_nr=res['nr']['scnr'], full_r=res['r']['scnr'], d=res['d']['scnr'])
        for gs in (2, 4, 8, 16, 32):
            for kind in ('nrg', 'rg'):
                proj = projector(kind, gs)
                st = oracle_starts(ch, kind, proj)
                # warm start from the D solution (D is contained in every group connected set)
                st.append(res['d']['T'])
                o = run_starts(ch, kind, st, ws, gsize=gs, outer=30)
                rec[f'{kind}{gs}'] = o['scnr'] if o else None
        out.append(rec); print(s, {k: round(10*np.log10(v),2) for k, v in rec.items() if k != 'seed' and v}, flush=True)
        json.dump(out, open(fn, 'w'))

def conv2():
    """objective versus iteration and time from a plain matched start: core MM versus full space MM."""
    out = []
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 5):
        ch = make(dict(ctr_dB=30.0), np.random.default_rng(7000 + seed)); U, r = reduce(ch); ws = WStep(4, 2)
        rec = dict(seed=seed)
        for kind, chh in (('nrball', r), ('rball', r), ('nr', ch), ('r', ch)):
            proj = projector(kind)
            if kind in ('nrball', 'rball'):
                T0 = init_family(chh, kind, proj)[0]
            else:
                T0 = lift(init_family(r, kind + 'ball', projector(kind + 'ball'))[0], U)
            o = ao(chh, kind, T0=T0, ws=ws, outer=40, tol=0, hist=True)
            rec[kind] = o['hist']; rec[kind + '_t'] = o['htime']
            print(seed, kind, round(10 * np.log10(o['hist'][0]), 2), round(10 * np.log10(o['hist'][-1]), 2), round(o['htime'][-1], 2), flush=True)
        out.append(rec); json.dump(out, open('results2/conv2.json', 'w'))

if __name__ == '__main__':
    {'beams': beams, 'conv': conv, 'conv2': conv2, 'runtime': runtime, 'group': group}[sys.argv[1]]()
