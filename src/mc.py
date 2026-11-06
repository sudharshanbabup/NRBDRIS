"""Monte Carlo driver. usage: python3 mc.py <exp_name> <worker_id> <n_workers>"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scenario import make
from solvers import solve_all, WStep, baselines

def pd_swerling1(scnr, L=64, pfa=1e-4):
    return pfa ** (1.0 / (1.0 + L * scnr))

EXPS = {
    'ctr':   dict(key='ctr_dB', vals=[0, 5, 10, 15, 20, 25, 30, 35, 40], base={}, R=300),
    'gam':   dict(key='Gam_dB', vals=[0, 4, 8, 12, 16, 20], base={'ctr_dB': 30.0}, R=300),
    'Q':     dict(key='Q', vals=[1, 2, 3, 4, 5], base={'ctr_dB': 30.0}, R=300),
    'P':     dict(key='P_dBm', vals=[15, 20, 25, 30, 35, 40], base={'ctr_dB': 30.0}, R=300),
    'N':     dict(key='N', vals=[32, 48, 64, 96, 128, 192, 256], base={'ctr_dB': 30.0}, R=200),
    'kap':   dict(key='kap_G_dB', vals=[0, 5, 10, 15, 20, 30], base={'ctr_dB': 30.0}, R=300),
}

def main():
    name, wid, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    E = EXPS[name]; out = f'results2/{name}_w{wid}.jsonl'
    os.makedirs('results2', exist_ok=True)
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            d = json.loads(ln); done.add((d['v'], d['seed']))
    jobs = [(v, s) for v in E['vals'] for s in range(E['R'])]
    jobs = [j for i, j in enumerate(jobs) if i % nw == wid and j not in done]
    wsd = {}
    with open(out, 'a') as fh:
        for v, s in jobs:
            prm = dict(E['base']); prm[E['key']] = v
            rng = np.random.default_rng(10007 * s + 17)
            ch = make(prm, rng)
            K = ch['H'].shape[1]; M = ch['G'].shape[1]
            ws = wsd.setdefault((M, K), WStep(M, K))
            t = time.time(); res = solve_all(ch, ws=ws, timing=True)
            rec = dict(v=v, seed=s, time=res.get('time'))
            for k in ('d', 'r', 'nr'):
                rec[k] = None if res[k] is None else float(res[k]['scnr'])
            bl = {}
            if name in ('ctr', 'P'):
                t1 = time.time(); bl = baselines(ch, res, ws); rec['time']['bl'] = time.time() - t1
            for k, o in bl.items():
                rec[k] = None if o is None else float(o['scnr'])
            fh.write(json.dumps(rec) + '\n'); fh.flush()
    open(f'results2/{name}_w{wid}.done', 'w').write('ok')

if __name__ == '__main__':
    main()
