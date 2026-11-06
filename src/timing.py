import sys, json, time, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np, torch
torch.set_num_threads(1)
from eval_corenet import load_net
from l2o import feat_vec, to_t, pack, zf_layer, metrics
from features import features
from scenario import make
from solvers import *
from core import lift, reduce, mvdr, scnr_mvdr, effective
net = load_net('results/corenet_nr.pt'); out = []
for N in (64, 256):
    for s in range(10 if N == 64 else 6):
        ch = make(dict(ctr_dB=30.0, N=N), np.random.default_rng(10007 * s + 17)); ws = WStep(4, 2); rec = dict(N=N, seed=s)
        t = time.time(); U, r = reduce(ch); o = run_starts(r, 'nrball', oracle_starts(r, 'nrball', projector('nrball')), ws); Th = lift(o['T'], U)
        rec['alg1'] = time.time() - t; rec['alg1_v'] = o['scnr']
        t = time.time(); o = run_starts(r, 'rball', oracle_starts(r, 'rball', projector('rball')), ws); Tr = lift(o['T'], U)
        rec['alg1r'] = time.time() - t; rec['alg1r_v'] = o['scnr']
        t = time.time(); o = run_starts(ch, 'nr', [lift(x, U) for x in oracle_starts(r, 'nrball', projector('nrball'))], ws, outer=40)
        rec['full'] = time.time() - t; rec['full_v'] = o['scnr'] if o else None
        t = time.time(); o = run_starts(ch, 'd', oracle_starts(ch, 'd', projector('d')), ws, outer=30); rec['d'] = time.time() - t; rec['d_v'] = o['scnr']
        t = time.time(); f = features(ch); T = to_t(pack([f]))
        with torch.no_grad(): Z = net(feat_vec(T)); W, exc = zf_layer(Z, T); sc, _ = metrics(Z, W, T)
        rec['net'] = time.time() - t; rec['net_v'] = float(sc[0])
        Zr = Z[0].numpy()   # the core is already in the basis of reduce(ch) (same QR with positive diagonal up to column scaling)
        t = time.time(); Th = lift(Zr, f['U']); rec['lift'] = time.time() - t
        out.append(rec); print(rec, flush=True); json.dump(out, open('results2/timing.json', 'w'))
