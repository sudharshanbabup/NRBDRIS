"""Evaluate CoreNet across surface sizes (N agnostic transfer) and run times."""
import sys, os, json, time, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np, torch
torch.set_num_threads(1)
from l2o import CoreNet, feat_vec, to_t, pack, zf_layer, metrics, contraction
from features import features
from scenario import make
from solvers import WStep, ao, solve_all
from core import mvdr, scnr_mvdr, sinr, lift, reduce, effective

def load_net(path, sym=False):
    D = dict(np.load('results/l2o_val.npz')); D['MQK'] = (4, 3, 2)
    nin = feat_vec(to_t(D, slice(0, 2))).shape[1]
    net = CoreNet(nin, 10, 4, 2, symmetric=sym); net.load_state_dict(torch.load(path)); net.eval()
    return net

def run(N, seeds, net, tag, do_ao=True):
    ws = WStep(4, 2); out = []
    for s in seeds:
        ch = make(dict(ctr_dB=float(os.environ.get('CTR', 30)), N=N), np.random.default_rng(10007 * s + 17))
        f = features(ch); T = to_t(pack([f]))
        t0 = time.time()
        with torch.no_grad():
            Z = net(feat_vec(T)); W, exc = zf_layer(Z, T); sc, si = metrics(Z, W, T)
        Zn = Z[0].numpy(); Th = lift(Zn, f['U'])          # the lift is part of the deployed pipeline
        t_net = time.time() - t0
        rec = dict(N=N, seed=s, net=float(sc[0]), net_feas=bool(exc[0] <= 0), t_net=t_net)
        # exact beamformer for the learned core
        U, r = reduce(ch)
        Zr = U.T @ Th @ U
        t0 = time.time(); u = effective(Zr, r)[2]; u = u / np.linalg.norm(u)
        Wsd = ws.solve(Zr, u, r)
        if Wsd is not None:
            for _ in range(2):
                u = mvdr(Zr, Wsd, r); Wn = ws.solve(Zr, u, r)
                if Wn is not None: Wsd = Wn
            rec['net_sdp'] = float(scnr_mvdr(Zr, Wsd, r))
        rec['t_net_sdp'] = t_net + time.time() - t0
        t0 = time.time(); o = ao(r, 'nrball', T0=Zr, ws=ws)
        rec['net_ao'] = float(o['scnr']) if o else None; rec['t_net_ao'] = t_net + time.time() - t0
        if do_ao:
            t0 = time.time(); res = solve_all(ch, ws=ws, timing=True)
            rec['ao_nr'] = float(res['nr']['scnr']); rec['ao_r'] = float(res['r']['scnr']); rec['ao_d'] = float(res['d']['scnr'])
            rec['t_ao'] = res['time']
        out.append(rec)
    return out

if __name__ == '__main__':
    net = load_net('results/corenet_nr.pt')
    Ns = [int(x) for x in sys.argv[1].split(',')]; n = int(sys.argv[2]); tag = sys.argv[3]
    res = []
    for N in Ns:
        res += run(N, range(n), net, tag, do_ao=False); json.dump(res, open(f'results2/corenet_eval_{tag}.json', 'w'))
        print(N, flush=True)
