import sys, os, json, time; sys.path.insert(0, '.')
import numpy as np, torch
from l2o import *
torch.set_num_threads(1)
sym = len(sys.argv) > 1 and sys.argv[1] == 'r'
if not os.path.exists('results/l2o_train.npz'):
    D = build(200000, [32, 48, 64], 1); np.savez('results/l2o_train.npz', **{k: v for k, v in D.items() if k != 'MQK'})
    Dv = build(4000, [32, 48, 64], 2); np.savez('results/l2o_val.npz', **{k: v for k, v in Dv.items() if k != 'MQK'})
def load(f):
    z = dict(np.load(f)); z['MQK'] = (4, 3, 2); return z
D = load('results/l2o_train.npz'); Dv = load('results/l2o_val.npz')
net, hist = train(D, Dv, steps=int(os.environ.get('STEPS', 30000)), symmetric=sym, log=True)
tag = 'r' if sym else 'nr'
torch.save(net.state_dict(), f'results/corenet_{tag}.pt'); json.dump(hist, open(f'results/corenet_{tag}_hist.json', 'w'))
