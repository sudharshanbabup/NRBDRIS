import sys, os, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import numpy as np
from l2o import sample_prm
from scenario import make
from solvers import solve_all, WStep
n = int(sys.argv[1]); rng = np.random.default_rng(2); out = []
ws = WStep(4, 2)
for i in range(n):
    prm = sample_prm(rng, [32, 48, 64]); ch = make(prm, rng)
    res = solve_all(ch, ws=ws)
    out.append(dict(i=i, prm=prm, nr=res['nr']['scnr'], r=res['r']['scnr'], d=res['d']['scnr']))
    json.dump(out, open('results/val_ao.json', 'w'))
