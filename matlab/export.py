"""Export simulation results to matlab_fig_1 panel JSON files (one per panel)."""
import json, glob, os, sys
import numpy as np
RES = sys.argv[1] if len(sys.argv) > 1 else '../src/results2'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data_raw'
os.makedirs(OUT, exist_ok=True)
db = lambda x: 10 * np.log10(np.asarray(x, float))
# one style per scheme, identical in every figure (proposed = blue)
STY = {
 'nr':   dict(label='NR-BD-RIS (proposed)', color=[0.000, 0.447, 0.698], marker='o', line='-', proposed=True),
 'r':    dict(label='R-BD-RIS', color=[0.835, 0.369, 0.000], marker='s', line='--'),
 'd':    dict(label='D-RIS', color=[0.000, 0.620, 0.451], marker='d', line='-.'),
 'th':   dict(color=[0.35, 0.35, 0.35], marker='none', line=':'),
 'net':  dict(label='CoreNet', color=[0.482, 0.196, 0.580], marker='^', line=':'),
 'nsdp': dict(label='CoreNet+SDP', color=[0.800, 0.475, 0.655], marker='v', line='--'),
 'nao':  dict(label='CoreNet+AO', color=[0.902, 0.624, 0.000], marker='p', line='-.'),
 'full': dict(label='NR full space MM', color=[0.337, 0.706, 0.914], marker='>', line='--'),
 'cf':   dict(label='R-BD-RIS, clutter unaware', color=[0.20, 0.20, 0.20], marker='x', line=':'),
 'proj': dict(label='R-BD-RIS, symmetric projection', color=[0.600, 0.400, 0.200], marker='+', line=':'),
 'mo':   dict(label='R-BD-RIS, manifold opt.', color=[0.000, 0.500, 0.500], marker='*', line=':'),
}
json.dump(STY, open('palette.json', 'w'), indent=1)

def S(key, x, y, **kw):
    d = dict(STY[key]); d.update(kw); d['x'] = list(map(float, x)); d['y'] = list(map(float, y)); return d

def panel(name, series, **kw):
    P = dict(name=name, layout=kw.pop('layout', 'pair'), series=series); P.update(kw)
    json.dump(P, open(os.path.join(OUT, name + '.json'), 'w'), indent=1)
    print('wrote', name)

def load_mc(e, Rmax):
    rows = []
    for f in glob.glob(f'{RES}/{e}_w*.jsonl'):
        for ln in open(f):
            d = json.loads(ln)
            if d['seed'] < Rmax and d['nr'] and d['r'] and d['d']: rows.append(d)
    vals = sorted(set(r['v'] for r in rows))
    agg = {}
    for v in vals:
        rr = [r for r in rows if r['v'] == v]
        seeds = set(r['seed'] for r in rr)
        agg[v] = dict(n=len(seeds), **{k: np.array([r[k] for r in rr]) for k in ('nr', 'r', 'd')})
    return agg

def mean_db(a): return float(np.mean(db(a)))

sys.path.insert(0, '../src'); from summ import load
A = '../src/results'
lin = lambda x: np.asarray(x, float)
Pd = lambda g: 1e-4 ** (1 / (1 + 64 * np.asarray(g)))
def agg(e):
    rows = load(e); vs = sorted(set(k[0] for k in rows)); out = {}
    if e == 'gam':   # a design that meets a higher SINR target also meets a lower one: keep the best over larger targets
        rows = {(v, s): {k: (max(rows[(w, s)][k] for w in vs[vs.index(v):] if (w, s) in rows) if k in ('nr', 'r', 'd') else d[k]) for k in d}
                for (v, s), d in rows.items()}
    for v in vs:
        rr = [d for k, d in rows.items() if k[0] == v]
        if e in ('ctr', 'P') and 'cf' in rr[0]: rr = [d for d in rr if d.get('cf') and d.get('proj') and d.get('mo')]   # drop the rare baseline failures for all schemes
        out[v] = {k: np.array([d[k] for d in rr]) for k in ('nr', 'r', 'd', 'cf', 'proj', 'mo') if all(d.get(k) for d in rr)}
    return vs, out

# ---------------- Fig. beampatterns
b = json.load(open(f'{A}/beams.json')); ang = b['ang']; tt = b['th_t']
vl = [dict(kind='vline', x=tt, line='-', color=[0.25, 0.25, 0.25])] + [dict(kind='vline', x=tt + o, line=':') for o in b['offs']]
step = 4
sl = lambda v: v[::step]
panel('fig_beam_r', [S('r', sl(ang), sl(b['r']['ill']), marker='none', label='R-BD-RIS, illumination = collection'),
                     S('d', sl(ang), sl(b['d']['ill']), marker='none', label='D-RIS, illumination = collection')] + vl,
      xlabel='RIS angle (deg)', ylabel='Normalized gain (dB)', xlim=[14, 26], ylim=[-70, 5], legend='south', lock_ylim=True)
panel('fig_beam_nr', [S('nr', sl(ang), sl(b['nr']['ill']), marker='none', label='NR-BD-RIS, illumination'),
                      S('nr', sl(ang), sl(b['nr']['col']), marker='none', line='--', proposed=False, label='NR-BD-RIS, collection')] + vl,
      xlabel='RIS angle (deg)', ylabel='Normalized gain (dB)', xlim=[14, 26], ylim=[-70, 5], legend='south', lock_ylim=True)

# ---------------- Fig. theory
t = json.load(open(f'{A}/theory_cnr.json')); t = sorted(t, key=lambda r: r['chi_dB'])
chi = [r['chi_dB'] for r in t]
panel('fig_th_cnr', [S('nr', chi, db([r['ao_nr'] for r in t]), line='none', label='NR-BD-RIS, Algorithm 1'),
                     S('r', chi, db([r['ao_r'] for r in t]), line='none', label='R-BD-RIS, Algorithm 1'),
                     S('th', chi, db([r['or_nr'] for r in t]), line='-', label='Analysis, NR'),
                     S('th', chi, db([r['or_r'] for r in t]), line='--', label='Analysis, R'),
                     dict(kind='hline', y=float(db(t[0]['lim_nr'])), line=':', label='High CNR limits', color=[0.0, 0.447, 0.698]),
                     dict(kind='hline', y=float(db(t[0]['lim_r'])), line=':', color=[0.835, 0.369, 0.0])],
      xlabel='Clutter to noise ratio {\\it\\chi} (dB)', ylabel='SCNR (dB)', legend='southwest')
o = json.load(open(f'{A}/theory_offset.json'))
nus = np.linspace(0.05, 1.5, 120); xi = np.sinc(nus) ** 2
ser = [S('th', nus, -db(1 - xi), line='-', label='Corollary 1, {\\itN} \\rightarrow \\infty')]
cols = {32: ([0.482, 0.196, 0.580], '^'), 64: ([0.000, 0.447, 0.698], 'o'), 128: ([0.902, 0.624, 0.000], 's')}
for N in (32, 64, 128):
    rr = sorted([r for r in o if r['N'] == N], key=lambda r: r['nu'])
    ser.append(dict(label=f'Algorithm 1, {{\\itN}} = {N}', x=[r['nu'] for r in rr], y=list(map(float, db([r['gain_ao'] for r in rr]))),
                    color=cols[N][0], marker=cols[N][1], line='none'))
panel('fig_th_off', ser, xlabel='Clutter offset {\\it\\nu} (beamwidths)', ylabel='Gain {\\itG} (dB)', xlim=[0, 1.55], legend='northeast')

# ---------------- Fig. convergence and run time
c = json.load(open('../src/results2/conv2.json'))[1]
def ct(k): return [max(x, 1e-3) for x in c[k + '_t']]
panel('fig_conv', [S('nr', ct('nrball'), db(c['nrball']), label='NR, core (proposed)', marker='none'),
                   S('r', ct('rball'), db(c['rball']), label='R, core', marker='none'),
                   S('full', ct('nr'), db(c['nr']), label='NR, full matrix', marker='none'),
                   dict(label='R, full matrix', x=ct('r'), y=list(map(float, db(c['r']))), color=[0.800, 0.475, 0.655], line=':', marker='none')],
      xlabel='Run time (s)', ylabel='SCNR (dB)', xscale='log', ylim=[-32, -12], xlim=[1e-3, 100], xticks=[1e-3, 1e-2, 1e-1, 1, 10, 100], xticklabels=['0.001', '0.01', '0.1', '1', '10', '100'], legend='southeast')
rt = json.load(open('../src/results2/runtime.json')); Ns = [32, 64, 128, 256]
m = lambda k, N: float(np.mean([r[k] for r in rt if r['N'] == N]))
panel('fig_time', [S('nr', Ns, [m('core_nr', N) for N in Ns], label='NR, core (proposed)'),
                   S('full', Ns, [m('full_nr', N) for N in Ns], label='NR, full matrix'),
                   S('d', Ns, [m('d', N) for N in Ns], label='D-RIS'),
                   dict(kind='ref', x=[64, 256], y=[0.4, 25.6])],
      xlabel='Number of elements {\\itN}', ylabel='Run time (s)', xscale='log', yscale='log',
      xticks=Ns, xticklabels=[str(n) for n in Ns], xlim=[28, 290], ylim=[0.01, 400], yticks=[0.01, 0.1, 1, 10, 100], yticklabels=['0.01', '0.1', '1', '10', '100'], annotations=[dict(x=140, y=8, text='O(N^3)')])

# ---------------- Fig. CTR and gain CDF
vs, g = agg('ctr')
panel('fig_ctr', [S('nr', vs, [mean_db(g[v]['nr']) for v in vs]), S('r', vs, [mean_db(g[v]['r']) for v in vs]),
                  S('d', vs, [mean_db(g[v]['d']) for v in vs])] + [S(k, vs, [mean_db(g[v][k]) for v in vs]) for k in ('mo', 'proj', 'cf')],
      xlabel='Clutter to target RCS ratio (dB)', ylabel='Average SCNR (dB)', legend='southwest')
ser = []
for v, colr, mk in ((20, [0.000, 0.620, 0.451], 'd'), (30, [0.835, 0.369, 0.000], 's'), (40, [0.000, 0.447, 0.698], 'o')):
    gg = np.sort(db(g[v]['nr'] / g[v]['r'])); F = np.arange(1, gg.size + 1) / gg.size
    idx = np.unique(np.linspace(0, gg.size - 1, 60).astype(int))
    ser.append(dict(label=f'CTR = {v} dB', x=list(map(float, gg[idx])), y=list(map(float, F[idx])), color=colr, marker='none', line='-' if v == 40 else ('--' if v == 30 else '-.')))
panel('fig_cdf', ser, xlabel='Gain {\\itG} (dB)', ylabel='CDF', ylim=[0, 1.02], xlim=[0, 16], legend='southeast', lock_ylim=True)

# ---------------- Fig. Pd vs P and SCNR vs Q
vs, g = agg('P')
panel('fig_pd', [S('nr', vs, [float(np.mean(Pd(g[v]['nr']))) for v in vs]), S('r', vs, [float(np.mean(Pd(g[v]['r']))) for v in vs]),
                 S('d', vs, [float(np.mean(Pd(g[v]['d']))) for v in vs])] + [S(k, vs, [float(np.mean(Pd(g[v][k]))) for v in vs]) for k in ('mo', 'proj', 'cf')],
      xlabel='Transmit power {\\itP} (dBm)', ylabel='Detection probability', ylim=[0, 1], legend='northwest', lock_ylim=True)
vs, g = agg('Q')
panel('fig_Q', [S('nr', vs, [mean_db(g[v]['nr']) for v in vs]), S('r', vs, [mean_db(g[v]['r']) for v in vs]),
                S('d', vs, [mean_db(g[v]['d']) for v in vs])],
      xlabel='Number of clutter patches {\\itQ}', ylabel='Average SCNR (dB)', xticks=[1, 2, 3, 4, 5], legend='southwest')

# ---------------- Fig. N and Rician factor
vs, g = agg('N')
panel('fig_N', [S('nr', vs, [float(np.mean(db(g[v]['nr'] / g[v]['r']))) for v in vs], label='NR-BD-RIS (proposed)'),
                S('d', vs, [float(np.mean(db(g[v]['d'] / g[v]['r']))) for v in vs], label='D-RIS'),
                dict(kind='hline', y=0.0, line='--', color=[0.835, 0.369, 0.0], label='R-BD-RIS (reference)')],
      xlabel='Number of elements {\\itN}', ylabel='SCNR relative to R-BD-RIS (dB)', xscale='log', xticks=[32, 64, 128, 256], xticklabels=['32', '64', '128', '256'], xlim=[29, 280], legend='northwest')
vs, g = agg('kap')
panel('fig_kap', [S('nr', vs, [mean_db(g[v]['nr']) for v in vs]), S('r', vs, [mean_db(g[v]['r']) for v in vs]),
                  S('d', vs, [mean_db(g[v]['d']) for v in vs])],
      xlabel='Rician factor of {\\bfG} (dB)', ylabel='Average SCNR (dB)', legend='southeast')

# ---------------- Fig. group connected and SINR target
import glob as _g
gr = []
for f in _g.glob('../src/results2/group_*.json'): gr += json.load(open(f))
sz = [1, 2, 4, 8, 16, 32, 64]
def gm(k): return float(np.mean(db([r[k] for r in gr])))
panel('fig_group', [S('nr', sz, [gm('d')] + [gm(f'nrg{s}') for s in sz[1:-1]] + [gm('full_nr')], label='NR-BD-RIS'),
                    S('r', sz, [gm('d')] + [gm(f'rg{s}') for s in sz[1:-1]] + [gm('full_r')], label='R-BD-RIS')],
      xlabel='Ports per group', ylabel='Average SCNR (dB)', xscale='log', xticks=sz, xticklabels=[str(s) for s in sz], xlim=[0.85, 75], legend='southeast')
vs, g = agg('gam')
panel('fig_gam', [S('nr', vs, [mean_db(g[v]['nr']) for v in vs]), S('r', vs, [mean_db(g[v]['r']) for v in vs]),
                  S('d', vs, [mean_db(g[v]['d']) for v in vs])],
      xlabel='SINR target {\\it\\Gamma} (dB)', ylabel='Average SCNR (dB)', legend='southwest')

# ---------------- Fig. CoreNet
h = json.load(open('../src/results/corenet_nr_hist.json'))
va = json.load(open('../src/results/val_ao.json'))
import torch
sys.path.insert(0, '../src')
panel('fig_l2o_train', [S('net', [r['it'] for r in h], [r['val_scnr_db'] for r in h], marker='none', label='CoreNet, validation'),
                        dict(kind='hline', y=float(np.mean(db([r['nr'] for r in va]))), line='--', color=[0.0, 0.447, 0.698], label='Algorithm 1, NR'),
                        dict(kind='hline', y=float(np.mean(db([r['r'] for r in va]))), line='-.', color=[0.835, 0.369, 0.0], label='Algorithm 1, R'),
                        dict(kind='hline', y=float(np.mean(db([r['d'] for r in va]))), line=':', color=[0.0, 0.620, 0.451], label='Algorithm 1, D-RIS')],
      xlabel='Training iteration', ylabel='Average SCNR (dB)', ylim=[-20, -13], legend='southeast')
cn = sum([json.load(open(f)) for f in _g.glob('../src/results2/corenet_eval_s30*.json')], []); rows = load('N')
Nv = [32, 48, 64, 96, 128, 192, 256]
for r in cn: r['net_ao'] = max(r.get('net_ao') or 0.0, r.get('net_sdp') or 0.0)   # CoreNet+AO keeps the better of its start and its refinement
cm = lambda k, N: float(np.mean(db([r[k] for r in cn if r['N'] == N])))
am = lambda k, N: float(np.mean(db([rows[(N, s)][k] for s in range(100) if (N, s) in rows and any(r['N'] == N and r['seed'] == s for r in cn)])))
panel('fig_l2o_N', [S('r', Nv, [am('r', N) - am('nr', N) for N in Nv], label='Algorithm 1, R-BD-RIS'),
                    S('net', Nv, [cm('net', N) - am('nr', N) for N in Nv]), S('nsdp', Nv, [cm('net_sdp', N) - am('nr', N) for N in Nv]),
                    S('nao', Nv, [cm('net_ao', N) - am('nr', N) for N in Nv]),
                    dict(kind='hline', y=0.0, line='-', color=[0.0, 0.447, 0.698], label='Algorithm 1, NR-BD-RIS'),
                    dict(kind='vline', x=64, line=':')],
      xlabel='Number of elements {\\itN}', ylabel='SCNR relative to Algorithm 1 (dB)', xscale='log', xticks=[32, 64, 128, 256], xticklabels=['32', '64', '128', '256'], xlim=[29, 280], ylim=[-8, 1], yticks=[-8, -6, -4, -2, 0], lock_ylim=True, legend='southwest')
