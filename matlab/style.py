"""Final figure style: restyle the v6 panel JSON files (same numbers) for single column figures.

Changes with respect to v6: one figure per column (layout 'col'), continuous lines only,
sparse markers on dense curves so every series differs in colour and marker,
and the same Times New Roman size for axes and legends (set in matlab_fig_1.m).
Reads data_raw/ (written by export.py) and writes data/ with the same numbers.
Usage: python3 style.py  (run inside matlab/)
"""
import json, glob, os

SRC = 'data_raw'
OUT = 'data'
os.makedirs(OUT, exist_ok=True)

BLUE, VERM, GREEN = [0.000, 0.447, 0.698], [0.835, 0.369, 0.000], [0.000, 0.620, 0.451]
PURP, PINK, AMBER = [0.482, 0.196, 0.580], [0.800, 0.475, 0.655], [0.902, 0.624, 0.000]
SKY, GREY = [0.337, 0.706, 0.914], [0.35, 0.35, 0.35]


def light(c, f=0.45):
    return [round(x + (1 - x) * f, 3) for x in c]


def fix(name, d):
    d['layout'] = 'col'
    S = d['series']
    for s in S:
        if s.get('line') not in (None, 'none'):
            s['line'] = '-'
        if s.get('kind') in ('hline', 'vline', 'ref'):
            s['line'] = '-'
    if name in ('fig_beam_r', 'fig_beam_nr'):
        n = len(S[0]['x'])
        S[0].update(marker='o' if name == 'fig_beam_nr' else 's', mstep=n // 10)
        if name == 'fig_beam_r':
            S[1].update(marker='d', mstep=n // 10)
        else:
            S[1].update(color=PURP, marker='^', mstep=n // 10, proposed=False)
            S[1]['label'] = 'NR-BD-RIS, collection'
        vl = [s for s in S if s.get('kind') == 'vline']
        vl[0].update(color=[0.15, 0.15, 0.15], lw=1.0)   # target
        for s in vl[1:]:
            s.update(color=[0.62, 0.62, 0.62], lw=0.8)    # clutter
    if name == 'fig_th_cnr':
        S[2].update(color=BLUE, lw=1.0)
        S[3].update(color=VERM, lw=1.0)
        S[4].update(color=light(BLUE), lw=0.9, label='High CNR limit, NR')
        S[5].update(color=light(VERM), lw=0.9, label='High CNR limit, R')
    if name == 'fig_th_off':
        S[0].update(color=GREY, lw=1.2)
    if name == 'fig_conv':   # markers at log spaced run times, not at every iteration
        import math
        for k, (s, mk) in enumerate(zip(S, ('o', 's', '>', 'v'))):
            lx = [math.log10(max(x, 1e-3)) for x in s['x']]
            targ = [lx[0] + (lx[-1] - lx[0]) * (j + 0.25 * k / 4) / 6 for j in range(7)]
            idx = sorted({min(range(len(lx)), key=lambda i: abs(lx[i] - t)) + 1 for t in targ})
            s.update(marker=mk, midx=idx)
    if name == 'fig_cdf':
        for s, mk in zip(S, ('d', 's', 'o')):
            s.update(marker=mk, mstep=6)
    if name == 'fig_l2o_train':
        S[0].update(marker='^', mstep=4)
        d.update(ylim=[-22.5, -13], lock_ylim=True, legend='southeast')
        for s, mk in zip(S[1:], ('o', 's', 'd')):
            s.update(marker=mk, lw=1.1)
    if name == 'fig_l2o_N':
        S[4].update(marker='o', lw=1.2)
        S[5].update(color=[0.62, 0.62, 0.62], lw=0.8)
    if name == 'fig_N':
        S[2].update(marker='s', lw=1.2)
    if name == 'fig_time':
        S[3].update(color=[0.55, 0.55, 0.55], lw=0.9)
        d['annotations'] = [dict(x=105, y=14, text='{\\itO}({\\itN}^3)', italic=False)]
    return d


for f in sorted(glob.glob(os.path.join(SRC, '*.json'))):
    name = os.path.basename(f)[:-5]
    d = fix(name, json.load(open(f)))
    json.dump(d, open(os.path.join(OUT, name + '.json'), 'w'), indent=1)
    print('wrote', name)
