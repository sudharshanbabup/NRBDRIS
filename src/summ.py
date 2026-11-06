import json, glob, sys, os, numpy as np
R = dict(ctr=300, P=300, gam=300, Q=300, N=200, kap=300)
def load(e):
    rows = {}
    for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results2', f'{e}_w*.jsonl')):
        for ln in open(f):
            d = json.loads(ln)
            if d['seed'] < R[e] and d['nr'] and d['r'] and d['d']: rows[(d['v'], d['seed'])] = d
    return rows
if __name__ == '__main__':
    for e in sys.argv[1:]:
        rows = load(e); vs = sorted(set(k[0] for k in rows))
        for v in vs:
            rr = [d for k, d in rows.items() if k[0] == v]
            f = lambda k: np.mean(10*np.log10([d[k] for d in rr]))
            g = np.array([10*np.log10(d['nr']/d['r']) for d in rr])
            pd = lambda k: np.mean([1e-4 ** (1/(1+64*d[k])) for d in rr])
            bl = ''.join(' %s %.2f(%d)' % (k, np.mean([10*np.log10(d[k]) for d in rr if d.get(k)]), sum(1 for d in rr if not d.get(k))) for k in ('cf', 'proj', 'mo') if all(k in d for d in rr))
            print(e, v, len(rr), 'NR %.2f R %.2f D %.2f | gain mean %.2f med %.2f p90 %.2f | Pd %.3f %.3f %.3f' % (f('nr'), f('r'), f('d'), g.mean(), np.median(g), np.percentile(g, 90), pd('nr'), pd('r'), pd('d')) + bl)
