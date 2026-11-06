"""CoreNet: N agnostic learning to optimise in the contraction core (Sec. V)."""
import sys, os, time, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from scenario import make
from features import features, split
torch.set_default_dtype(torch.float64)

def sample_prm(rng, Ns):
    return dict(N=int(rng.choice(Ns)), ctr_dB=float(rng.uniform(0, 40)), P_dBm=float(rng.uniform(20, 40)),
                Gam_dB=float(rng.uniform(0, 15)))

def build(n, Ns, seed):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        prm = sample_prm(rng, Ns); ch = make(prm, rng); f = features(ch)
        rows.append(f)
    return pack(rows)

def pack(rows):
    f0 = rows[0]
    D = dict(R=np.stack([r['R'] for r in rows]), rho0=np.array([r['rho0'] for r in rows]),
             chi=np.stack([r['chi'] for r in rows]), e=np.stack([r['e'] for r in rows]),
             dv=np.stack([r['dv'] for r in rows]), Gam=np.stack([r['Gam'] for r in rows]))
    D['MQK'] = (f0['M'], f0['Q'], f0['K'])
    return D

def to_t(D, idx=None):
    sl = slice(None) if idx is None else idx
    T = {k: torch.as_tensor(v[sl]) for k, v in D.items() if k != 'MQK'}
    T['MQK'] = D['MQK']; return T

def feat_vec(T):
    R = T['R']; B = R.shape[0]
    x = [R.real.reshape(B, -1), R.imag.reshape(B, -1), torch.log10(T['rho0'])[:, None] / 2,
         torch.log10(T['chi']) / 2, torch.log10(T['e']) / 2, torch.log10(T['Gam']) ]
    dv = T['dv']; nd = dv.abs().pow(2).sum(1).sqrt() + 1e-30
    x += [torch.log10(nd) / 2, (dv / nd[:, None, :]).real.reshape(B, -1), (dv / nd[:, None, :]).imag.reshape(B, -1)]
    return torch.cat(x, 1)

class CoreNet(nn.Module):
    def __init__(self, nin, d, M, K, width=512, depth=4, symmetric=False):
        super().__init__()
        L = [nn.Linear(nin, width), nn.LayerNorm(width), nn.GELU()]
        for _ in range(depth - 1): L += [nn.Linear(width, width), nn.LayerNorm(width), nn.GELU()]
        self.body = nn.Sequential(*L)
        self.hz = nn.Linear(width, 2 * d * d)
        self.d, self.M, self.K, self.sym = d, M, K, symmetric
    def forward(self, x):
        h = self.body(x); B = x.shape[0]; d, M, K = self.d, self.M, self.K
        y = self.hz(h).view(B, 2, d, d); Y = torch.complex(y[:, 0], y[:, 1])
        if self.sym: Y = 0.5 * (Y + Y.transpose(1, 2))
        return contraction(Y)

def contraction(Y):
    """Z = 2 Y (I + Y^H Y)^(-1): singular values 2s/(1+s^2) <= 1, onto the closed contraction ball,
    symmetric whenever Y is symmetric, and built from a linear solve only (smooth to any order)."""
    d = Y.shape[-1]
    A = torch.eye(d, dtype=Y.dtype) + Y.conj().transpose(-1, -2) @ Y
    return 2 * torch.linalg.solve(A.transpose(-1, -2), Y.transpose(-1, -2)).transpose(-1, -2)

def zf_layer(Z, T):
    """Closed form BS beamformer given the core Z (Sec. V-B): zero forcing user beams with the minimum
    powers that meet the SINR targets, remaining power on the user null space projection of the target
    illumination vector. Returns W (unit total power when feasible) and the power excess."""
    M, Q, K = T['MQK']; R = T['R']
    Gn = R[:, :, :M]; an = R[:, :, M]; Hn = R[:, :, M + 1 + Q:]
    GT = Gn.transpose(1, 2)
    g = T['dv'] + T['e'].unsqueeze(1) * (GT @ (Z.transpose(1, 2) @ Hn))   # B x M x K, SINR uses g_k^T w
    A = g.transpose(1, 2)                                                   # K x M
    AA = A @ A.conj().transpose(1, 2)
    eps = 1e-9 * AA.diagonal(dim1=1, dim2=2).real.sum(1).view(-1, 1, 1)
    V = A.conj().transpose(1, 2) @ torch.linalg.inv(AA + eps * torch.eye(A.shape[1], dtype=A.dtype))   # M x K, A V = I
    vn = V.abs().pow(2).sum(1).sqrt()
    p = T['Gam'] * vn.pow(2)
    excess = p.sum(1) - 1
    scale = torch.clamp(p.sum(1), min=1.0)
    p = p / scale.unsqueeze(1)
    Wc = V / vn.unsqueeze(1) * p.sqrt().unsqueeze(1)
    f0 = (GT @ (Z.transpose(1, 2) @ an.unsqueeze(-1))).squeeze(-1)
    Pn = torch.eye(M, dtype=Z.dtype) - V @ A
    ws = (Pn @ f0.conj().unsqueeze(-1)).squeeze(-1)
    ws = ws / (ws.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-30) * torch.clamp(1 - p.sum(1), min=0).sqrt().unsqueeze(1)
    W = torch.cat([Wc, ws.unsqueeze(-1)], 2)
    return W, excess

def metrics(Z, W, T):
    M, Q, K = T['MQK']; R = T['R']
    Gn = R[:, :, :M]; an = R[:, :, M]; Cn = R[:, :, M + 1:M + 1 + Q]; Hn = R[:, :, M + 1 + Q:]
    GT = Gn.transpose(1, 2)
    b0 = (GT @ (Z @ an.unsqueeze(-1))).squeeze(-1)
    f0 = (GT @ (Z.transpose(1, 2) @ an.unsqueeze(-1))).squeeze(-1)
    Bq = GT @ (Z @ Cn); Fq = GT @ (Z.transpose(1, 2) @ Cn)
    Rx = W @ W.conj().transpose(1, 2)
    p0 = torch.einsum('bm,bmn,bn->b', f0, Rx, f0.conj()).real
    pq = torch.einsum('bmq,bmn,bnq->bq', Fq, Rx, Fq.conj()).real
    Rin = torch.eye(M, dtype=Z.dtype).expand(Z.shape[0], M, M) + (Bq * (T['chi'] * pq).unsqueeze(1)) @ Bq.conj().transpose(1, 2)
    q = torch.linalg.solve(Rin, b0.unsqueeze(-1)).squeeze(-1)
    scnr = T['rho0'] * p0 * (b0.conj() * q).sum(1).real
    g = T['dv'] + T['e'].unsqueeze(1) * (GT @ (Z.transpose(1, 2) @ Hn))      # B x M x K
    S = (g.transpose(1, 2) @ W).abs().pow(2)                                  # B x K x (K+M)
    sig = torch.diagonal(S[:, :, :K], dim1=1, dim2=2)
    sinr = sig / (S.sum(2) - sig + 1)
    return scnr, sinr

def train(D, Dval, steps=20000, bs=512, lr=1e-3, symmetric=False, log=None, seed=0):
    torch.manual_seed(seed)
    M, Q, K = D['MQK']; d = D['R'].shape[1]
    T0 = to_t(D, slice(0, 2)); nin = feat_vec(T0).shape[1]
    net = CoreNet(nin, d, M, K, symmetric=symmetric)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lam = 1.0; n = D['R'].shape[0]; hist = []
    Tv = to_t(Dval)
    t0 = time.time()
    for it in range(steps):
        idx = np.random.randint(0, n, bs); T = to_t(D, idx)
        Z = net(feat_vec(T)); W, exc = zf_layer(Z, T); scnr, sinr = metrics(Z, W, T)
        viol = torch.relu(exc).unsqueeze(1)
        loss = -torch.log(scnr + 1e-12).mean() + lam * viol.sum(1).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sch.step()
        rate = (viol > 0).float().mean().item()
        lam = float(np.clip(lam * np.exp(0.05 * (rate - 0.01) / 0.01), 0.01, 1e3))    # dual ascent on the infeasibility rate
        if it % 200 == 0 or it == steps - 1:
            with torch.no_grad():
                Z = net(feat_vec(Tv)); W, exc = zf_layer(Z, Tv); s, si = metrics(Z, W, Tv)
                vr = (exc > 0).float().mean().item()
                rec = dict(it=it, loss=loss.item(), val_scnr_db=(10 * torch.log10(s)).mean().item(), val_viol=vr, lam=lam, t=time.time() - t0)
                hist.append(rec)
                if log: print(rec, flush=True)
    return net, hist

if __name__ == '__main__':
    mode = sys.argv[1]
    os.makedirs('results', exist_ok=True)
    if mode == 'data':
        t = time.time()
        D = build(int(sys.argv[2]), [32, 48, 64], int(sys.argv[3]))
        np.savez(f'results/l2o_train_{sys.argv[3]}.npz', **{k: v for k, v in D.items() if k != 'MQK'})
        print('built', time.time() - t)
