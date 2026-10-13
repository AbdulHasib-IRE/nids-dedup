
import os, glob, gc, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
import torch
from scipy import stats
from sklearn.metrics import f1_score

PROJECT = '/content/drive/MyDrive/NIDS_Contamination'
C64, V2 = '/content/canon64', '/content/canon_v2'
P2 = f'{PROJECT}/results/phase2'
SRC = {'nslkdd':(C64,'nslkdd'), 'cicids2017':(V2,'cicids2017'), 'cic_diad':(V2,'cic_diad'),
       'unswnb15':(V2,'unswnb15'), 'cicids2018':(V2,'cicids2018'), 'botiot':(V2,'botiot'),
       'nbaiot':(C64,'nbaiot')}
NICE = {'nslkdd':'NSL-KDD','cicids2017':'CICIDS2017','cic_diad':'CIC IoT-DIAD 2024',
        'unswnb15':'UNSW-NB15','cicids2018':'CSE-CIC-IDS2018','botiot':'Bot-IoT','nbaiot':'N-BaIoT'}
META = {'y','prov','ts','rowhash','rowhash_port'}
FLOOR, DEV = 500, 'cuda'
SEEDS = [42,123,456,789,1011,1213,1415,1617,1819,2021]
MARGIN, MARGIN2 = 1.0, 0.5          # pre-registered TOST margins (macro-F1 percentage points)

def _codes(arr, vocab):
    u, inv = np.unique(arr, return_inverse=True)
    g = np.array([vocab.setdefault(v, len(vocab)) for v in u], dtype=np.int32)
    return g[inv]

class DS:
    """Valid rows only: no NaN in numeric features, classes with >= FLOOR rows."""
    def __init__(self, name):
        t0 = time.time(); base, sub = SRC[name]; self.name = name
        fs = sorted(glob.glob(f'{base}/{sub}/*.parquet'))
        sch = pq.ParquetFile(fs[0]).schema.names
        self.num = [c for c in sch if c not in META and not c.startswith('cat__')]
        self.cat = [c for c in sch if c.startswith('cat__')]
        yv, cv = {}, [dict() for _ in self.cat]
        Y, H, X, C = [], [], [], []
        for f in fs:
            t = pq.read_table(f, columns=['y','rowhash'] + self.num + self.cat).to_pandas()
            Y.append(_codes(t['y'].astype(str).to_numpy(), yv))
            H.append(t['rowhash'].to_numpy('uint64'))
            X.append(t[self.num].to_numpy('float64'))
            if self.cat:
                C.append(np.stack([_codes(t[c].astype(str).to_numpy(), cv[j])
                                   for j, c in enumerate(self.cat)], 1))
            del t; gc.collect()
        y = np.concatenate(Y); h = np.concatenate(H); X = np.concatenate(X); del Y, H
        ylab = np.empty(len(yv), dtype=object)
        for k, v in yv.items(): ylab[v] = k
        ok = ~np.isnan(X).any(axis=1)
        self.n_raw, self.n_nan = int(len(y)), int((~ok).sum())
        cnt = np.bincount(y[ok], minlength=len(yv))
        keepc = np.flatnonzero(cnt >= FLOOR)
        self.dropped_classes = {str(ylab[c]): int(cnt[c]) for c in np.flatnonzero((cnt > 0) & (cnt < FLOOR))}
        ok &= np.isin(y, keepc)
        remap = np.full(len(yv), -1); remap[keepc] = np.arange(len(keepc))
        self.classes = ylab[keepc]
        self.y = remap[y[ok]].astype(np.int64); self.h = h[ok]; self.X = X[ok]
        if self.cat:
            Cm = np.concatenate(C)[ok]; self.C = Cm
            self.vocab_sizes = [len(v) for v in cv]
        else:
            self.C, self.vocab_sizes = None, []
        del X, h, y, C; gc.collect()
        self.N = len(self.y)
        self.n_feat = len(self.num) + sum(self.vocab_sizes)
        self.load_s = time.time() - t0

    def feats(self, idx):
        A = self.X[idx]
        if self.C is None: return A
        oh = [np.eye(k, dtype=np.float64)[self.C[idx, j]] for j, k in enumerate(self.vocab_sizes)]
        return np.concatenate([A] + oh, axis=1)

    def e4(self, seed, n_tr, n_te, test_frac=0.3):
        """Fixed test set; natural train set; deduplicated train set of equal size and class mix."""
        rng = np.random.RandomState(seed)
        p = rng.permutation(self.N); cut = int(self.N * test_frac)
        te_pool, tr_pool = p[:cut], p[cut:]
        q  = rng.choice(te_pool, n_te, replace=False)
        tr = rng.choice(tr_pool, min(n_tr, len(tr_pool)), replace=False)
        hq = np.unique(self.h[q])
        lk = np.isin(self.h[tr], hq)
        leak = float(np.isin(self.h[q], self.h[tr]).mean())
        keep = tr[~lk]
        clean = tr_pool[~np.isin(self.h[tr_pool], hq)]
        clean = clean[~np.isin(clean, keep)]
        yc, repl = self.y[clean], []
        for c, k in zip(*np.unique(self.y[tr[lk]], return_counts=True)):
            pool = clean[yc == c]
            if len(pool) < k: pool = clean
            repl.append(rng.choice(pool, k, replace=False))
        tr_d = np.concatenate([keep] + repl) if repl else keep.copy()
        assert not np.isin(self.h[tr_d], hq).any()
        return dict(q=q, tr=tr, tr_d=tr_d, leak=leak, n_leak_tr=int(lk.sum()), rng=rng)

    @staticmethod
    def plant(base, q, dose, rng):
        k = int(round(len(q) * dose)); tr = base.copy()
        if k: tr[rng.choice(len(base), k, replace=False)] = rng.choice(q, k, replace=False)
        return tr

def macro_f1(yt, yp):
    return float(f1_score(yt, yp, labels=np.unique(yt), average='macro', zero_division=0))

# ------------------------------------------------------------------ learners
def knn64(Xtr, ytr, Xte, seed=0):
    A = torch.from_numpy(Xtr).to(DEV, torch.float64)
    mu, sd = A.mean(0), A.std(0).clamp_min(1e-12); A = (A - mu) / sd
    an = (A * A).sum(1)
    chunk = max(32, int(1.5e9 / (8 * A.shape[0])))
    out = np.empty(len(Xte), dtype=np.int64); ex0 = 0
    for i in range(0, len(Xte), chunk):
        b = (torch.from_numpy(Xte[i:i+chunk]).to(DEV, torch.float64) - mu) / sd
        d2 = an.unsqueeze(0) - 2.0 * (b @ A.T) + (b * b).sum(1, keepdim=True)
        cand = d2.topk(min(8, A.shape[0]), dim=1, largest=False).indices
        ex = ((A[cand] - b.unsqueeze(1)) ** 2).sum(-1)
        mn = ex.min(1, keepdim=True).values
        big = torch.full_like(cand, 2**62)
        out[i:i+chunk] = torch.where(ex <= mn, cand, big).min(1).values.cpu().numpy()
        ex0 += int((mn.squeeze(1) == 0).sum()); del b, d2, cand, ex
    del A, an; torch.cuda.empty_cache()
    return ytr[out], {'exact0': ex0 / len(Xte)}

def lgbm(Xtr, ytr, Xte, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1, n_jobs=-1,
                           verbose=-1, random_state=seed, class_weight='balanced')
    m.fit(Xtr.astype(np.float32), ytr)
    return m.predict(Xte.astype(np.float32)), {}

def rf(Xtr, ytr, Xte, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed,
                               class_weight='balanced_subsample')
    m.fit(Xtr.astype(np.float32), ytr)
    return m.predict(Xte.astype(np.float32)), {}

def logreg(Xtr, ytr, Xte, seed=0):
    """Multinomial L2 logistic regression (sklearn objective, C=1, balanced weights), L-BFGS on GPU."""
    cls = np.unique(ytr); yi = torch.from_numpy(np.searchsorted(cls, ytr)).to(DEV)
    X = torch.from_numpy(Xtr).to(DEV, torch.float64)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-12); X = (X - mu) / sd
    K, n = len(cls), len(ytr)
    w = (n / (K * torch.bincount(yi, minlength=K).double()))[yi]
    W = torch.zeros(X.shape[1], K, dtype=torch.float64, device=DEV, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=400, history_size=20,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-10)
    def closure():
        opt.zero_grad()
        loss = (w * torch.nn.functional.cross_entropy(X @ W + b, yi, reduction='none')).sum() \
               + 0.5 * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        Xt = (torch.from_numpy(Xte).to(DEV, torch.float64) - mu) / sd
        p = (Xt @ W + b).argmax(1).cpu().numpy()
    n_iter = opt.state[opt._params[0]]['n_iter']
    del X, Xt, W, b; torch.cuda.empty_cache()
    return cls[p], {'lbfgs_iter': int(n_iter)}

def mlp(Xtr, ytr, Xte, seed=0, early_stop=True, max_epochs=200, patience=15, bs=512,
        lr=1e-3, alpha=1e-4):
    """128-64 ReLU MLP, Adam. early_stop=True: 10% validation, patience 15, best weights restored.
       early_stop=False: stop when training loss fails to improve by 1e-4 for 15 epochs."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    cls = np.unique(ytr); Y = torch.from_numpy(np.searchsorted(cls, ytr)).to(DEV)
    X = torch.from_numpy(Xtr).to(DEV, torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6); X = (X - mu) / sd
    n = len(ytr); idx = torch.from_numpy(rng.permutation(n)).to(DEV)
    nv = int(0.1 * n) if early_stop else 0
    vi, ti = idx[:nv], idx[nv:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, len(cls))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=alpha)
    best, best_state, bad, best_loss, ep_used, tl = -1.0, None, 0, np.inf, 0, np.nan
    for ep in range(max_epochs):
        net.train(); perm = ti[torch.randperm(len(ti), device=DEV)]; tot = torch.zeros((), device=DEV)
        for s in range(0, len(perm), bs):
            j = perm[s:s+bs]; opt.zero_grad()
            l = torch.nn.functional.cross_entropy(net(X[j]), Y[j]); l.backward(); opt.step()
            tot += l.detach() * len(j)
        tl = (tot / len(perm)).item(); ep_used = ep + 1
        if early_stop:
            net.eval()
            with torch.no_grad(): acc = (net(X[vi]).argmax(1) == Y[vi]).float().mean().item()
            if acc > best + 1e-4:
                best, bad = acc, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else: bad += 1
        else:
            if tl < best_loss - 1e-4: best_loss, bad = tl, 0
            else: bad += 1
        if bad >= patience: break
    if early_stop and best_state is not None: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        Xt = (torch.from_numpy(Xte).to(DEV, torch.float32) - mu) / sd
        p = torch.cat([net(Xt[s:s+65536]).argmax(1) for s in range(0, len(Xt), 65536)]).cpu().numpy()
        tra = torch.cat([net(X[s:s+65536]).argmax(1) for s in range(0, len(X), 65536)])
        tracc = (tra == Y).float().mean().item()
    del X, Xt, net; torch.cuda.empty_cache()
    return cls[p], {'epochs': ep_used, 'train_loss': tl, 'train_acc': tracc,
                    'hit_cap': ep_used == max_epochs}

def mlp_es(a, b, c, seed=0):   return mlp(a, b, c, seed, early_stop=True)
def mlp_noes(a, b, c, seed=0): return mlp(a, b, c, seed, early_stop=False)
LEARNERS = {'1-NN': knn64, 'RandomForest': rf, 'LightGBM': lgbm, 'MLP-ES': mlp_es,
            'MLP-noES': mlp_noes, 'LogReg': logreg}

# ------------------------------------------------------------------ statistics
def ci_t(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    if n < 2: return m, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return m, m - h, m + h

def paired(a, b, margin=MARGIN):
    """a, b in percentage points, paired by seed (same test set). Returns t, Wilcoxon, TOST."""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    m, lo, hi = ci_t(d); s = d.std(ddof=1); se = s / np.sqrt(n)
    p_t = float(stats.ttest_1samp(d, 0).pvalue) if s > 0 else (0.0 if m != 0 else 1.0)
    p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    if se == 0: p_tost = 0.0 if abs(m) < margin else 1.0
    else:
        p1 = 1 - stats.t.cdf((m + margin) / se, n - 1); p2 = stats.t.cdf((m - margin) / se, n - 1)
        p_tost = float(max(p1, p2))
    return dict(mean=float(m), lo=float(lo), hi=float(hi), sd=float(s), p_t=p_t, p_w=p_w,
                p_tost=p_tost, dz=float(m / s) if s > 0 else np.nan, n=int(n))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

# === OVERRIDES v2 ===
def knn64(Xtr, ytr, Xte, seed=0):
    """Exact 1-NN: direct Euclidean distances in float64 (no ||a||^2-2ab+||b||^2 expansion).
       Ties at the minimum distance resolved by the lowest training index."""
    A = torch.from_numpy(Xtr).to(DEV, torch.float64)
    mu, sd = A.mean(0), A.std(0).clamp_min(1e-12); A = (A - mu) / sd
    chunk = max(16, int(1.0e9 / (8 * A.shape[0])))
    out = np.empty(len(Xte), dtype=np.int64); ex0 = 0
    for i in range(0, len(Xte), chunk):
        b = (torch.from_numpy(Xte[i:i+chunk]).to(DEV, torch.float64) - mu) / sd
        d = torch.cdist(b, A, compute_mode='donot_use_mm_for_euclid_dist')
        mn = d.min(1, keepdim=True).values
        out[i:i+chunk] = (d <= mn).to(torch.uint8).argmax(1).cpu().numpy()
        ex0 += int((mn == 0).sum()); del b, d, mn
    del A; torch.cuda.empty_cache()
    return ytr[out], {'exact0': ex0 / len(Xte)}

def logreg(Xtr, ytr, Xte, seed=0, max_iter=3000):
    cls = np.unique(ytr); yi = torch.from_numpy(np.searchsorted(cls, ytr)).to(DEV)
    X = torch.from_numpy(Xtr).to(DEV, torch.float64)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-12); X = (X - mu) / sd
    K, n = len(cls), len(ytr)
    w = (n / (K * torch.bincount(yi, minlength=K).double()))[yi]
    W = torch.zeros(X.shape[1], K, dtype=torch.float64, device=DEV, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-10)
    def closure():
        opt.zero_grad()
        loss = (w * torch.nn.functional.cross_entropy(X @ W + b, yi, reduction='none')).sum() \
               + 0.5 * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        Xt = (torch.from_numpy(Xte).to(DEV, torch.float64) - mu) / sd
        p = (Xt @ W + b).argmax(1).cpu().numpy()
    it = int(opt.state[opt._params[0]]['n_iter'])
    del X, Xt, W, b; torch.cuda.empty_cache()
    return cls[p], {'lbfgs_iter': it, 'converged': it < max_iter}

LEARNERS = {'1-NN': knn64, 'RandomForest': rf, 'LightGBM': lgbm, 'MLP-ES': mlp_es,
            'MLP-noES': mlp_noes, 'LogReg': logreg}
