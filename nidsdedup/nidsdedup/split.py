"""Duplicate-aware evaluation splits.

group_split           no hash appears on both sides (the recommended default split).
dedup_training        keep a fixed test set; remove training rows that are exact twins of
                      test rows, refill from the same class, and report any shortfall.
matched_natural       subsample a natural training set to the class counts of a deduplicated
                      one, so the two can be compared at equal size and class mix.
leakage_rate          share of test rows with an exact twin in a training set.
"""
from __future__ import annotations
import numpy as np


def leakage_rate(h, train_idx, test_idx) -> float:
    h = np.asarray(h)
    return float(np.isin(h[test_idx], h[train_idx]).mean()) if len(test_idx) else 0.0


def group_split(h, y=None, test_size=0.3, seed=0, stratify=True):
    """Split row indices so that all copies of a record fall on the same side.

    With stratify=True, unique hashes are assigned per majority label, which keeps the class
    mix of both sides close to the full data. Returns (train_idx, test_idx)."""
    h = np.asarray(h); rng = np.random.default_rng(seed)
    uh, inv = np.unique(h, return_inverse=True)
    if stratify and y is not None:
        y = np.asarray(y)
        # majority label of each hash group
        codes, yc = np.unique(y, return_inverse=True)
        cnt = np.zeros((len(uh), len(codes)), dtype=np.int64)
        np.add.at(cnt, (inv, yc), 1)
        gy = cnt.argmax(1)
        test_groups = np.zeros(len(uh), dtype=bool)
        for c in range(len(codes)):
            g = np.flatnonzero(gy == c); rng.shuffle(g)
            # take groups until the class's test share reaches test_size (by rows)
            rows = cnt[g].sum(1); cut = np.searchsorted(np.cumsum(rows), test_size * rows.sum())
            test_groups[g[:cut + 1 if cut < len(g) else cut]] = True
    else:
        g = rng.permutation(len(uh)); test_groups = np.zeros(len(uh), dtype=bool)
        test_groups[g[:int(round(test_size * len(uh)))]] = True
    is_test = test_groups[inv]
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def dedup_training(h, y, train_idx, test_idx, pool_idx=None, seed=0):
    """Remove training rows whose hash occurs in the test set and refill each class from pool_idx
    (rows not in train or test whose hash is not in the test set). Returns (dedup_idx, shortfall),
    where shortfall counts rows that could not be refilled within their class."""
    h, y = np.asarray(h), np.asarray(y); rng = np.random.default_rng(seed)
    hq = np.unique(h[test_idx]); leak = np.isin(h[train_idx], hq)
    keep = train_idx[~leak]
    if pool_idx is None:
        pool_idx = np.setdiff1d(np.arange(len(h)), np.r_[train_idx, test_idx])
    clean = pool_idx[~np.isin(h[pool_idx], hq)]; clean = clean[~np.isin(clean, keep)]
    repl, short = [], 0
    for c, k in zip(*np.unique(y[train_idx[leak]], return_counts=True)):
        pool = clean[y[clean] == c]
        take = min(k, len(pool)); short += int(k - take)
        if take: repl.append(rng.choice(pool, take, replace=False))
    out = np.concatenate([keep] + repl) if repl else keep.copy()
    assert not np.isin(h[out], hq).any()
    return out, short


def matched_natural(y, natural_idx, dedup_idx, seed=0):
    """Per-class random subsample of natural_idx with the class counts of dedup_idx."""
    y = np.asarray(y); rng = np.random.default_rng(seed); out = []
    for c, k in zip(*np.unique(y[dedup_idx], return_counts=True)):
        pool = natural_idx[y[natural_idx] == c]
        out.append(rng.choice(pool, k, replace=False))
    return np.concatenate(out) if out else natural_idx[:0]
