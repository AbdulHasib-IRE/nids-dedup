import numpy as np, pandas as pd
from nidsdedup import (row_hash, audit_hashes, per_class, split_columns, group_split,
                       dedup_training, matched_natural, leakage_rate)


def brute(h, y, s):
    df = pd.DataFrame({'h': h, 'y': y, 's': s}); g = df.groupby('h')
    n = len(df); size = g.size(); ny = g.y.nunique(); ns = g.s.nunique()
    maj = df.groupby(['h', 'y']).size().groupby(level=0).max()
    d = size > 1
    return dict(dup_pct=size[d].sum() / n * 100, redund_pct=(n - len(size)) / n * 100,
                incons_pct=size[ny > 1].sum() / n * 100, max_group=size.max(),
                irreducible_errors=int((size - maj).sum()), groups_cross_class=int((d & (ny > 1)).sum()),
                groups_cross_source=int((d & (ns > 1)).sum()), groups_within_source=int((d & (ns == 1)).sum()))


def test_audit_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(50, 3000)
        h = rng.integers(0, max(2, n // rng.integers(1, 6)), n).astype(np.uint64)
        y = rng.choice(['a', 'b', 'c'], n); s = rng.choice(['f1', 'f2', 'f3', 'f4'], n)
        got, rowdup = audit_hashes(h, y, s); ref = brute(h, y, s)
        for k, v in ref.items(): assert np.isclose(got[k], v), (k, got[k], v)
        assert rowdup.sum() == got['dup_rows']


def test_row_hash_text_identity_and_ids():
    df = pd.DataFrame({'Flow ID': ['x', 'y', 'z'], 'Dst Port': ['80', '443', '80'],
                       'a': ['1.0', '1.0', '1.00'], 'b': [' 2', '2', '2'], 'Label': ['n', 'n', 'n']})
    f, p = split_columns(df.columns, ['Label'])
    assert f == ['a', 'b'] and p == ['Dst Port']
    h = row_hash(df, f)
    assert h[0] == h[1] and h[0] != h[2]          # whitespace stripped; '1.0' != '1.00' (text identity)
    hp = row_hash(df, f + p); assert hp[0] != hp[1]
    df2 = df.copy(); df2.loc[0, 'a'] = None; assert row_hash(df2, f)[0] != h[0]


def test_per_class():
    pc = per_class(np.array([1, 1, 2, 3, 3, 3], np.uint64), np.array(['a', 'a', 'a', 'b', 'b', 'b']))
    assert np.isclose(pc.loc['a', 'dup_pct'], 200 / 3) and pc.loc['b', 'distinct'] == 1


def test_group_split_has_no_leakage():
    rng = np.random.default_rng(1); h = rng.integers(0, 500, 5000).astype(np.uint64); y = rng.choice(list('abcd'), 5000)
    tr, te = group_split(h, y, 0.3, seed=3)
    assert leakage_rate(h, tr, te) == 0 and len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == 5000
    assert 0.2 < len(te) / 5000 < 0.4


def test_dedup_training_and_matching():
    rng = np.random.default_rng(2); n = 6000
    h = rng.integers(0, 2000, n).astype(np.uint64); y = rng.choice(list('ab'), n)
    p = rng.permutation(n); te, tr, pool = p[:1000], p[1000:3000], p[3000:]
    ded, short = dedup_training(h, y, tr, te, pool, seed=0)
    assert leakage_rate(h, ded, te) == 0
    assert len(ded) + short == len(tr)
    m = matched_natural(y, tr, ded, seed=0)
    assert all((y[m] == c).sum() == (y[ded] == c).sum() for c in 'ab')
