"""Exact-duplicate audit for tabular NIDS benchmarks.

A record's identity is the text of its feature fields after identifier columns are removed.
Hashing the text (not parsed numbers) makes the audit independent of dtype: float32 casting
can merge distinct records, and imputation can create duplicates that were never in the data.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

# Identifier fields removed before hashing (matched after lower-casing and removing spaces/underscores).
DEFAULT_ID_COLUMNS = ['Flow ID', 'Src IP', 'Dst IP', 'Source IP', 'Destination IP', 'Src Port', 'Source Port',
                      'Timestamp', 'srcip', 'dstip', 'sport', 'stime', 'ltime', 'stcpb', 'dtcpb', 'pkSeqID',
                      'seq', 'saddr', 'daddr', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'id', 'Unnamed: 0']
# Destination ports are kept out of the policy hash and used only in a sensitivity hash.
DEFAULT_PORT_COLUMNS = ['Destination Port', 'Dst Port', 'dsport', 'dport']
NA_TOKEN = '<NA>'


def _key(c: str) -> str:
    return re.sub(r'\s+', ' ', str(c)).strip().lower().replace(' ', '').replace('_', '')


def split_columns(columns, label_cols, id_cols=None, port_cols=None):
    """Return (feature_columns, port_columns) for a list of column names."""
    ids = {_key(c) for c in (DEFAULT_ID_COLUMNS if id_cols is None else id_cols)}
    ports = {_key(c) for c in (DEFAULT_PORT_COLUMNS if port_cols is None else port_cols)}
    labels = {_key(c) for c in label_cols}
    feats = [c for c in columns if _key(c) not in ids | ports | labels]
    prt = [c for c in columns if _key(c) in ports]
    return feats, prt


def row_hash(frame: pd.DataFrame, columns) -> np.ndarray:
    """64-bit hash of each row's field text. Values are compared as stripped strings; missing -> '<NA>'."""
    s = frame[list(columns)].astype(str).apply(lambda c: c.str.strip())
    s = s.where(frame[list(columns)].notna(), NA_TOKEN)
    return hash_pandas_object(s, index=False).to_numpy('uint64')


def _codes(values) -> np.ndarray:
    return pd.factorize(pd.Series(values), sort=False)[0].astype(np.int64)


def audit_hashes(h, y, source=None) -> tuple[dict, np.ndarray]:
    """Duplicate statistics from row hashes, labels and (optional) source ids.

    Returns (stats, row_is_duplicate). Definitions:
      dup_pct       share of rows in a group of size >= 2
      redund_pct    share of rows that are not the first copy, (n - unique) / n
      incons_pct    share of rows in groups carrying more than one label
      ceiling_pct   best achievable accuracy if each group gets its majority label
      group origin  cross_class (>1 label), cross_source (>1 source), within_source (1 source);
                    cross_class overlaps the other two.
    """
    h = np.asarray(h, dtype=np.uint64); n = len(h)
    yc = _codes(y); pc = _codes(source) if source is not None else np.zeros(n, np.int64)
    o = np.argsort(h, kind='stable'); hs, ys, ps = h[o], yc[o], pc[o]
    st = np.r_[0, np.flatnonzero(hs[1:] != hs[:-1]) + 1]; sz = np.diff(np.r_[st, n])
    gid = np.repeat(np.arange(len(st)), sz)

    def distinct(v):
        K = int(v.max()) + 1
        u, c = np.unique(gid * K + v, return_counts=True)
        g = u // K
        nd = np.bincount(g, minlength=len(st))
        b = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
        return nd, np.maximum.reduceat(c, b)

    ny, ymax = distinct(ys); npv, _ = distinct(ps)
    d = sz > 1; irr = int((sz - ymax).sum())
    rowdup = np.empty(n, dtype=bool); rowdup[o] = np.repeat(d, sz)
    stats = {'rows': int(n), 'unique': int(len(st)), 'dup_rows': int(sz[d].sum()),
             'dup_pct': float(sz[d].sum() / n * 100), 'redund_pct': float((n - len(st)) / n * 100),
             'incons_rows': int(sz[ny > 1].sum()), 'incons_pct': float(sz[ny > 1].sum() / n * 100),
             'max_group': int(sz.max()), 'irreducible_errors': irr, 'ceiling_pct': float((1 - irr / n) * 100),
             'dup_groups': int(d.sum()), 'groups_cross_class': int((d & (ny > 1)).sum()),
             'groups_cross_source': int((d & (npv > 1)).sum()),
             'groups_within_source': int((d & (npv == 1)).sum()),
             'mean_group_size': float(sz[d].mean()) if d.any() else 0.0}
    return stats, rowdup


def per_class(h, y) -> pd.DataFrame:
    """Rows, duplicate rate and distinct-vector share for each class."""
    df = pd.DataFrame({'h': np.asarray(h, np.uint64), 'y': np.asarray(y)})
    size = df.groupby('h')['h'].transform('size')
    out = df.assign(dup=size > 1).groupby('y').agg(rows=('h', 'size'), dup_pct=('dup', 'mean'),
                                                   distinct=('h', 'nunique'))
    out['dup_pct'] *= 100; out['distinct_pct'] = out.distinct / out.rows * 100
    return out.sort_values('rows', ascending=False)
