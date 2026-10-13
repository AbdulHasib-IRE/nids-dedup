"""Command line: audit one or more CSV files, or write a duplicate-aware split.

  nidsdedup audit data/*.csv --label Label [--source file|<column>] [--json out.json]
  nidsdedup split data/*.csv --label Label --test-size 0.3 --seed 42 --out split.npz
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from .core import split_columns, row_hash, audit_hashes, per_class
from .split import group_split, leakage_rate


def _read(paths, label, source, id_cols, port_cols, chunksize):
    H, HP, Y, S, n_hdr = [], [], [], [], 0
    master = None
    for p in paths:
        for chunk in pd.read_csv(p, dtype=str, chunksize=chunksize, low_memory=False, encoding='latin-1'):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            if label not in chunk.columns:
                sys.exit(f'label column {label!r} not found in {p}')
            hdr = chunk[label].astype(str).str.strip() == label          # repeated header rows
            n_hdr += int(hdr.sum()); chunk = chunk.loc[~hdr]
            feats, ports = split_columns(chunk.columns, [label] + ([source] if source not in (None, 'file') else []),
                                         id_cols, port_cols)
            master = master or feats
            for c in master:
                if c not in chunk.columns: chunk[c] = np.nan     # schema drift: missing field -> <NA>
            H.append(row_hash(chunk, master))
            HP.append(row_hash(chunk, master + ports) if ports else H[-1])
            Y.append(chunk[label].astype(str).str.strip().to_numpy())
            S.append(np.full(len(chunk), os.path.basename(p)) if source in (None, 'file')
                     else chunk[source].astype(str).to_numpy())
    return np.concatenate(H), np.concatenate(HP), np.concatenate(Y), np.concatenate(S), master, n_hdr


def main(argv=None):
    ap = argparse.ArgumentParser(prog='nidsdedup', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('audit', 'split'):
        p = sub.add_parser(name)
        p.add_argument('files', nargs='+'); p.add_argument('--label', required=True)
        p.add_argument('--source', default='file', help="'file' (default) or a column naming the capture source")
        p.add_argument('--drop', nargs='*', default=None, help='identifier columns (default: built-in list)')
        p.add_argument('--ports', nargs='*', default=None, help='port columns for the sensitivity hash')
        p.add_argument('--chunksize', type=int, default=1_000_000)
        if name == 'audit': p.add_argument('--json', default=None)
        else:
            p.add_argument('--test-size', type=float, default=0.3); p.add_argument('--seed', type=int, default=42)
            p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    files = sorted({f for pat in a.files for f in glob.glob(pat)})
    if not files: sys.exit('no input files')
    h, hp, y, s, feats, n_hdr = _read(files, a.label, a.source, a.drop, a.ports, a.chunksize)
    if a.cmd == 'audit':
        res = {'files': len(files), 'features': len(feats), 'repeated_header_rows': n_hdr,
               'policy': audit_hashes(h, y, s)[0], 'policy_plus_ports': audit_hashes(hp, y, s)[0],
               'per_class': per_class(h, y).reset_index().to_dict('records')}
        pol = res['policy']
        print(f"rows={pol['rows']:,}  files={len(files)}  features={len(feats)}  repeated headers removed={n_hdr}")
        for k in ('dup_pct', 'redund_pct', 'incons_pct', 'ceiling_pct'):
            print(f"  {k:<12} policy={pol[k]:8.3f}   +ports={res['policy_plus_ports'][k]:8.3f}")
        print(f"  max_group={pol['max_group']:,}  dup_groups={pol['dup_groups']:,}  cross_class={pol['groups_cross_class']:,}  "
              f"cross_source={pol['groups_cross_source']:,}  within_source={pol['groups_within_source']:,}")
        if a.json: json.dump(res, open(a.json, 'w'), indent=1, default=str)
    else:
        tr, te = group_split(h, y, a.test_size, a.seed)
        np.savez_compressed(a.out, train=tr, test=te)
        print(f'train={len(tr):,} test={len(te):,} leakage={leakage_rate(h, tr, te)*100:.4f}%  -> {a.out}')


if __name__ == '__main__':
    main()
