# nidsdedup

Exact-duplicate audit and duplicate-aware train/test splits for tabular network intrusion
detection (NIDS) benchmarks such as NSL-KDD, CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15, Bot-IoT,
N-BaIoT and CIC IoT-DIAD 2024.

A record's identity is the text of its feature fields after identifier columns (IP addresses,
ports, flow IDs, timestamps, sequence numbers, MAC addresses) are removed. Hashing text rather
than parsed numbers keeps the audit independent of dtype. Missing values are hashed as `<NA>`
and never imputed.

## Install
    pip install .            # requires numpy and pandas

## Audit a benchmark
    nidsdedup audit "CICIDS2017/*.csv" --label Label --json audit.json

Reports, for the identifier policy and for policy + destination port:
duplicate rate, redundancy, label-inconsistent rate, largest group, accuracy ceiling
(majority label per group), group origin (cross-class / cross-source / within-source), and
per-class duplicate rate and distinct-vector share. Repeated header rows are removed.

## Duplicate-aware split
    nidsdedup split "CICIDS2017/*.csv" --label Label --test-size 0.3 --seed 42 --out split.npz

All copies of a record go to the same side, so no test row has an exact twin in training.

## Python API
```python
from nidsdedup import split_columns, row_hash, audit_hashes, per_class
from nidsdedup import group_split, dedup_training, matched_natural, leakage_rate

feats, ports = split_columns(df.columns, label_cols=['Label'])
h = row_hash(df, feats)
stats, is_dup = audit_hashes(h, df['Label'], source=df['file'])
train, test = group_split(h, df['Label'], test_size=0.3, seed=42)
```
`dedup_training` keeps a fixed test set and removes the training twins of test rows,
refilling from the same class; `matched_natural` subsamples the natural training set to the
same class counts, so both training sets can be compared at equal size.

## Tests
    python -m pytest tests

## License
MIT
