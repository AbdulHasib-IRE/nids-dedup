"""nidsdedup: exact-duplicate audit and duplicate-aware splits for NIDS benchmarks."""
from .core import (DEFAULT_ID_COLUMNS, DEFAULT_PORT_COLUMNS, split_columns, row_hash,
                   audit_hashes, per_class)
from .split import group_split, dedup_training, matched_natural, leakage_rate
__version__ = '0.1.0'
