# Exact duplicates in IoT and enterprise intrusion detection benchmarks

Code, results and audit tool for the study "Exact Duplicates Inflate Intrusion Detection Scores for Some Learners and
Benchmarks but Not Others: Evidence from Seven IoT and Enterprise Datasets"
(A. Hasib, S. C. Das, Md. Ashiqussalehin; University of Frontier Technology, Bangladesh).

## Contents
| Folder | What it holds |
|---|---|
| `nidsdedup/` | The audit and duplicate-aware splitting tool (version 0.1.0); `pip install ./nidsdedup` |
| `code/nids_lib.py` | Shared data loading, learners and statistics used by every experiment cell |
| `code/statistics.py` | Paired and equivalence tests, Holm correction and effect sizes computed from the result files |
| `notebook/` | The Colab notebook with every experiment cell, outputs cleared (cells that only drafted an early manuscript and its figures are left out) |
| `results/phase2/` | Summary and per-seed result files of the experiments E4-E10 and the audit checks |
| `results/derived/` | Further result files and the output of `code/statistics.py` |
| `manifest/` | Where each dataset was obtained, the file names and sizes used; package versions |

## Datasets
The seven benchmarks (NSL-KDD, CICIDS2017, UNSW-NB15, CSE-CIC-IDS2018, Bot-IoT, N-BaIoT, CIC IoT-DIAD 2024)
are public and are not included here. `manifest/dataset_sources.csv` lists where each one was obtained and
the file names and sizes that were used.

## Reproduce
1. Statistics from the stored results (no dataset or GPU needed):
   ```
   pip install numpy pandas scipy
   python code/statistics.py
   ```
   This rewrites `results/derived/m4_*.csv` and `results/derived/numbers_v2.json`.
2. Full experiments: download the datasets from the sources in the manifest and run the notebook cells in order
   on a GPU runtime (the experiments ran on an NVIDIA A100; see `manifest/`).
3. The tool's tests: `pip install ./nidsdedup pytest` and `pytest nidsdedup/tests`.

`SHA256SUMS.txt` lists a checksum for every file.

## License
Code: MIT (see `LICENSE`). The datasets keep their original licenses.
