# Exact duplicates in IoT and enterprise intrusion detection benchmarks

Code, results and audit tool for the paper "Exact Duplicates Inflate Intrusion Detection Scores for Some Learners and
Benchmarks but Not Others: Evidence from Seven IoT and Enterprise Datasets"
(A. Hasib, S. C. Das, Md. Ashiqussalehin; University of Frontier Technology, Bangladesh).

## Contents
| Folder | What it holds |
|---|---|
| `nidsdedup/` | The audit and duplicate-aware splitting tool (version 0.1.0); `pip install ./nidsdedup` |
| `code/nids_lib.py` | Shared data loading, learners and statistics used by every experiment cell |
| `notebook/` | The Colab notebook with all cells and their printed outputs |
| `results/phase2/` | Summary and per-seed result files of the experiments E4-E10 and the audit checks |
| `paper_reproduction/` | Scripts and data that rebuild every figure and the robustness tables (no GPU needed) |
| `manifest/` | Where each dataset was obtained, the file names and sizes used; package versions |

## Datasets
The seven benchmarks (NSL-KDD, CICIDS2017, UNSW-NB15, CSE-CIC-IDS2018, Bot-IoT, N-BaIoT, CIC IoT-DIAD 2024)
are public and are not included here. `manifest/dataset_sources.csv` lists where each one was obtained and
the file names and sizes that were used.

## Reproduce
1. Figures and tables only: see `paper_reproduction/README.md`.
2. Full experiments: download the datasets from the sources in the manifest and run the notebook cells in order
   on a GPU runtime (the experiments ran on an NVIDIA A100; see `manifest/`).

## License
Code: MIT (see `LICENSE`). The datasets keep their original licenses.
