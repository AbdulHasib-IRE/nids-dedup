# Reproducing the tables and figures of the paper

This folder rebuilds every figure (Fig1-Fig18) and every table generated for the robustness analyses
from the result files exported by the experiment notebook. No dataset or GPU is needed.

```
pip install numpy pandas scipy matplotlib pillow
python scripts/analysis_v2.py      # margin sensitivity, effect sizes, summary numbers -> data/m4_*.csv, data/numbers_v2.json
python scripts/make_tables_v2.py   # LaTeX bodies of the new tables -> tables/
python scripts/make_figs2.py       # Fig1-Fig14 -> figures/ (1000 dpi PNG)
python scripts/make_figs3.py       # revised Fig1 and Fig15-Fig18 -> figures/ (run after make_figs2.py)
```

`data/` holds the summary and per-seed result files (CSV/JSON) of all experiments (E4-E10), the audit
tables and the sensitivity checks. The per-seed checkpoints written during the runs are in
`results/phase2/` of the repository.
