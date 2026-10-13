"""Generate LaTeX table bodies for the new analyses directly from the result CSVs (no hand transcription)."""
import json, numpy as np, pandas as pd
import os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data')
P = DATA; OUT = os.path.join(HERE, '..', 'tables')
os.makedirs(OUT, exist_ok=True)
ORDER = ['NSL-KDD', 'CIC IoT-DIAD 2024', 'N-BaIoT', 'Bot-IoT', 'CICIDS2017', 'CSE-CIC-IDS2018', 'UNSW-NB15']
LE6 = ['1-NN', 'RandomForest', 'LightGBM', 'MLP-ES', 'MLP-noES', 'LogReg']
LN = {'1-NN': '1-NN', 'RandomForest': 'RF', 'LightGBM': 'LightGBM', 'MLP-ES': 'MLP-ES', 'MLP-noES': 'MLP-noES', 'LogReg': 'LogReg'}

def f(x, d=2):
    s = f'{x:.{d}f}'
    if s.startswith('-'):
        if float(s) == 0: s = s[1:]
        else: return f'$-${s[1:]}'
    return s

def mark(r, ecol='p_tost1_holm'):
    s = ('S' if r['p_t_holm'] < .05 else '') + ('E' if r[ecol] < .05 else '')
    return (' ' + s) if s else ''

# ---------------------------------------------------------------- E8: twin and replacement effects
S8 = pd.read_csv(f'{P}/e8_summary_ALL.csv').set_index(['dataset', 'learner', 'contrast'])
lines = []
for block, c in [('Twin effect (placebo minus deduplicated)', 'twin'), ('Replacement effect (natural minus placebo)', 'replacement')]:
    lines.append(r'\multicolumn{7}{@{}l}{\textit{' + block + r'}} \\')
    for d in ORDER:
        cells = [f(S8.loc[(d, l, c), 'mean']) + mark(S8.loc[(d, l, c)]) for l in LE6]
        lines.append(f'{d} & ' + ' & '.join(cells) + r' \\')
    if c == 'twin': lines.append(r'\midrule')
open(f'{OUT}/tab_e8.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- E8 appendix: consistent / conflicting removal (long format)
lines = []
for d in ORDER:
    for i, l in enumerate(LE6):
        cells = []
        for c in ['consistent', 'conflicting']:
            if (d, l, c) not in S8.index: cells.append('--'); continue
            r = S8.loc[(d, l, c)]
            cells.append(f(r['mean']) + ' [' + f(r['lo']) + ', ' + f(r['hi']) + ']' + mark(r))
        lines.append((d if i == 0 else '') + f' & {LN[l]} & ' + ' & '.join(cells) + r' \\')
    if d != ORDER[-1]: lines.append(r'\addlinespace')
open(f'{OUT}/tab_e8cc.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- margins
lines = []
for name, lab in [('uniform', 'Uniform (E4)'), ('matched', 'Size-matched (E4c)'), ('size', 'Training size (E5m)')]:
    T = pd.read_csv(f'{P}/m4_{name}.csv'); S = T.p_t_holm < .05
    E = {m: T[f'p_tost_{m}_holm'] < .05 for m in ['0.5', '1.0', '2.0']}
    row = [lab, str(len(T)), str(int(S.sum()))] + [str(int(E[m].sum())) for m in E] + [str(int((~S & ~E[m]).sum())) for m in E]
    lines.append(' & '.join(row) + r' \\')
open(f'{OUT}/tab_margins.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- E9
S9 = pd.read_csv(f'{P}/e9_summary_ALL.csv').set_index(['dataset', 'learner', 'contrast'])
O9 = ['NSL-KDD', 'CIC IoT-DIAD 2024', 'N-BaIoT', 'Bot-IoT', 'UNSW-NB15']
lines = []
for d in O9:
    for i, l in enumerate(['1-NN', 'LightGBM', 'LogReg']):
        a, b, g = S9.loc[(d, l, 'rand_nat-ded')], S9.loc[(d, l, 'src_nat-ded')], S9.loc[(d, l, 'rand_ded-src_ded')]
        first = (f'{d} & {a["classes"]} & {a["rand_twin"]:.2f} / {a["src_twin"]:.2f}' if i == 0 else ' & & ')
        cell = lambda r: f(r['mean']) + ' [' + f(r['lo']) + ', ' + f(r['hi']) + ']' + mark(r)
        lines.append(first + f' & {l} & ' + cell(a) + ' & ' + cell(b) + ' & ' + f(g['mean']) + ' [' + f(g['lo']) + ', ' + f(g['hi']) + ']' + (' S' if g['p_t_holm'] < .05 else '') + r' \\')
    if d != O9[-1]: lines.append(r'\addlinespace')
open(f'{OUT}/tab_e9.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- E10 main: recovery per learner and arm
N = json.load(open(f'{P}/numbers_v2.json'))['e10_recovery']
L5 = ['1-NN', 'RandomForest', 'LightGBM', 'MLP-ES', 'LogReg']
lines = []
for l in L5:
    lines.append(LN[l] + ' & ' + ' & '.join(f'{N[a]["mean"][l]:.3f} ({N[a]["ranks"][l]:.2f})' for a in ['d05', 'd10', 'd25', 'bal10']) + r' \\')
lines.append(r'\midrule')
def pfmt(p):
    if p >= 0.001: return f'{p:.3f}'
    m, e = f'{p:.1e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
lines.append(r'Friedman $\chi^2$ ($p$) & ' + ' & '.join(f'{N[a]["chi2"]:.2f} ({pfmt(N[a]["p"])})' for a in ['d05', 'd10', 'd25', 'bal10']) + r' \\')
open(f'{OUT}/tab_e10.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- E10 appendix: absolute gains and headroom
S10 = pd.read_csv(f'{P}/e10_summary_ALL.csv').set_index(['dataset', 'learner', 'arm'])
lines = []
for d in ORDER:
    for i, l in enumerate(L5):
        r0 = S10.loc[(d, l, 'd05')]
        cells = []
        for a in ['d05', 'd10', 'd25', 'bal10']:
            r = S10.loc[(d, l, a)]; cells.append(f(r['gain']) + (r'$^{*}$' if r['p_t_holm'] < .05 else ''))
        lines.append((d if i == 0 else '') + f' & {LN[l]} & {r0["f1_dose0"]:.2f} & {r0["headroom"]:.2f} & ' + ' & '.join(cells) + r' \\')
    if d != ORDER[-1]: lines.append(r'\addlinespace')
open(f'{OUT}/tab_e10full.tex', 'w').write('\n'.join(lines) + '\n')
print(open(f'{OUT}/tab_e8.tex').read()); print(open(f'{OUT}/tab_margins.tex').read()); print(open(f'{OUT}/tab_e9.tex').read())
print(open(f'{OUT}/tab_e10.tex').read()); print(open(f'{OUT}/tab_e10full.tex').read()[:1500]); print(open(f'{OUT}/tab_e8cc.tex').read()[:1500])
