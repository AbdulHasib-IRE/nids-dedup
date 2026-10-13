"""All manuscript figures, drawn only from the result files in ../data.
300 dpi files; panels are designed at 6.8 x 5.2 in (2,040 x 1,560 px) so that when placed
at 0.47\\columnwidth they print at roughly twice their nominal resolution."""
import os, itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(HERE, '..', 'data'))
OUT = os.path.join(HERE, '..', 'figures'); os.makedirs(OUT, exist_ok=True); DPI = 1000
W, H = 6.8, 5.2
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 1.1, 'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
    'lines.linewidth': 2.0, 'savefig.dpi': DPI, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b',
    'axes.labelcolor': '#0b0b0b'})
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
# learner identity: validated categorical slots 1-6 (fixed order) + distinct markers
LS = {'1-NN': ('#2a78d6', 'o'), 'RandomForest': ('#eb6834', 's'), 'LightGBM': ('#1baf7a', '^'),
      'MLP-ES': ('#eda100', 'D'), 'MLP-noES': ('#e87ba4', 'v'), 'LogReg': ('#008300', 'P')}
LN = {'1-NN': '1-NN', 'RandomForest': 'RF', 'LightGBM': 'LightGBM', 'MLP-ES': 'MLP-ES', 'MLP-noES': 'MLP-noES', 'LogReg': 'LogReg'}
LE = list(LS)
C7, C8 = '#4a3aa7', '#e34948'          # measure colours for audit figures (slots 7, 8)
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#f0efec', '#e34948', '#8f1d1c'])
t5 = pd.read_csv('table5_6_tier1_final.csv'); pol = t5[t5.hash == 'policy'].set_index('dataset')
prt = t5[t5.hash == 'ports'].set_index('dataset')
ORDER = pol.sort_values('dup_pct').index.tolist()
BS = {'CIC IoT-DIAD 2024': 'CIC IoT-DIAD'}; bs = lambda d: BS.get(d, d)
BM = dict(zip(ORDER, ['o', 's', '^', 'D', 'v', 'P', 'X']))

def check(fig, name):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); boxes = []
    for ax in fig.axes:
        lim = {'x': sorted(ax.get_xlim()), 'y': sorted(ax.get_ylim())}
        for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title]:
            if t.get_text() and t.get_visible(): boxes.append((t.get_text()[:25], t.get_window_extent(r)))
        for axis, tl in (('x', ax.get_xticklabels()), ('y', ax.get_yticklabels())):
            for t in tl:
                v = t.get_position()[0 if axis == 'x' else 1]
                if t.get_text() and lim[axis][0] - 1e-9 <= v <= lim[axis][1] + 1e-9: boxes.append((t.get_text()[:25], t.get_window_extent(r)))
        if ax.get_legend(): boxes.append(('LEGEND', ax.get_legend().get_window_extent(r)))
    bad = [(a[0], b[0]) for a, b in itertools.combinations(boxes, 2) if a[1].overlaps(b[1])
           and a[1].intersection(a[1], b[1]).width > 1 and a[1].intersection(a[1], b[1]).height > 1]
    fb = fig.bbox; out = [n for n, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1]
    for ax in fig.axes:   # legend vs data markers
        lg = ax.get_legend()
        if not lg: continue
        lb = lg.get_window_extent(r)
        for c in ax.collections + ax.lines:
            try: pts = np.asarray(c.get_offsets() if hasattr(c, 'get_offsets') and len(c.get_offsets()) else c.get_xydata(), float)
            except Exception: continue
            if len(pts) and any(lb.contains(*ax.transData.transform(p)) for p in pts if np.all(np.isfinite(p))):
                bad.append(('LEGEND', 'data')); break
    print(f'{name:6s} overlaps={bad if bad else "none"} outside={out if out else "none"}')

def save(fig, name, tight=True):
    if tight: fig.tight_layout(pad=0.4)
    check(fig, name); fig.savefig(f'{OUT}/{name}.png', dpi=DPI); plt.close(fig)

def sgn(v):
    t = f'{v:+.1f}'
    return '0.0' if t in ('+0.0', '-0.0') else t
def hgrid(ax): ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
def vgrid(ax): ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
def mk_handles(names, title=None):
    return [Line2D([], [], color=LS[l][0], marker=LS[l][1], ls='', ms=9, label=LN[l]) for l in names]
def bm_handles(): return [Line2D([], [], color=INK, mfc='white', marker=BM[d], ls='', ms=9, mew=1.2, label=bs(d)) for d in ORDER]

# ======================= Fig 1: pipeline overview =======================
fig, ax = plt.subplots(figsize=(W, 6.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
def box(x, y, w, h, txt, fc='#f0efec', ec=INK2, fs=12, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.6', fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal')
def arrow(x0, y0, x1, y1): ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.6, mutation_scale=16))
box(4, 88, 92, 9, '7 public NIDS benchmarks, full populations\n121,704,461 records', bold=True)
box(4, 72, 92, 9.5, 'Identifier removal (IPs, ports, IDs, timestamps)\ntext hash of each record; ports in a second hash')
box(0.5, 52, 47, 13, 'RQ1  Duplicate audit\nrates, label conflicts,\nceiling, group origin', fc='#e3e0f4')
box(52.5, 52, 47, 13, 'Fixed test set\n70/30 pool split per seed', fc='#dceaf9')
box(52.5, 30, 47, 15, 'RQ2  Natural vs deduplicated\ntraining, same test set:\nuniform and size-matched\nRQ3  training-size curve', fc='#dceaf9', fs=11)
box(0.5, 30, 47, 15, 'RQ4  Planted test copies\n0 / 50 / 100% dose,\nconstant training size;\nMLP with and without ES', fc='#fbe1dd', fs=11)
box(4, 12, 92, 10, '6 learners x 7 benchmarks x 10 seeds\n1-NN, RF, LightGBM, MLP-ES, MLP-noES, LogReg')
box(4, 0.5, 92, 7.5, 'Paired t, TOST (+/-1 pp), Holm, Friedman/Nemenyi', fs=12)
arrow(50, 88, 50, 82); arrow(35, 72, 24, 66); arrow(65, 72, 76, 66); arrow(76, 52, 76, 46)
arrow(62, 52, 30, 46); arrow(76, 30, 62, 23); arrow(24, 30, 38, 23); arrow(50, 12, 50, 8.6)
save(fig, 'Fig1', tight=False)

# ======================= Fig 2: duplicate and inconsistent rates =======================
for tag, col, xl in [('a', 'dup_pct', 'Duplicate rows (%)'), ('b', 'incons_pct', 'Label-inconsistent rows (%)')]:
    fig, ax = plt.subplots(figsize=(W, H)); y = np.arange(len(ORDER)); h = 0.38
    ax.barh(y + h / 2, [pol.loc[d, col] for d in ORDER], h, color=C7, label='Identifier policy')
    ax.barh(y - h / 2, [prt.loc[d, col] if d in prt.index else 0 for d in ORDER], h, color=C8, label='Policy + ports')
    for i, d in enumerate(ORDER):
        if d not in prt.index: ax.text(0.01 * ax.get_xlim()[1], y[i] - h / 2, 'no port field', va='center', fontsize=11, color=INK2)
        if pol.loc[d, col] == 0: ax.text(0.01 * ax.get_xlim()[1], y[i] + h / 2, '0', va='center', fontsize=11, color=INK2)
    ax.set_yticks(y, [bs(d) for d in ORDER]); ax.set_xlabel(xl); hgrid(ax)
    ax.legend(loc='lower right', frameon=False); save(fig, f'Fig2{tag}')

# ======================= Fig 3: largest group and irreducible error =======================
fig, ax = plt.subplots(figsize=(W, H)); y = np.arange(len(ORDER))
v = [pol.loc[d, 'max_mult'] for d in ORDER]
ax.barh(y, v, 0.55, color=C7); ax.set_xscale('log'); ax.set_xlim(1, 2e7)
for i, x in enumerate(v): ax.text(x * 1.25, y[i], f'{x:,}', va='center', fontsize=11, color=INK2)
ax.set_yticks(y, [bs(d) for d in ORDER]); ax.set_xlabel('Largest group of identical records'); hgrid(ax); ax.minorticks_off()
save(fig, 'Fig3a')
fig, ax = plt.subplots(figsize=(W, H))
v = [100 - pol.loc[d, 'ceiling_acc_pct'] for d in ORDER]
ax.barh(y, v, 0.55, color=C8)
for i, x in enumerate(v): ax.text(x + 0.02, y[i], f'{x:.2f}', va='center', fontsize=11, color=INK2)
ax.set_xlim(0, 1.5); ax.set_yticks(y, [bs(d) for d in ORDER]); ax.set_xlabel('Irreducible error from label conflicts (%)'); hgrid(ax)
save(fig, 'Fig3b')

# ======================= Fig 4: group origin, per-class duplicate rate =======================
fig, ax = plt.subplots(figsize=(W, H))
ws = [100 * pol.loc[d, 'grp_within_source'] / pol.loc[d, 'dup_groups'] for d in ORDER]
cs = [100 * pol.loc[d, 'grp_cross_source'] / pol.loc[d, 'dup_groups'] for d in ORDER]
cc = [100 * pol.loc[d, 'grp_cross_class'] / pol.loc[d, 'dup_groups'] for d in ORDER]
ax.barh(y, ws, 0.6, color=C7, label='Within one source', edgecolor='white', lw=1.5)
ax.barh(y, cs, 0.6, left=ws, color=C8, label='Across sources', edgecolor='white', lw=1.5)
ax.scatter(cc, y, marker='D', s=70, color=INK, zorder=3, label='Cross-class (overlaps)')
ax.set_yticks(y, [bs(d) for d in ORDER]); ax.set_xlim(0, 100); ax.set_xlabel('Duplicate groups (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.16), ncol=2, frameon=False, fontsize=11.5, columnspacing=1.0)
save(fig, 'Fig4a')
pc = pd.read_csv('per_class_dup_final.csv'); rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(W, H))
for i, d in enumerate(ORDER):
    s = pc[pc.dataset == d]
    ax.scatter(s.dup_pct, i + rng.uniform(-0.2, 0.2, len(s)), s=55, color=C7, marker=BM[d], alpha=0.75, lw=0)
ax.set_yticks(range(len(ORDER)), [bs(d) for d in ORDER]); ax.set_xlim(-3, 103)
ax.set_xlabel('Duplicate rows per class (%)'); hgrid(ax); save(fig, 'Fig4b')

# ======================= Fig 5: distinct vectors, shortcut baseline =======================
dv = pd.read_csv('distinct_vectors_per_class.csv')
fig, ax = plt.subplots(figsize=(W, H))
for i, d in enumerate(ORDER):
    s = dv[dv.dataset == d]
    ax.scatter(s.distinct_pct, i + rng.uniform(-0.2, 0.2, len(s)), s=55, color=C7, marker=BM[d], alpha=0.75, lw=0)
ax.set_xscale('log'); ax.set_xlim(0.01, 200); ax.minorticks_off()
ax.set_xticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
ax.set_yticks(range(len(ORDER)), [bs(d) for d in ORDER]); ax.set_xlabel('Distinct vectors per class (% of rows)'); hgrid(ax)
lo = dv.nsmallest(2, 'distinct_pct')
ax.annotate('54 and 55 vectors\n(FTP-BruteForce,\nSlowHTTPTest)', xy=(lo.distinct_pct.iloc[0], ORDER.index('CSE-CIC-IDS2018')),
            xytext=(0.06, ORDER.index('CSE-CIC-IDS2018') - 2.1), fontsize=11, color=INK2,
            arrowprops=dict(arrowstyle='-', color=INK2, lw=1))
save(fig, 'Fig5a')
sh = pd.read_csv('table10_shortcut_baseline.csv').set_index('dataset')
fig, ax = plt.subplots(figsize=(W, H))
for i, d in enumerate(ORDER):
    a, b = sh.loc[d, 'single_feature_f1'], sh.loc[d, 'lgbm_all_f1']
    ax.plot([a, b], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter(a, i, s=90, color=C8, zorder=2); ax.scatter(b, i, s=90, color=C7, zorder=2)
    ax.text(b + 1.5, i, f'{sh.loc[d, "ratio"]:.2f}', va='center', fontsize=11, color=INK2)
ax.set_yticks(range(len(ORDER)), [bs(d) for d in ORDER]); ax.set_xlim(25, 112); ax.set_xlabel('Macro-F1 (%)'); hgrid(ax)
ax.legend(handles=[Line2D([], [], color=C8, marker='o', ls='', ms=9, label='Best single feature'),
                   Line2D([], [], color=C7, marker='o', ls='', ms=9, label='All features (LightGBM)')],
          loc='upper center', bbox_to_anchor=(0.45, 1.14), ncol=2, frameon=False, fontsize=11.5)
save(fig, 'Fig5b')

# ======================= Fig 6: uniform design (6 learners) =======================
e = pd.read_csv('e46_summary_ALL.csv')
M = e.pivot(index='dataset', columns='learner', values='defl').loc[ORDER, LE]
S = e.pivot(index='dataset', columns='learner', values='p_t_holm').loc[ORDER, LE]
Q = e.pivot(index='dataset', columns='learner', values='p_tost1_holm').loc[ORDER, LE]
fig, ax = plt.subplots(figsize=(W, H))
im = ax.imshow(M.values, cmap=DIV, norm=TwoSlopeNorm(0, -20, 20), aspect='auto')
for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
    v = M.values[i, j]; tag = ('*' if S.values[i, j] < .05 else '') + ('=' if Q.values[i, j] < .05 else '')
    ax.text(j, i, f'{sgn(v)}{tag}', ha='center', va='center', fontsize=11, color='white' if abs(v) > 11 else INK)
ax.set_xticks(range(len(LE)), ['1-NN', 'RF', 'Light-\nGBM', 'MLP-\nES', 'MLP-\nnoES', 'Log-\nReg']); ax.set_yticks(range(len(ORDER)), [bs(d) for d in ORDER])
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label('Natural $-$ deduplicated (pp)'); cb.outline.set_visible(False)
save(fig, 'Fig6a')
fig, ax = plt.subplots(figsize=(W, H)); ax.axhline(0, color=INK2, lw=1)
ax.axhspan(-1, 1, color='#f0efec', zorder=0)
for l in LE:
    s = e[e.learner == l]
    ax.scatter(s.leak_pct, s.defl, color=LS[l][0], marker=LS[l][1], s=75, lw=0, alpha=0.9)
ax.set_xlabel('Test rows with a training twin (%)'); ax.set_ylabel('Natural $-$ deduplicated (pp)'); vgrid(ax)
ax.legend(handles=mk_handles(LE), loc='center left', bbox_to_anchor=(0.02, 0.52), frameon=False, ncol=2, fontsize=11.5)
save(fig, 'Fig6b')

# ======================= Fig 7: size-matched stratified (E4c) =======================
c = pd.read_csv('e4c_long.csv'); c['defl'] = c.f1_nat - c.f1_ded
cs_ = pd.read_csv('e4c_summary_ALL.csv').set_index(['dataset', 'learner']); L3 = ['1-NN', 'LightGBM', 'LogReg']
fig, ax = plt.subplots(figsize=(W, H)); yy = np.arange(len(ORDER))
ax.axvspan(-1, 1, color='#f0efec', zorder=0); ax.axvline(0, color=INK2, lw=1)
for k, l in enumerate(L3):
    g = c[c.learner == l].groupby('dataset').defl.agg(['mean', 'std'])
    ax.errorbar([g.loc[d, 'mean'] for d in ORDER], yy + (1 - k) * 0.24, xerr=[g.loc[d, 'std'] for d in ORDER],
                fmt=LS[l][1], color=LS[l][0], ms=9, capsize=3, elinewidth=1.5, label=LN[l], zorder=3)
ax.set_yticks(yy, [bs(d) for d in ORDER]); ax.set_xlabel('Natural $-$ deduplicated macro-F1 (pp)'); hgrid(ax)
ax.legend(loc='lower right', frameon=False); save(fig, 'Fig7a')
fig, ax = plt.subplots(figsize=(W, H)); ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
for l in L3:
    for d in ORDER:
        ax.scatter(cs_.loc[(d, l), 'dacc_tw'], cs_.loc[(d, l), 'dacc_nt'], color=LS[l][0], marker=BM[d], s=85, lw=0)
ax.set_ylim(-3, 3); ax.set_xlabel('Accuracy change, twinned test rows (pp)'); ax.set_ylabel('Accuracy change,\nnon-twinned rows (pp)')
ax.legend(handles=[Patch(color=LS[l][0], label=LN[l]) for l in L3] + bm_handles(), loc='upper left', ncol=2,
          frameon=False, fontsize=10.5, handletextpad=0.3, columnspacing=0.8)
save(fig, 'Fig7b')

# ======================= Fig 8: matched vs unmatched; twinned-row gain at 50% dose =======================
b = pd.read_csv('e4b_summary_ALL.csv'); b = b[b.design == 'strat'].set_index(['dataset', 'learner'])
fig, ax = plt.subplots(figsize=(W, H)); lim = (-6, 16); ax.plot(lim, lim, color=GRID, lw=1.5, zorder=0)
for l in L3:
    for d in ORDER:
        ax.scatter(b.loc[(d, l), 'defl'], cs_.loc[(d, l), 'defl'], color=LS[l][0], marker=BM[d], s=85, lw=0)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel('Stratified, sizes not matched (pp)'); ax.set_ylabel('Stratified, size-matched (pp)')
ax.legend(handles=[Patch(color=LS[l][0], label=LN[l]) for l in L3] + bm_handles(), loc='upper left', ncol=2,
          frameon=False, fontsize=10.5, handletextpad=0.3, columnspacing=0.8)
save(fig, 'Fig8a')
lg = pd.read_csv('e46_long.csv'); d50 = lg[lg.set == 'dose50'].copy(); d50['gap'] = d50.acc_twin - d50.acc_notwin
G = d50.groupby(['dataset', 'learner']).gap.mean().unstack().loc[ORDER, LE]
fig, ax = plt.subplots(figsize=(W, H))
im = ax.imshow(G.values, cmap=DIV, norm=TwoSlopeNorm(0, -8, 8), aspect='auto')
for i, j in itertools.product(range(G.shape[0]), range(G.shape[1])):
    v = G.values[i, j]; ax.text(j, i, f'{sgn(v)}', ha='center', va='center', fontsize=11, color='white' if abs(v) > 5 else INK)
ax.set_xticks(range(len(LE)), ['1-NN', 'RF', 'Light-\nGBM', 'MLP-\nES', 'MLP-\nnoES', 'Log-\nReg']); ax.set_yticks(range(len(ORDER)), [bs(d) for d in ORDER])
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label('Twinned $-$ non-twinned accuracy (pp)'); cb.outline.set_visible(False)
save(fig, 'Fig8b')

# ======================= Fig 9: size-matched training-size curves (E5m) =======================
# median and interquartile range over seeds: one LightGBM UNSW-NB15 run at 40k collapsed (F1 3.93), which makes the SD uninformative
e5 = pd.read_csv('e5m_long.csv'); e5['defl'] = e5.nat_matched - e5.ded
for tag, l in [('a', '1-NN'), ('b', 'LightGBM')]:
    fig, ax = plt.subplots(figsize=(W, H)); ax.axhline(0, color=INK2, lw=1)
    caps = sorted(e5[e5.learner == l].cap.unique())
    for d in ORDER:
        g = e5[(e5.learner == l) & (e5.dataset == d)].groupby('cap').defl
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.errorbar(med.index, med, yerr=[med - q1, q3 - med], color=INK2, marker=BM[d], mfc='white', mew=1.4, ms=8,
                    capsize=3, elinewidth=1.2, lw=1.4)
    ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(caps, [f'{int(k/1000) if k >= 10000 else k/1000:g}k' for k in caps])
    ax.set_xlim(caps[0] / 1.5, caps[-1] * 1.5)
    ax.legend(handles=bm_handles(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              fontsize=10.5, handletextpad=0.2, columnspacing=0.7)
    ax.set_xlabel('Training rows per class (cap)'); ax.set_ylabel('Matched natural $-$ deduplicated (pp)'); vgrid(ax)
    save(fig, f'Fig9{tag}')

# ======================= Fig 10: dose-response, 1-NN vs LogReg =======================
dsets = {'dose0': 0, 'dose50': 50, 'dose100': 100}
for tag, l in [('a', '1-NN'), ('b', 'LogReg')]:
    fig, ax = plt.subplots(figsize=(W, H)); ends = []
    for d in ORDER:
        s = lg[(lg.learner == l) & (lg.dataset == d) & lg.set.isin(dsets)].copy(); s['dose'] = s.set.map(dsets)
        g = s.groupby('dose').f1.agg(['mean', 'std'])
        ax.errorbar(g.index, g['mean'], yerr=g['std'], color=LS[l][0], marker=BM[d], mfc='white', mew=1.4, ms=8, capsize=3, lw=1.6)
        ends.append([g['mean'].iloc[-1], bs(d)])
    ax.set_xticks([0, 50, 100]); ax.set_xlim(-8, 108); ax.set_ylim(20, 103)
    ax.legend(handles=bm_handles(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              fontsize=10.5, handletextpad=0.2, columnspacing=0.7)
    ax.set_xlabel('Planted dose (% of test size)'); ax.set_ylabel(f'{LN[l]} macro-F1 (%)'); vgrid(ax)
    save(fig, f'Fig10{tag}')

# ======================= Fig 11: recovery =======================
for tag, col, yl in [('a', 'rec50', 'Recovery at 50% dose'), ('b', 'rec100', 'Recovery at 100% dose')]:
    fig, ax = plt.subplots(figsize=(W, H))
    for i, l in enumerate(LE):
        s = e[e.learner == l].set_index('dataset')
        for j, d in enumerate(ORDER):
            ax.scatter(i + (j - 3) * 0.08, s.loc[d, col], color=LS[l][0], marker=BM[d], s=60, lw=0)
        ax.hlines(s[col].mean(), i - 0.33, i + 0.33, color=INK, lw=2.2)
    ax.axhline(0, color=INK2, lw=1); ax.set_ylim(-0.12, 1.12)
    ax.set_xticks(range(len(LE)), [LN[l] for l in LE], rotation=35, ha='right'); ax.set_ylabel(yl); vgrid(ax)
    if tag == 'a': ax.legend(handles=bm_handles(), loc='upper right', ncol=2, frameon=False, fontsize=10.5, handletextpad=0.2, columnspacing=0.6)
    save(fig, f'Fig11{tag}')

# ======================= Fig 12: Nemenyi CD diagrams =======================
CD = 2.850 * np.sqrt(6 * 7 / (6 * 7))
for tag, col in [('a', 'rec50'), ('b', 'rec100')]:
    R = e.pivot(index='dataset', columns='learner', values=col)[LE].rank(axis=1, ascending=False).mean().sort_values()
    fig, ax = plt.subplots(figsize=(W, 4.4)); ax.set_xlim(-1.6, 8.6); ax.set_ylim(0.5, 10.4); ax.axis('off')
    ax.hlines(8, 1, 6, color=INK, lw=1.5)
    for k in range(1, 7): ax.vlines(k, 8, 8.35, color=INK, lw=1.5); ax.text(k, 8.6, str(k), ha='center', fontsize=13)
    ax.hlines(9.6, 1, 1 + CD, color=INK, lw=2); ax.vlines([1, 1 + CD], 9.45, 9.75, color=INK, lw=2)
    ax.text(1 + CD / 2, 9.85, f'CD = {CD:.2f}', ha='center', va='bottom', fontsize=12)
    names = list(R.index); half = 3
    for idx, l in enumerate(names):
        r = R[l]; left = idx < half; yl_ = 5.4 - (idx % half) * 1.5
        xt = 0.85 if left else 6.15
        ax.plot([r, r, xt + (0.05 if left else -0.05)], [8, yl_, yl_], color=LS[l][0], lw=1.8)
        ax.text(xt, yl_, f'{LN[l]} ({r:.2f})', ha='right' if left else 'left', va='center', fontsize=12.5, color=INK)
    cl = []
    for i in range(len(names)):
        j = max(k for k in range(i, len(names)) if R[names[k]] - R[names[i]] <= CD)
        if j > i and not any(a <= i and j <= b2 for a, b2 in cl): cl.append((i, j))
    for n, (i, j) in enumerate(cl):
        yb = 7.6 - n * 0.42; ax.hlines(yb, R[names[i]] - 0.04, R[names[j]] + 0.04, color=INK, lw=4)
    print(tag, 'ranks', R.round(3).to_dict(), 'cliques', [(names[i], names[j]) for i, j in cl])
    save(fig, f'Fig12{tag}', tight=False)

# ======================= Fig 13: MLP with vs without early stopping =======================
es, no = e[e.learner == 'MLP-ES'].set_index('dataset'), e[e.learner == 'MLP-noES'].set_index('dataset')
fig, ax = plt.subplots(figsize=(W, H)); yy = np.arange(len(ORDER))
for i, d in enumerate(ORDER):
    ax.plot([es.loc[d, 'rec50'], no.loc[d, 'rec50']], [i, i], color=GRID, lw=3, zorder=1)
    ax.errorbar(es.loc[d, 'rec50'], i + 0.12, xerr=[[es.loc[d, 'rec50'] - es.loc[d, 'rec50_lo']], [es.loc[d, 'rec50_hi'] - es.loc[d, 'rec50']]],
                fmt=LS['MLP-ES'][1], color=LS['MLP-ES'][0], ms=9, capsize=3)
    ax.errorbar(no.loc[d, 'rec50'], i - 0.12, xerr=[[no.loc[d, 'rec50'] - no.loc[d, 'rec50_lo']], [no.loc[d, 'rec50_hi'] - no.loc[d, 'rec50']]],
                fmt=LS['MLP-noES'][1], color=LS['MLP-noES'][0], ms=9, capsize=3)
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(yy, [bs(d) for d in ORDER]); ax.set_xlabel('Recovery at 50% dose (95% CI)'); hgrid(ax)
ax.legend(handles=mk_handles(['MLP-ES', 'MLP-noES']), loc='lower right', frameon=False); save(fig, 'Fig13a')
fig, ax = plt.subplots(figsize=(W, H)); w = 0.38
ax.barh(yy + w / 2, [es.loc[d, 'epochs_natural'] for d in ORDER], w, color=LS['MLP-ES'][0], label='MLP-ES')
ax.barh(yy - w / 2, [no.loc[d, 'epochs_natural'] for d in ORDER], w, color=LS['MLP-noES'][0], label='MLP-noES', hatch='//', edgecolor='white')
ax.axvline(200, color=INK2, lw=1.2, ls='--'); ax.text(198, -0.75, 'cap', ha='right', va='center', fontsize=11, color=INK2); ax.set_ylim(-1.1, len(ORDER) - 0.4)
ax.set_yticks(yy, [bs(d) for d in ORDER]); ax.set_xlim(0, 215); ax.set_xlabel('Epochs used, natural training set'); hgrid(ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False); save(fig, 'Fig13b')

# ======================= Fig 14: measurement sensitivity (P2-11) =======================
sp = pd.read_csv('sens_precision.csv'); su = pd.read_csv('sens_subsample.csv')
txt = sp[sp.hash == 'text'].set_index('dataset'); f32 = sp[sp.hash == 'float32'].set_index('dataset')
fig, ax = plt.subplots(figsize=(W, H)); yy = np.arange(len(ORDER)); w = 0.38
dd = [f32.loc[d, 'dup_pct'] - txt.loc[d, 'dup_pct'] for d in ORDER]
di = [f32.loc[d, 'incons_pct'] - txt.loc[d, 'incons_pct'] for d in ORDER]
ax.barh(yy + w / 2, dd, w, color=C7, label='Duplicate rows')
ax.barh(yy - w / 2, di, w, color=C8, label='Label-conflict rows')
for i in range(len(ORDER)):
    for v, off in ((dd[i], w / 2), (di[i], -w / 2)):
        if v > 0.05: ax.text(v + 0.2, i + off, f'+{v:.2f}', va='center', fontsize=11, color=INK)
ax.set_yticks(yy, [bs(d) for d in ORDER]); ax.set_xlim(0, 16.5); ax.set_xlabel('Increase over text hash after float32 cast (pp)'); hgrid(ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False); save(fig, 'Fig14a')
fig, ax = plt.subplots(figsize=(W, H))
for d in ORDER:
    g = su[su.dataset == d].groupby('n').leak_pct.mean()
    ax.plot(g.index, g.values, color=INK2, marker=BM[d], mfc='white', mew=1.4, ms=8, lw=1.4)
ax.set_xscale('log'); ax.set_xlabel('Rows audited (log scale; last point = full benchmark)'); ax.set_ylabel('Test rows with a training twin (%)'); vgrid(ax)
ax.legend(handles=bm_handles(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10.5, handletextpad=0.2, columnspacing=0.7)
save(fig, 'Fig14b')
