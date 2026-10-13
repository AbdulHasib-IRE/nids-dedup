"""Revised Fig 1 and the new Figs 15-18 (controls and robustness), drawn only from the result CSVs.
Uses the style, colours, markers and overlap checker of make_figs2.py (lines 1-64 executed verbatim)."""
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_figs2.py')).read().split('\n')
exec('\n'.join(src[:next(i for i, l in enumerate(src) if l.startswith('# ======================= Fig 1')) ]))
DPI = int(os.environ.get('FDPI', DPI)); plt.rcParams['savefig.dpi'] = DPI
from scipy import stats

from matplotlib.legend import Legend
def save2(fig, name):
    fig.tight_layout(pad=0.4); fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for ax in fig.axes:
        legs = [a for a in ax.artists if isinstance(a, Legend)] + ([ax.get_legend()] if ax.get_legend() else [])
        boxes = [l.get_window_extent(r) for l in legs]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i].overlaps(boxes[j]): bad.append('legend-legend')
        for b in boxes:
            for c in ax.collections:
                pts = np.asarray(c.get_offsets(), float)
                if any(b.contains(*ax.transData.transform(q)) for q in pts if np.all(np.isfinite(q))): bad.append('legend-data'); break
        for t in ax.texts:
            for b in boxes:
                if t.get_window_extent(r).overlaps(b): bad.append('legend-text')
    print(f'{name:6s} two-legend check: {bad if bad else "none"}')
    fig.savefig(f'{OUT}/{name}.png', dpi=DPI); plt.close(fig)

# ======================= Fig 1: study design (revised, no crossing arrows, text fit checked) =======================
fig, ax = plt.subplots(figsize=(W, 9.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
BOXES = []
def box(x, y, w, h, title, body='', fc='#f0efec', fs=10.5, center=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fc, ec=INK2, lw=1.2); ax.add_patch(p)
    ts = []
    if body:
        ts.append(ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=fs + 0.5, color=INK, fontweight='bold'))
        ts.append(ax.text(x + w / 2, (y + (h - 4) / 2) if center else (y + h - 5.4), body, ha='center', va='center' if center else 'top', fontsize=fs - 0.3, color=INK, linespacing=1.4))
    else:
        ts.append(ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.4))
    BOXES.append((p, ts))
def arrow(x0, y0, x1, y1): ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.4, mutation_scale=14, shrinkA=0, shrinkB=0))
LX, LW, RX, RW = 2, 38, 52, 46
box(2, 91.5, 96, 7, '7 public NIDS benchmarks, full releases: 121,704,461 records', fs=11)
box(2, 79.5, 96, 9, 'Identifier removal; text hash per record, second hash with ports;\na second digest verifies every duplicate group', fs=10.5)
arrow(50, 91.5, 50, 88.5)
box(LX, 15, LW, 59.5, 'RQ1  Audit', 'duplicate rate,\nlabel-conflict rate,\naccuracy ceiling,\norigin of duplicate groups;\n\nsensitivity to ports,\nfloat32 casting,\nsubsampling and\ncolumn count', fc='#e3e0f4', center=True)
box(RX, 67, RW, 7.5, 'Fixed test set per seed\n(70/30 pool split)', fs=10, fc='#dceaf9')
arrow(LX + LW / 2, 79.5, LX + LW / 2, 74.5); arrow(RX + RW / 2, 79.5, RX + RW / 2, 74.5)
box(RX, 50, RW, 13.5, 'RQ2, RQ3  Deduplication', 'uniform and size-matched;\ntwinned vs non-twinned rows;\ntraining-size curve', fc='#dceaf9', fs=10)
box(RX, 32, RW, 14.5, 'RQ4  Planted test copies', '5 to 100% doses and\nclass-balanced 10%; MLP with\nand without early stopping', fc='#fbe1dd', fs=10)
box(RX, 15, RW, 13.5, 'RQ5  Controls', 'placebo removal; consistent vs\nconflicting twins; equivalence\nmargins; held-out sources', fc='#dceaf9', fs=10)
SP = RX - 4                                      # spine from the test set to the three training experiments
ax.plot([RX, SP, SP], [70.75, 70.75, 21.75], color=INK2, lw=1.4, solid_capstyle='butt')
for yc in (56.75, 39.25, 21.75): arrow(SP, yc, RX, yc)
box(2, 0.5, 96, 11, '6 learners: 1-NN, RF, LightGBM, MLP-ES, MLP-noES, LogReg\n10 seeds (5 for the controls); paired t and TOST (\u00b10.5, 1, 2 pp),\nHolm correction, effect sizes, Friedman/Nemenyi', fs=9.6)
arrow(RX + RW / 2, 15, RX + RW / 2, 11.5)                       # training experiments -> shared learners and statistics
fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
for p, ts in BOXES:
    bb = p.get_window_extent(r)
    for t in ts:
        tb = t.get_window_extent(r)
        if tb.x0 < bb.x0 + 3 or tb.x1 > bb.x1 - 3 or tb.y0 < bb.y0 + 2 or tb.y1 > bb.y1 - 2: bad.append(t.get_text()[:30])
    for q, _ in BOXES:
        if q is not p and bb.overlaps(q.get_window_extent(r)): bad.append('box overlap')
print('Fig1 text-fit problems:', bad if bad else 'none')
fig.savefig(f'{OUT}/Fig1.png', dpi=DPI); plt.close(fig)

# ======================= Fig 15: attribution controls (E8) =======================
S8 = pd.read_csv('e8_summary_ALL.csv')
piv = S8.pivot_table(index=['dataset', 'learner'], columns='contrast', values='mean')
fig, ax = plt.subplots(figsize=(W, H))
for (d, l), r in piv.iterrows():
    ax.scatter(r['replacement'], r['twin'], color=LS[l][0], marker=BM[d], s=70, lw=0, zorder=3)
ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
ax.axvspan(-1, 1, color=GRID, alpha=0.5, lw=0, zorder=0)
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-8, 16.5)
ax.set_xlabel('Replacement effect, natural minus placebo (pp)'); ax.set_ylabel('Twin effect, placebo minus deduplicated (pp)')
vgrid(ax)
leg1 = ax.legend(handles=mk_handles(LE), loc='upper left', frameon=False, fontsize=10.5, ncol=2, handletextpad=0.2, columnspacing=0.6)
ax.add_artist(leg1)
ax.legend(handles=bm_handles(), loc='upper right', frameon=False, fontsize=10, ncol=1, handletextpad=0.2)
save2(fig, 'Fig15a')

fig, ax = plt.subplots(figsize=(W, H))
for (d, l), r in piv.iterrows():
    if pd.isna(r.get('consistent')) or pd.isna(r.get('conflicting')): continue
    ax.scatter(r['consistent'], r['conflicting'], color=LS[l][0], marker=BM[d], s=70, lw=0, zorder=3)
ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
ax.set_xlim(-8, 13); ax.set_ylim(-8, 13)
ax.set_xlabel('Only label-consistent twins removed (pp)'); ax.set_ylabel('Only label-conflicting twins removed (pp)')
vgrid(ax)
leg1 = ax.legend(handles=mk_handles(LE), loc='upper right', frameon=False, fontsize=10.5, ncol=2, handletextpad=0.2, columnspacing=0.6)
ax.add_artist(leg1)
ax.legend(handles=[h for h in bm_handles() if h.get_label() != 'N-BaIoT'], loc='upper left', frameon=False, fontsize=10, ncol=1, handletextpad=0.2)
save2(fig, 'Fig15b')

# ======================= Fig 16: per-seed spread (matched design) and verdicts by margin =======================
C = pd.read_csv('e4c_long.csv'); C['delta'] = C.f1_nat - C.f1_ded
fig, ax = plt.subplots(figsize=(W, H)); yy = np.arange(len(ORDER)); off = {'1-NN': 0.25, 'LightGBM': 0.0, 'LogReg': -0.25}
for i, d in enumerate(ORDER):
    for l in ['1-NN', 'LightGBM', 'LogReg']:
        v = C[(C.dataset == d) & (C.learner == l)].delta.values
        ax.scatter(v, np.full(len(v), i + off[l]), color=LS[l][0], marker=LS[l][1], s=26, lw=0, alpha=0.85, zorder=3)
        ax.vlines(np.median(v), i + off[l] - 0.11, i + off[l] + 0.11, color=INK, lw=1.6, zorder=4)
ax.axvspan(-1, 1, color=GRID, alpha=0.6, lw=0, zorder=0); ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(yy, [bs(d) for d in ORDER]); ax.set_xlabel('$\\Delta$ per seed, size-matched design (pp)'); hgrid(ax)
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.legend(handles=mk_handles(['1-NN', 'LightGBM', 'LogReg']), loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False)
save(fig, 'Fig16a')

cnt = {}
for name in ['uniform', 'matched', 'size']:
    T = pd.read_csv(f'm4_{name}.csv')
    for mg in ['0.5', '1.0', '2.0']:
        E = T[f'p_tost_{mg}_holm'] < .05; S = T.p_t_holm < .05
        cnt[(name, mg)] = [int((S & ~E).sum()), int((S & E).sum()), int((~S & E).sum()), int((~S & ~E).sum())]
lab = {'uniform': 'Uniform', 'matched': 'Size-matched', 'size': 'Training size'}
fig, ax = plt.subplots(figsize=(W, H)); cols = [C8, '#7a5fc0', C7, '#bdbbb6']; names = ['Significant only', 'Significant and equivalent', 'Equivalent only', 'Inconclusive']
xs, xt = [], []
for gi, name in enumerate(['uniform', 'matched', 'size']):
    for mi, mg in enumerate(['0.5', '1.0', '2.0']):
        x = gi * 4 + mi; xs.append(x); xt.append(f'$\\pm${float(mg):g}')
        n = sum(cnt[(name, mg)]); bot = 0
        for k, v in enumerate(cnt[(name, mg)]):
            ax.bar(x, v / n * 100, 0.8, bottom=bot, color=cols[k], label=names[k] if (gi == 0 and mi == 0) else None); bot += v / n * 100
    ax.text(gi * 4 + 1, -9, lab[name], ha='center', va='top', fontsize=13)
ax.set_xticks(xs, xt); ax.set_ylim(0, 100); ax.set_ylabel('Comparisons (%)'); vgrid(ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11)
save2(fig, 'Fig16b')

# ======================= Fig 17: dose-response at small doses and class-balanced planting =======================
L6 = pd.read_csv('e46_long.csv'); L10 = pd.read_csv('e10_long.csv'); S5 = [42, 123, 456, 789, 1011]
L6 = L6[L6.seed.isin(S5)]
fig, ax = plt.subplots(figsize=(W, H))
for d in ORDER:
    base = L6[(L6.learner == '1-NN') & (L6.dataset == d) & (L6.set == 'dose0')].set_index('seed').f1
    g = {0: 0.0}
    for arm, x in [('d05', 5), ('d10', 10), ('d25', 25)]:
        v = L10[(L10.learner == '1-NN') & (L10.dataset == d) & (L10.arm == arm)].set_index('seed').f1
        g[x] = float((v - base).mean())
    for st, x in [('dose50', 50), ('dose100', 100)]:
        v = L6[(L6.learner == '1-NN') & (L6.dataset == d) & (L6.set == st)].set_index('seed').f1
        g[x] = float((v - base).mean())
    xs_ = sorted(g); ax.plot(xs_, [g[x] for x in xs_], color=INK2, marker=BM[d], mfc='white', mew=1.4, ms=8, lw=1.4)
ax.set_xscale('symlog', linthresh=5); ax.set_xticks([0, 5, 10, 25, 50, 100], ['0', '5', '10', '25', '50', '100'])
ax.set_xlabel('Planted dose (% of test size)'); ax.set_ylabel('1-NN macro-F1 gain over dose 0 (pp)'); vgrid(ax)
ax.legend(handles=bm_handles(), loc='upper left', ncol=2, frameon=False, fontsize=10.5, handletextpad=0.2, columnspacing=0.6)
save(fig, 'Fig17a')

S10 = pd.read_csv('e10_summary_ALL.csv'); pv = S10.pivot_table(index=['dataset', 'learner'], columns='arm', values='gain')
fig, ax = plt.subplots(figsize=(W, H))
for (d, l), r in pv.iterrows():
    ax.scatter(r['d10'], r['bal10'], color=LS[l][0], marker=BM[d], s=70, lw=0, zorder=3)
ax.plot([-10, 45], [-10, 45], color=INK2, lw=1, ls='--'); ax.set_xlim(-8, 45); ax.set_ylim(-8, 45)
ax.set_xlabel('Gain, random 10% dose (pp)'); ax.set_ylabel('Gain, class-balanced 10% dose (pp)'); vgrid(ax)
leg1 = ax.legend(handles=mk_handles(['1-NN', 'RandomForest', 'LightGBM', 'MLP-ES', 'LogReg']), loc='upper left', frameon=False, fontsize=10.5, ncol=1, handletextpad=0.2)
ax.add_artist(leg1)
ax.legend(handles=bm_handles(), loc='lower right', frameon=False, fontsize=10, ncol=2, handletextpad=0.2, columnspacing=0.6)
save2(fig, 'Fig17b')

# ======================= Fig 18: held-out capture sources (E9) =======================
S9 = pd.read_csv('e9_summary_ALL.csv'); O9 = [d for d in ORDER if d in set(S9.dataset)]
p9 = S9.pivot_table(index=['dataset', 'learner'], columns='contrast', values='mean')
fig, ax = plt.subplots(figsize=(W, H))
for (d, l), r in p9.iterrows():
    ax.scatter(r['rand_nat-ded'], r['src_nat-ded'], color=LS[l][0], marker=BM[d], s=80, lw=0, zorder=3)
ax.plot([-9, 5], [-9, 5], color=INK2, lw=1, ls='--'); ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlim(-9, 5); ax.set_ylim(-9, 5)
ax.annotate('one collapsed seed', xy=(p9.loc[('UNSW-NB15', 'LightGBM'), 'rand_nat-ded'], p9.loc[('UNSW-NB15', 'LightGBM'), 'src_nat-ded']), xytext=(-3.2, -6.4), fontsize=11, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=1))
ax.set_xlabel('$\\Delta$, random row split (pp)'); ax.set_ylabel('$\\Delta$, held-out sources (pp)'); vgrid(ax)
leg1 = ax.legend(handles=mk_handles(['1-NN', 'LightGBM', 'LogReg']), loc='upper left', frameon=False, fontsize=10.5)
ax.add_artist(leg1)
ax.legend(handles=[h for h in bm_handles() if h.get_label() in [bs(d) for d in O9]], loc='lower right', frameon=False, fontsize=10, ncol=1, handletextpad=0.2)
save2(fig, 'Fig18a')

gap = S9[S9.contrast == 'rand_ded-src_ded'].set_index(['dataset', 'learner'])
fig, ax = plt.subplots(figsize=(W, H)); yy = np.arange(len(O9)); off = {'1-NN': 0.25, 'LightGBM': 0.0, 'LogReg': -0.25}
for i, d in enumerate(O9):
    for l in ['1-NN', 'LightGBM', 'LogReg']:
        r = gap.loc[(d, l)]
        ax.errorbar(r['mean'], i + off[l], xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]], fmt=LS[l][1], color=LS[l][0], ms=8, capsize=3)
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(yy, [bs(d) for d in O9]); ax.set_ylim(-0.6, len(O9) - 0.4)
ax.set_xlabel('Random minus held-out-source macro-F1,\ntwin-free training (pp, 95% CI)'); hgrid(ax)
ax.legend(handles=mk_handles(['1-NN', 'LightGBM', 'LogReg']), loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False)
save(fig, 'Fig18b')
