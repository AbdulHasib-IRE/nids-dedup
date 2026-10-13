"""All new numbers for the revised paper (M4, E8, E9, E10, hash check), computed from the result CSVs.
Writes numbers_v2.json and prints tables. Every value cited in the revised text must come from here
or from the earlier verified CSVs."""
import json, numpy as np, pandas as pd
from scipy import stats

import os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data')
P = DATA
OUT = {}

def ci_t(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return m, m - h, m + h

def paired(a, b, margin):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    m, lo, hi = ci_t(d); s = d.std(ddof=1); se = s / np.sqrt(n)
    p_t = float(stats.ttest_1samp(d, 0).pvalue) if s > 0 else (0.0 if m != 0 else 1.0)
    if se == 0: p_tost = 0.0 if abs(m) < margin else 1.0
    else:
        p1 = 1 - stats.t.cdf((m + margin) / se, n - 1); p2 = stats.t.cdf((m - margin) / se, n - 1)
        p_tost = float(max(p1, p2))
    return dict(mean=m, lo=lo, hi=hi, sd=s, p_t=p_t, p_tost=p_tost, dz=m / s if s > 0 else np.nan, n=n,
                dmin=d.min(), dmax=d.max(), npos=int((d > 0).sum()), nneg=int((d < 0).sum()))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

# ---------------------------------------------------------------- M4: margin sensitivity and effect sizes
def margin_table(pairs, name):
    """pairs: list of (dataset, learner, cap, a, b). Holm within the experiment, per margin."""
    res = {}
    rows = []
    for ds, ln, cap, a, b in pairs:
        r = dict(dataset=ds, learner=ln, cap=cap, **paired(a, b, 1.0))
        for mg in (0.5, 1.0, 2.0):
            r[f'p_tost_{mg}'] = paired(a, b, mg)['p_tost']
        rows.append(r)
    T = pd.DataFrame(rows); T['p_t_holm'] = holm(T.p_t.values)
    for mg in (0.5, 1.0, 2.0):
        T[f'p_tost_{mg}_holm'] = holm(T[f'p_tost_{mg}'].values)
    T['S'] = T.p_t_holm < .05
    out = {}
    for mg in (0.5, 1.0, 2.0):
        E = T[f'p_tost_{mg}_holm'] < .05
        out[str(mg)] = dict(S=int(T.S.sum()), E=int(E.sum()), both=int((T.S & E).sum()),
                            inconclusive=int((~T.S & ~E).sum()), n=len(T))
    res['counts'] = out
    T.to_csv(f'{P}/m4_{name}.csv', index=False)
    return T, res

L = pd.read_csv(f'{P}/e46_long.csv')
W = L.pivot_table(index=['dataset', 'learner', 'seed'], columns='set', values='f1')
pairs = [(d, l, None, W.loc[(d, l)]['natural'].values, W.loc[(d, l)]['dose0'].values)
         for d, l in sorted(set((i[0], i[1]) for i in W.index))]
T4, OUT['m4_uniform'] = margin_table(pairs, 'uniform')

C = pd.read_csv(f'{P}/e4c_long.csv')
pairs = [(d, l, None, g.sort_values('seed').f1_nat.values, g.sort_values('seed').f1_ded.values)
         for (d, l), g in C.groupby(['dataset', 'learner'])]
T4c, OUT['m4_matched'] = margin_table(pairs, 'matched')

M = pd.read_csv(f'{P}/e5m_long.csv')
pairs = [(d, l, cap, g.sort_values('seed').nat_matched.values, g.sort_values('seed').ded.values)
         for (d, l, cap), g in M.groupby(['dataset', 'learner', 'cap'])]
T5, OUT['m4_size'] = margin_table(pairs, 'size')

# effect sizes and seed-level consistency for the matched design
OUT['m4_matched_rows'] = T4c[['dataset', 'learner', 'mean', 'lo', 'hi', 'sd', 'dz', 'dmin', 'dmax', 'npos', 'nneg',
                              'p_t_holm', 'p_tost_0.5_holm', 'p_tost_1.0_holm', 'p_tost_2.0_holm']].round(4).to_dict('records')
sig = T4c[T4c.S]
OUT['m4_matched_sig_all_seeds_same_sign'] = int(((sig.npos == sig.n) | (sig.nneg == sig.n)).sum())
OUT['m4_matched_sig_n'] = int(len(sig))
OUT['m4_matched_sig_absdz_min'] = float(sig.dz.abs().min())
OUT['m4_matched_sig_absdz_max'] = float(sig.dz.abs().max())
# verdict changes between margins for the matched design
v = {}
for mg in ('0.5', '1.0', '2.0'):
    v[mg] = np.where(T4c.S, 'S', '') + np.where(T4c[f'p_tost_{mg}_holm'] < .05, 'E', '')
T4c['v05'], T4c['v1'], T4c['v2'] = v['0.5'], v['1.0'], v['2.0']
OUT['m4_matched_verdicts'] = T4c[['dataset', 'learner', 'v05', 'v1', 'v2']].to_dict('records')

# benchmark-level view: mean matched delta per learner across the 7 benchmarks with a t interval over benchmarks
bl = {}
for ln, g in T4c.groupby('learner'):
    m, lo, hi = ci_t(g['mean'].values); bl[ln] = dict(mean=m, lo=lo, hi=hi, n=len(g),
                                                      wilcoxon_p=float(stats.wilcoxon(g['mean'].values).pvalue))
OUT['m4_benchmark_level'] = bl

# ---------------------------------------------------------------- E8 attribution controls
S8 = pd.read_csv(f'{P}/e8_summary_ALL.csv'); L8 = pd.read_csv(f'{P}/e8_long.csv')
OUT['e8_n_tests'] = int(len(S8))
OUT['e8'] = S8.round(4).to_dict('records')
rep = S8[S8.contrast == 'replacement']
OUT['e8_replacement_sig_holm'] = int((rep.p_t_holm < .05).sum())
OUT['e8_replacement_n'] = int(len(rep))
OUT['e8_replacement_sig_raw'] = rep[rep.p_t < .05][['dataset', 'learner', 'mean', 'p_t']].round(4).to_dict('records')
OUT['e8_replacement_absmax'] = float(rep['mean'].abs().max())
OUT['e8_replacement_equiv'] = int((rep.p_tost1_holm < .05).sum())
tw = S8[S8.contrast == 'twin']; tot = S8[S8.contrast == 'total']
OUT['e8_twin_sig'] = int((tw.p_t_holm < .05).sum()); OUT['e8_twin_equiv'] = int((tw.p_tost1_holm < .05).sum())
OUT['e8_total_sig'] = int((tot.p_t_holm < .05).sum()); OUT['e8_total_equiv'] = int((tot.p_tost1_holm < .05).sum())
# twin-row counts and conflicting share per benchmark
cnt = L8.groupby(['dataset', 'seed'])[['n_twin_rows', 'n_twin_conf']].first().groupby('dataset').mean()
cnt['conf_share'] = cnt.n_twin_conf / cnt.n_twin_rows * 100
OUT['e8_twin_counts'] = cnt.round(3).to_dict('index')
# NAT reproduces E4c? compare 1-NN NAT f1 with e4c f1_nat on the same seeds
nat = L8[(L8.arm == 'NAT') & (L8.learner == '1-NN')].set_index(['dataset', 'seed']).f1
c4 = C[C.learner == '1-NN'].set_index(['dataset', 'seed']).f1_nat
j = nat.to_frame('a').join(c4.to_frame('b'), how='inner')
OUT['e8_nat_vs_e4c_maxdiff'] = float((j.a - j.b).abs().max()); OUT['e8_nat_vs_e4c_n'] = int(len(j))

# ---------------------------------------------------------------- E9 source holdout
S9 = pd.read_csv(f'{P}/e9_summary_ALL.csv'); L9 = pd.read_csv(f'{P}/e9_long.csv')
OUT['e9'] = S9.round(4).to_dict('records')
OUT['e9_n_tests'] = int(len(S9))
med = L9.assign(gap=L9.RAND_DED - L9.SRC_DED, src=L9.SRC_NAT - L9.SRC_DED, rnd=L9.RAND_NAT - L9.RAND_DED) \
        .groupby(['dataset', 'learner'])[['gap', 'src', 'rnd']].median()
OUT['e9_medians'] = {f'{d}|{l}': v for (d, l), v in med.round(4).to_dict('index').items()}

# ---------------------------------------------------------------- E10 low and balanced doses
S10 = pd.read_csv(f'{P}/e10_summary_ALL.csv'); L10 = pd.read_csv(f'{P}/e10_long.csv')
W10 = L10.pivot_table(index=['dataset', 'learner', 'seed'], columns='arm', values='f1')
D10 = W10[['d05', 'd10', 'd25', 'bal10']].sub(W10['d00'], axis=0)
OUT['e10'] = S10.round(4).to_dict('records')
OUT['e10_medians'] = {f'{d}|{l}': v for (d, l), v in D10.groupby(level=[0, 1]).median().round(4).to_dict('index').items()}
OUT['e10_sig_by_arm'] = S10[S10.p_t_holm < .05].groupby('arm').size().to_dict()
OUT['e10_n_tests'] = int(len(S10))
OUT['e10_n_bal10'] = L10.groupby('dataset').n_bal10.first().to_dict()
OUT['e10_n_bal10_by_seed'] = L10.groupby(['dataset', 'seed']).n_bal10.first().groupby('dataset').agg(['min', 'max']).to_dict('index')
# mean recovery per learner across 7 benchmarks for each arm, and Friedman over benchmarks
rec = S10.pivot_table(index=['arm', 'dataset'], columns='learner', values='recovery')
fr = {}
for a in ['d05', 'd10', 'd25', 'bal10']:
    R = rec.loc[a]; st = stats.friedmanchisquare(*[R[c].values for c in R.columns])
    ranks = R.rank(axis=1, ascending=False).mean()
    fr[a] = dict(mean=R.mean().round(4).to_dict(), chi2=float(st.statistic), p=float(st.pvalue), ranks=ranks.round(3).to_dict())
OUT['e10_recovery'] = fr
# dose-0 reproduction: E6 dose0 vs E10 d00 for 1-NN
E6 = L[(L.set == 'dose0')].set_index(['dataset', 'learner', 'seed']).f1
j = W10['d00'].to_frame('a').join(E6.to_frame('b'), how='inner')
OUT['e10_d00_vs_e6_maxdiff'] = float((j.a - j.b).abs().max()); OUT['e10_d00_vs_e6_n'] = int(len(j))
# per-seed outliers > 10 pp below dose0
OUT['e10_outliers'] = D10[(D10 < -10).any(axis=1)].round(2).reset_index().to_dict('records')

# ---------------------------------------------------------------- hash check
H = pd.read_csv(f'{P}/hash_check.csv')
OUT['hash'] = H.to_dict('records')
OUT['hash_total_rows'] = int(H.rows.sum()); OUT['hash_split_groups'] = int(H.text_groups_with_two_value_digests.sum())
OUT['hash_identities_total'] = int(H.text_identities.sum())

# ---------------------------------------------------------------- Bot-IoT per-class (unmatched stratified design)
PC = pd.read_csv(f'{P}/e4b_perclass_long.csv')
B = PC[(PC.dataset == 'Bot-IoT') & (PC.design == 'strat')].groupby(['learner', 'class'])[['n_test', 'f1_nat', 'f1_ded']].mean()
OUT['botiot_perclass_strat'] = {f'{l}|{c}': v for (l, c), v in B.round(3).to_dict('index').items()}

json.dump(OUT, open(f'{P}/numbers_v2.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print('M4 uniform', OUT['m4_uniform']); print('M4 matched', OUT['m4_matched']); print('M4 size', OUT['m4_size'])
print(T4c[['dataset', 'learner', 'mean', 'sd', 'dz', 'dmin', 'dmax', 'npos', 'v05', 'v1', 'v2']].round(2).to_string())
print('sig same sign all seeds', OUT['m4_matched_sig_all_seeds_same_sign'], '/', OUT['m4_matched_sig_n'], 'dz range', OUT['m4_matched_sig_absdz_min'], OUT['m4_matched_sig_absdz_max'])
print('benchmark-level', json.dumps(bl, indent=0, default=float))
print('E8 replacement sig holm', OUT['e8_replacement_sig_holm'], '/', OUT['e8_replacement_n'], 'raw', OUT['e8_replacement_sig_raw'], 'absmax', OUT['e8_replacement_absmax'], 'equiv', OUT['e8_replacement_equiv'])
print('E8 twin sig/equiv', OUT['e8_twin_sig'], OUT['e8_twin_equiv'], 'total', OUT['e8_total_sig'], OUT['e8_total_equiv'])
print('E8 counts', cnt.round(2).to_string()); print('NAT vs E4c', OUT['e8_nat_vs_e4c_maxdiff'], OUT['e8_nat_vs_e4c_n'])
print('E9 medians'); print(med.round(2).to_string())
print('E10 sig by arm', OUT['e10_sig_by_arm']); print('E10 recovery', json.dumps(fr, indent=0, default=float))
print('E10 d00 vs E6', OUT['e10_d00_vs_e6_maxdiff'], OUT['e10_d00_vs_e6_n']); print('bal10 by seed', OUT['e10_n_bal10_by_seed'])
print('hash', OUT['hash_total_rows'], OUT['hash_split_groups'], OUT['hash_identities_total'])
