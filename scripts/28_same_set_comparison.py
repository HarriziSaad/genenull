import importlib.util
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"

MIN_PER_CLASS = 3
N_BOOT = 2000
MODERN = [('varity', 'VARITY_R_LOO'), ('alphamissense', 'AlphaMissense'),
          ('gmvp', 'gMVP')]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_script16():
    spec = importlib.util.spec_from_file_location(
        "wg16", Path(__file__).with_name("16_within_gene_ranking.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gene_null_random(d, seed=42):
    s = np.full(len(d), np.nan)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for tr, te in skf.split(d, d.label):
        freq = d.iloc[tr].groupby('gene_symbol').label.mean()
        s[te] = d.iloc[te].gene_symbol.map(freq).fillna(0.5).values
    return s


def per_gene_u(gene_codes, label, score, n_genes):
    r = pd.Series(score).groupby(gene_codes).rank(method='average').values
    pos_rank = np.bincount(gene_codes, weights=np.where(label == 1, r, 0.0),
                           minlength=n_genes)
    p = np.bincount(gene_codes, weights=label, minlength=n_genes)
    t = np.bincount(gene_codes, minlength=n_genes)
    u = pos_rank - p * (p + 1) / 2
    return u, p * (t - p)


def main():
    m16 = load_script16()
    d, cols = m16.load()
    cols = [c for c, _ in MODERN if c in cols]
    names = dict(MODERN)
    d = d.dropna(subset=cols).reset_index(drop=True)
    log(f"common intersection: {len(d):,} variants, {d.gene_symbol.nunique():,} genes")

    cnt = d.groupby('gene_symbol').label.agg(['sum', 'size'])
    ok = cnt[(cnt['sum'] >= MIN_PER_CLASS)
             & (cnt['size'] - cnt['sum'] >= MIN_PER_CLASS)].index
    e = d[d.gene_symbol.isin(ok)].reset_index(drop=True)
    log(f"evaluable genes: {len(ok):,}, variants: {len(e):,}  "
        f"(manuscript: 1,860 genes, 96,142 variants)")

    e['gene_null'] = gene_null_random(e)
    allcols = cols + ['gene_null']
    names['gene_null'] = 'Gene-identity null (random split)'

    codes, uniq = pd.factorize(e.gene_symbol)
    G = len(uniq)
    y = e.label.values.astype(np.int64)

    U, PN, rows = {}, None, []
    for c in allcols:
        s = e[c].values
        u, pn = per_gene_u(codes, y, s, G)
        U[c], PN = u, pn
        per_gene_auc = u / pn
        rows.append({'predictor': names[c],
                     'global_same_set': float(roc_auc_score(y, s)),
                     'within_mean': float(per_gene_auc.mean()),
                     'within_pooled': float(u.sum() / pn.sum())})
    t = pd.DataFrame(rows)
    t['change_mean'] = t.within_mean - t.global_same_set
    t['change_pooled'] = t.within_pooled - t.global_same_set

    t['global_full_intersection'] = [
        float(roc_auc_score(d.label, d[c])) if c in d else np.nan
        for c in allcols[:-1]] + [np.nan]

    log("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for k in ('global_same_set', 'within_mean', 'within_pooled'):
        sub = t[t.predictor != names['gene_null']].sort_values(k, ascending=False)
        log(f"  ranking by {k:<16}: {' > '.join(sub.predictor)}")

    log(f"\npaired gene bootstrap, {N_BOOT} replicates")
    rng = np.random.default_rng(42)
    gene_rows = [np.flatnonzero(codes == g) for g in range(G)]
    pairs = list(combinations(cols, 2))
    boot = {f"{a}|{b}|{k}": [] for a, b in pairs
            for k in ('global', 'mean', 'pooled')}
    for c in cols:
        boot[f"{c}|null|global"] = []
    for i in range(N_BOOT):
        gs = rng.integers(0, G, G)
        idx = np.concatenate([gene_rows[g] for g in gs])
        yy = y[idx]
        gl = {c: roc_auc_score(yy, e[c].values[idx]) for c in allcols}
        mean = {c: (U[c][gs] / PN[gs]).mean() for c in cols}
        pooled = {c: U[c][gs].sum() / PN[gs].sum() for c in cols}
        for a, b in pairs:
            boot[f"{a}|{b}|global"].append(gl[a] - gl[b])
            boot[f"{a}|{b}|mean"].append(mean[a] - mean[b])
            boot[f"{a}|{b}|pooled"].append(pooled[a] - pooled[b])
        for c in cols:
            boot[f"{c}|null|global"].append(gl[c] - gl['gene_null'])
        if (i + 1) % 500 == 0:
            log(f"  {i + 1}/{N_BOOT}")

    pd.DataFrame(boot).to_csv(RES / 'same_set_bootstrap.csv.gz', index=False)

    obs = t.set_index('predictor')
    key_obs = {'global': 'global_same_set', 'mean': 'within_mean',
               'pooled': 'within_pooled'}
    comparisons = []
    for key, vals in boot.items():
        a, b, kind = key.split('|')
        v = np.array(vals)
        if b == 'null':
            o = obs.loc[names[a], 'global_same_set'] - \
                obs.loc[names['gene_null'], 'global_same_set']
            label_ = f"{names[a]} - gene null"
        else:
            o = obs.loc[names[a], key_obs[kind]] - obs.loc[names[b], key_obs[kind]]
            label_ = f"{names[a]} - {names[b]}"
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        comparisons.append({'comparison': label_, 'metric': kind,
                            'diff': float(o),
                            'ci_low': float(np.percentile(v, 2.5)),
                            'ci_high': float(np.percentile(v, 97.5)),
                            'p': float(min(p, 1.0))})
    cmp = pd.DataFrame(comparisons)
    log("\n" + cmp.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    t.to_csv(RES / 'same_set_comparison.csv', index=False)
    out = {'n_variants_intersection': int(len(d)),
           'n_genes_evaluable': int(G), 'n_variants_evaluable': int(len(e)),
           'min_per_class': MIN_PER_CLASS, 'n_boot': N_BOOT,
           'table': t.to_dict(orient='records'),
           'comparisons': comparisons,
           'note': 'All quantities on the same evaluable-gene variants. '
                   'Pooled within-gene AUROC = sum of per-gene U / sum of '
                   'n_pos*n_neg. Bootstrap resamples whole genes.'}
    def clean(o):
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    (RES / 'same_set_comparison.json').write_text(
        json.dumps(clean(out), indent=2, allow_nan=False))
    log(f"\nwrote {RES / 'same_set_comparison.json'}")


if __name__ == '__main__':
    main()
