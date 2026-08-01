import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
RES.mkdir(parents=True, exist_ok=True)

MIN_PER_CLASS = 3
N_BOOT = 2000
RNG = np.random.default_rng(42)

PREDICTORS = [
    ('varity', 'VARITY_R_LOO'),
    ('alphamissense', 'AlphaMissense'),
    ('gmvp', 'gMVP'),
    ('polyphen2', 'PolyPhen-2'),
    ('sift', 'SIFT'),
]

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def load():
    d = pd.read_parquet(DATA / 'variants_all.parquet')
    am = pd.read_parquet(DATA / 'alphamissense_scores.parquet')
    am['position_1'] = am.protein_variant.str[1:-1].astype(int)
    am['wt_aa'] = am.protein_variant.str[0]
    am['mut_aa'] = am.protein_variant.str[-1]
    d = d.merge(am[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa',
                    'alphamissense']],
                on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'], how='left')
    v = pd.read_parquet(DATA / 'varity_scores.parquet')
    d = d.merge(v[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa', 'varity']],
                on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'], how='left')
    g = pd.read_parquet(DATA / 'gmvp_scores.parquet')
    d = d.merge(g, on=['gene_symbol', 'position_1', 'wt_aa', 'mut_aa'],
                how='left')
    ps = DATA / 'polyphen_sift_scores.parquet'
    if ps.exists():
        t = pd.read_parquet(ps)
        d = d.merge(t, on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                    how='left')
    cols = [c for c, _ in PREDICTORS if c in d.columns and d[c].notna().any()]
    return d, cols

def within_gene_table(d, cols):
    rows = []
    for gene, grp in d.groupby('gene_symbol'):
        n_pos = int((grp.label == 1).sum())
        n_neg = int((grp.label == 0).sum())
        if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
            continue
        r = {'gene_symbol': gene, 'n': len(grp), 'n_pos': n_pos, 'n_neg': n_neg}
        for c in cols:
            r[c] = roc_auc_score(grp.label, grp[c])
        rows.append(r)
    return pd.DataFrame(rows)

def gene_bootstrap_mean(w, col, n=N_BOOT):
    v = w[col].values
    idx = np.arange(len(v))
    vals = [v[RNG.choice(idx, len(idx), replace=True)].mean() for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def paired_gene_bootstrap(w, a, b, n=N_BOOT):
    da = w[a].values - w[b].values
    idx = np.arange(len(da))
    diffs = np.array([da[RNG.choice(idx, len(idx), replace=True)].mean()
                      for _ in range(n)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return (float(da.mean()), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), float(min(p, 1.0)))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-modern', action='store_true')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    d, cols = load()
    if args.only_modern:
        cols = [c for c in cols if c in ('varity', 'alphamissense', 'gmvp')]
        log("restricted to the three modern predictors")
    d = d.dropna(subset=cols).reset_index(drop=True)
    tag = args.tag or ('_modern' if args.only_modern else '_all')
    log(f"common intersection: {len(d):,} variants, {d.gene_symbol.nunique():,} genes")

    global_auc = {c: float(roc_auc_score(d.label, d[c])) for c in cols}
    global_rank = sorted(cols, key=lambda c: -global_auc[c])
    log("\nGLOBAL AUROC (conventional benchmark):")
    for i, c in enumerate(global_rank, 1):
        log(f"  {i}. {dict(PREDICTORS)[c]:<16} {global_auc[c]:.4f}")

    w = within_gene_table(d, cols)
    log(f"\n{len(w):,} genes evaluable "
        f"(>={MIN_PER_CLASS} of each class), covering {int(w.n.sum()):,} variants")

    within_auc = {c: float(w[c].mean()) for c in cols}
    within_rank = sorted(cols, key=lambda c: -within_auc[c])
    log("\nMEAN WITHIN-GENE AUROC (gene-controlled):")
    for i, c in enumerate(within_rank, 1):
        lo, hi = gene_bootstrap_mean(w, c)
        log(f"  {i}. {dict(PREDICTORS)[c]:<16} {within_auc[c]:.4f} [{lo:.4f}-{hi:.4f}]")

    flipped = []
    log("\nPAIRWISE COMPARISONS")
    for a, b in combinations(cols, 2):
        g_diff = global_auc[a] - global_auc[b]
        mean_d, lo, hi, p = paired_gene_bootstrap(w, a, b)
        same_sign = np.sign(g_diff) == np.sign(mean_d)
        na, nb = dict(PREDICTORS)[a], dict(PREDICTORS)[b]
        log(f"  {na} vs {nb}")
        log(f"     global      {g_diff:+.4f}")
        log(f"     within-gene {mean_d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
        if not same_sign:
            verdict = 'REVERSED'
            if p >= 0.05:
                verdict = 'reversed but not significant'
            flipped.append({'a': na, 'b': nb, 'global_diff': g_diff,
                            'within_diff': mean_d, 'p': p, 'verdict': verdict})
            log(f"     -> {verdict}")
        else:
            log(f"     -> same direction")

    log("\nROBUSTNESS OF THE WITHIN-GENE RANKING")
    aggs = {
        'unweighted mean': {c: float(w[c].mean()) for c in cols},
        'variant-weighted mean': {c: float(np.average(w[c], weights=w.n))
                                  for c in cols},
        'median': {c: float(w[c].median()) for c in cols},
    }
    strict = w[(w.n_pos >= 10) & (w.n_neg >= 10)]
    if len(strict) > 50:
        aggs[f'mean, genes with >=10/class (n={len(strict)})'] = {
            c: float(strict[c].mean()) for c in cols}
    for name, vals in aggs.items():
        order = ' > '.join(dict(PREDICTORS)[c]
                           for c in sorted(cols, key=lambda c: -vals[c]))
        log(f"  {name:<40} {order}")

    mx = w[cols].max(axis=1)
    tied = (w[cols].eq(mx, axis=0)).sum(axis=1) > 1
    log(f"\n  ties for best predictor: {int(tied.sum())} of {len(w)} genes "
        f"({tied.mean():.1%}) - excluded from win counts")
    wins = w[~tied][cols].idxmax(axis=1).value_counts()
    log("  per-gene win counts (genes with a unique best predictor):")
    for c, n in wins.items():
        log(f"    {dict(PREDICTORS)[c]:<16} {n:>5} genes "
            f"({n / int((~tied).sum()):.1%})")

    log("\n  ceiling sensitivity:")
    variants = {
        'all genes': w,
        'excluding any predictor at AUROC 1.0': w[~(w[cols] == 1.0).any(axis=1)],
        'excluding genes where all are at 1.0': w[~(w[cols] == 1.0).all(axis=1)],
    }
    for name, sub in variants.items():
        if len(sub) < 50:
            continue
        m = {c: sub[c].mean() for c in cols}
        order = ' > '.join(dict(PREDICTORS)[c]
                           for c in sorted(cols, key=lambda c: -m[c]))
        log(f"    {name:<40} (n={len(sub):>5})  {order}")

    from scipy.stats import wilcoxon
    log("\n  Wilcoxon signed-rank (paired, rank-based):")
    wilcox = {}
    for a, b in combinations(cols, 2):
        diff = w[a] - w[b]
        nz = int((diff != 0).sum())
        wins_a = int((diff > 0).sum())
        pv = float(wilcoxon(w[a], w[b]).pvalue)
        wilcox[f'{dict(PREDICTORS)[a]} vs {dict(PREDICTORS)[b]}'] = {
            'median_diff': float(diff.median()), 'wins': wins_a,
            'non_ties': nz, 'win_rate': wins_a / nz if nz else float('nan'),
            'p': pv}
        log(f"    {dict(PREDICTORS)[a]:<14} vs {dict(PREDICTORS)[b]:<14} "
            f"wins {wins_a}/{nz} ({wins_a / nz:.1%})  p={pv:.2e}")

    out = w.copy()
    out.to_csv(RES / f'within_gene_ranking{tag}.csv', index=False)
    report = {
        'n_variants': int(len(d)),
        'n_genes_total': int(d.gene_symbol.nunique()),
        'n_genes_evaluable': int(len(w)),
        'min_per_class': MIN_PER_CLASS,
        'global_auroc': {dict(PREDICTORS)[c]: global_auc[c] for c in cols},
        'global_ranking': [dict(PREDICTORS)[c] for c in global_rank],
        'within_gene_auroc': {dict(PREDICTORS)[c]: within_auc[c] for c in cols},
        'within_gene_ranking': [dict(PREDICTORS)[c] for c in within_rank],
        'ranking_changed': [dict(PREDICTORS)[c] for c in global_rank]
                           != [dict(PREDICTORS)[c] for c in within_rank],
        'reversals': flipped,
        'robustness': {k: {dict(PREDICTORS)[c]: v for c, v in vals.items()}
                       for k, vals in aggs.items()},
        'per_gene_wins_excluding_ties': {dict(PREDICTORS)[c]: int(n)
                                         for c, n in wins.items()},
        'n_genes_tied_for_best': int(tied.sum()),
        'wilcoxon': wilcox,
        'ceiling_sensitivity': {
            name: {dict(PREDICTORS)[c]: float(sub[c].mean()) for c in cols}
            for name, sub in variants.items() if len(sub) >= 50},
        'note': 'Within-gene AUROC removes between-gene label structure by '
                'construction: every comparison is between two variants in the '
                'same gene, so the gene prior is constant and uninformative.',
        'caveat': 'Only genes with at least MIN_PER_CLASS of each label are '
                  'evaluable, so this analysis is restricted to well-characterised '
                  'genes and is not representative of ClinVar as a whole.',
    }
    (RES / f'within_gene_ranking{tag}.json').write_text(json.dumps(report, indent=2))
    log(f"\nranking changed: {report['ranking_changed']}")
    log(f"wrote {RES / ('within_gene_ranking'+tag+'.csv')}")

if __name__ == '__main__':
    main()
