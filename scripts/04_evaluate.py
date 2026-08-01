import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
RES.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
RNG = np.random.default_rng(42)

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def safe_auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    ok = ~np.isnan(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], s[ok])

def cluster_bootstrap(y, s, clusters, stat=safe_auc, n=N_BOOT):
    y, s, clusters = np.asarray(y), np.asarray(s), np.asarray(clusters)
    point = stat(y, s)
    if np.isnan(point):
        return point, np.nan, np.nan
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    vals = []
    for _ in range(n):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        v = stat(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def bootstrap_difference(ya, sa, ca, yb, sb, cb, n=N_BOOT):
    ya, sa, ca = np.asarray(ya), np.asarray(sa), np.asarray(ca)
    yb, sb, cb = np.asarray(yb), np.asarray(sb), np.asarray(cb)
    ua, ub = np.unique(ca), np.unique(cb)
    ia = {c: np.where(ca == c)[0] for c in ua}
    ib = {c: np.where(cb == c)[0] for c in ub}
    diffs = []
    for _ in range(n):
        A = np.concatenate([ia[c] for c in RNG.choice(ua, len(ua), replace=True)])
        B = np.concatenate([ib[c] for c in RNG.choice(ub, len(ub), replace=True)])
        va, vb = safe_auc(ya[A], sa[A]), safe_auc(yb[B], sb[B])
        if not (np.isnan(va) or np.isnan(vb)):
            diffs.append(va - vb)
    if not diffs:
        return np.nan, np.nan, np.nan, np.nan
    d = np.array(diffs)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(min(p, 1.0)))

def paired_bootstrap(y, s1, s2, clusters, n=N_BOOT):
    y, s1, s2, clusters = (np.asarray(y), np.asarray(s1),
                           np.asarray(s2), np.asarray(clusters))
    ok = ~(np.isnan(s1) | np.isnan(s2))
    y, s1, s2, clusters = y[ok], s1[ok], s2[ok], clusters[ok]
    uniq = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uniq}
    diffs = []
    for _ in range(n):
        I = np.concatenate([idx[c] for c in RNG.choice(uniq, len(uniq), replace=True)])
        a, b = safe_auc(y[I], s1[I]), safe_auc(y[I], s2[I])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    if not diffs:
        return np.nan, np.nan, np.nan, np.nan
    d = np.array(diffs)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(min(p, 1.0)))

def add_gene_priors(df):
    freq = df.groupby('gene_symbol').label.mean()
    df['gene_prior_insample'] = df.gene_symbol.map(freq)

    holdout = np.full(len(df), np.nan)
    for c in df.cluster.unique():
        te = (df.cluster == c).values
        f = df.loc[~te].groupby('gene_symbol').label.mean()
        holdout[te] = df.loc[te, 'gene_symbol'].map(f).fillna(0.5).values
    df['gene_prior_holdout'] = holdout

    rand = np.full(len(df), np.nan)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=42).split(
            df, df.label):
        f = df.iloc[tr].groupby('gene_symbol').label.mean()
        rand[te] = df.iloc[te].gene_symbol.map(f).fillna(0.5).values
    df['gene_prior_randomsplit'] = rand
    return df

def main():
    frames = []
    for name in ('mito', 'control'):
        p = DATA / f'variants_{name}.parquet'
        if not p.exists():
            log(f"missing {p}; run 01_build_datasets.py first")
            return
        d = pd.read_parquet(p)
        d['set'] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    cl = pd.read_parquet(DATA / 'protein_clusters.parquet')[['uniprot_acc', 'cluster']]
    df = df.merge(cl, on='uniprot_acc', how='left')
    miss = df.cluster.isna()
    if miss.any():
        df.loc[miss, 'cluster'] = (df.cluster.max(skipna=True) or 0) + 1 + \
            pd.factorize(df.loc[miss, 'uniprot_acc'])[0]
    df['cluster'] = df.cluster.astype(int)

    amf = DATA / 'alphamissense_scores.parquet'
    if amf.exists():
        am = pd.read_parquet(amf)
        am['position_1'] = am.protein_variant.str[1:-1].astype(int)
        am['wt_aa'] = am.protein_variant.str[0]
        am['mut_aa'] = am.protein_variant.str[-1]
        df = df.merge(am[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa',
                          'alphamissense']],
                      on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                      how='left')
        log(f"AlphaMissense coverage: {df.alphamissense.notna().mean():.1%}")

    for fname, col, key in (('gmvp_scores.parquet', 'gmvp', 'gene_symbol'),
                            ('varity_scores.parquet', 'varity', 'uniprot_acc')):
        p = DATA / fname
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        if col not in t.columns:
            alt = [c for c in t.columns if c.startswith(col)]
            if not alt:
                continue
            t = t.rename(columns={alt[0]: col})
        cols = [key, 'position_1', 'wt_aa', 'mut_aa', col]
        missing = [c for c in cols if c not in t.columns]
        if missing:
            log(f"skipping {col}: {fname} lacks {missing}")
            continue
        df = df.merge(t[cols], on=[key, 'position_1', 'wt_aa', 'mut_aa'],
                      how='left')
        log(f"{col} coverage: {df[col].notna().mean():.1%} (joined on {key})")

    for name in ('mito', 'control'):
        f = RES / f'esm2_zeroshot_{name}.parquet'
        if f.exists():
            e = pd.read_parquet(f, columns=['uniprot_acc', 'position_1',
                                            'mut_aa', 'esm2_score'])
            df = df.merge(e, on=['uniprot_acc', 'position_1', 'mut_aa'],
                          how='left', suffixes=('', '_dup'))
            if 'esm2_score_dup' in df:
                df['esm2_score'] = df.esm2_score.fillna(df.esm2_score_dup)
                df = df.drop(columns=['esm2_score_dup'])

    df = add_gene_priors(df)

    predictors = [c for c in ['esm2_score', 'alphamissense', 'varity', 'gmvp',
                              'polyphen2', 'sift', 'gene_prior_insample',
                              'gene_prior_randomsplit',
                              'gene_prior_holdout'] if c in df.columns]
    log(f"predictors available: {predictors}")

    report = {'n_total': int(len(df)), 'n_clusters': int(df.cluster.nunique()),
              'predictors': predictors, 'by_set': {}, 'mito_vs_control': {},
              'vs_gene_prior': {}}

    for star_floor in (0, 2):
        tag = f'{star_floor}star'
        sub = df[df.stars >= star_floor]
        report['by_set'][tag] = {}
        log(f"\n===== review-status floor >= {star_floor} stars "
            f"(n={len(sub):,}) =====")
        for s in ('mito', 'control'):
            d = sub[sub.set == s]
            if len(d) < 50:
                continue
            row = {'n': int(len(d)), 'n_clusters': int(d.cluster.nunique()),
                   'pathogenic_fraction': float(d.label.mean())}
            for p in predictors:
                auc, lo, hi = cluster_bootstrap(d.label, d[p], d.cluster)
                row[p] = {'auroc': auc, 'ci': [lo, hi],
                          'aupr': float(average_precision_score(
                              d.label[d[p].notna()], d[p][d[p].notna()]))
                          if d[p].notna().sum() > 10 else np.nan}
                log(f"  {s:<8} {p:<22} AUROC={auc:.3f} [{lo:.3f}-{hi:.3f}]")
            report['by_set'][tag][s] = row

        m, c = sub[sub.set == 'mito'], sub[sub.set == 'control']
        if len(m) > 50 and len(c) > 50:
            report['mito_vs_control'][tag] = {}
            log(f"  --- mitochondrial minus control ---")
            for p in predictors:
                d, lo, hi, pv = bootstrap_difference(
                    m.label, m[p], m.cluster, c.label, c[p], c.cluster)
                report['mito_vs_control'][tag][p] = {
                    'delta_auroc': d, 'ci': [lo, hi], 'p': pv}
                log(f"  {p:<24} delta={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3f}")

        if 'gene_prior_insample' in predictors and len(m) > 50:
            report['vs_gene_prior'][tag] = {}
            for p in predictors:
                if p.startswith('gene_prior'):
                    continue
                d, lo, hi, pv = paired_bootstrap(
                    m.label, m[p], m.gene_prior_insample, m.cluster)
                report['vs_gene_prior'][tag][p] = {
                    'delta_auroc': d, 'ci': [lo, hi], 'p': pv}
                log(f"  {p:<24} minus gene prior: delta={d:+.3f} "
                    f"[{lo:+.3f},{hi:+.3f}] p={pv:.3f}")

    df.to_parquet(RES / 'evaluation_table.parquet', index=False)
    (RES / 'evaluation_report.json').write_text(
        json.dumps(report, indent=2, default=float))
    log(f"\nwrote {RES}")

if __name__ == '__main__':
    main()
