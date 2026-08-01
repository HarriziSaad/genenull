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

N_BOOT = 500
RNG = np.random.default_rng(42)

LABELS = [
    ('varity_r_full', 'VARITY_R (trained on ClinVar)'),
    ('varity', 'VARITY_R_LOO'),
    ('gmvp', 'gMVP'),
    ('alphamissense', 'AlphaMissense'),
    ('esm2_score', 'ESM-2 zero-shot'),
]

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def gene_bootstrap_ci(y, score, genes, n=N_BOOT):
    uniq = np.unique(genes)
    idx = {g: np.where(genes == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        I = np.concatenate([idx[g] for g in RNG.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[I])) < 2:
            continue
        vals.append(roc_auc_score(y[I], score[I]))
    if not vals:
        return [np.nan, np.nan]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def main():
    d = pd.read_parquet(DATA / 'variants_all.parquet')

    am = DATA / 'alphamissense_scores.parquet'
    if am.exists():
        a = pd.read_parquet(am)
        a['position_1'] = a.protein_variant.str[1:-1].astype(int)
        a['wt_aa'] = a.protein_variant.str[0]
        a['mut_aa'] = a.protein_variant.str[-1]
        d = d.merge(a[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa',
                       'alphamissense']],
                    on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'], how='left')
    va = DATA / 'varity_scores.parquet'
    if va.exists():
        v = pd.read_parquet(va)
        keep = ['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa', 'varity']
        if 'varity_r_full' in v.columns:
            keep.append('varity_r_full')
        d = d.merge(v[keep], on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                    how='left')
    gm = DATA / 'gmvp_scores.parquet'
    if gm.exists():
        d = d.merge(pd.read_parquet(gm),
                    on=['gene_symbol', 'position_1', 'wt_aa', 'mut_aa'], how='left')
    for name in ('mito', 'control'):
        f = RES / f'esm2_zeroshot_{name}.parquet'
        if f.exists():
            e = pd.read_parquet(f)[['uniprot_acc', 'position_1', 'wt_aa',
                                    'mut_aa', 'esm2_score']]
            d = d.merge(e, on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                        how='left', suffixes=('', '_dup'))
            if 'esm2_score_dup' in d.columns:
                d['esm2_score'] = d.esm2_score.fillna(d.esm2_score_dup)
                d = d.drop(columns=['esm2_score_dup'])

    present = [c for c, _ in LABELS if c in d.columns and d[c].notna().any()]
    for c in present:
        log(f"{c}: {d[c].notna().mean():.1%} coverage")

    core = [c for c in present if c != 'esm2_score']
    s = d.dropna(subset=core).reset_index(drop=True)
    y = s.label.values
    genes = s.gene_symbol.values
    log(f"common intersection: {len(s):,} variants, {s.gene_symbol.nunique():,} genes")

    null = np.full(len(s), np.nan)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=42).split(s, y):
        f = s.iloc[tr].groupby('gene_symbol').label.mean()
        null[te] = s.iloc[te].gene_symbol.map(f).fillna(0.5).values
    null_auc = float(roc_auc_score(y, null))
    null_ci = gene_bootstrap_ci(y, null, genes)
    log(f"gene-identity null: {null_auc:.4f} {null_ci}")

    rows = []
    for c in core:
        auc = float(roc_auc_score(y, s[c]))
        rows.append({
            'predictor': dict(LABELS)[c], 'column': c,
            'n': int(len(s)), 'auroc': auc,
            'ci_lo': gene_bootstrap_ci(y, s[c].values, genes)[0],
            'ci_hi': gene_bootstrap_ci(y, s[c].values, genes)[1],
            'aupr': float(average_precision_score(y, s[c])),
            'margin_over_null': auc - null_auc,
        })
        log(f"  {dict(LABELS)[c]:<32} AUROC={auc:.4f}  "
            f"margin={auc - null_auc:+.4f}")
    rows.append({'predictor': 'Gene-identity null', 'column': 'gene_prior',
                 'n': int(len(s)), 'auroc': null_auc,
                 'ci_lo': null_ci[0], 'ci_hi': null_ci[1],
                 'aupr': float(average_precision_score(y, null)),
                 'margin_over_null': 0.0})

    out = pd.DataFrame(rows).sort_values('auroc', ascending=False)
    out.to_csv(RES / 'head_to_head.csv', index=False)

    meta = {'n_common': int(len(s)), 'n_genes': int(s.gene_symbol.nunique()),
            'gene_identity_null_auroc': null_auc}
    if 'varity_r_full' in s.columns and 'varity' in s.columns:
        meta['varity_circularity_inflation'] = float(
            roc_auc_score(y, s.varity_r_full) - roc_auc_score(y, s.varity))
        meta['circularity_note'] = (
            'Leave-one-out removes the variant itself but leaves every other '
            'variant in the same gene in training, so gene-level label '
            'structure survives it intact.')
    if 'esm2_score' in present:
        e = d.dropna(subset=['esm2_score'])
        meta['esm2'] = {
            'n': int(len(e)),
            'auroc': float(roc_auc_score(e.label, e.esm2_score)),
            'note': 'mitochondrial and matched control sets only; not part of '
                    'the common intersection',
        }
    (RES / 'head_to_head.json').write_text(json.dumps(meta, indent=2))
    log(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
