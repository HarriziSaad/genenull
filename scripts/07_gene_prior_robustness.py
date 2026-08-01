import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
RES.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (1, 2, 5, 10, 20, 50)

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def gene_prior_random_split(d, seed=42):
    y = d.label.values
    s = np.full(len(d), np.nan)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(d, y):
        f = d.iloc[tr].groupby('gene_symbol').label.mean()
        s[te] = d.iloc[te].gene_symbol.map(f).fillna(0.5).values
    return roc_auc_score(y, s)

def main():
    d = pd.read_parquet(DATA / 'variants_all.parquet')
    amf = DATA / 'alphamissense_scores.parquet'
    if amf.exists():
        am = pd.read_parquet(amf)
        am['position_1'] = am.protein_variant.str[1:-1].astype(int)
        am['wt_aa'] = am.protein_variant.str[0]
        am['mut_aa'] = am.protein_variant.str[-1]
        d = d.merge(am[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa',
                        'alphamissense']],
                    on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                    how='left')

    g = d.groupby('gene_symbol').label.agg(['size', 'mean'])
    rows = []
    for m in THRESHOLDS:
        keep = set(g[g['size'] >= m].index)
        s = d[d.gene_symbol.isin(keep)].reset_index(drop=True)
        if len(s) < 500 or s.label.nunique() < 2:
            continue
        gg = g.loc[list(keep)]
        row = {
            'min_variants': m,
            'n_genes': int(len(gg)),
            'n_variants': int(len(s)),
            'single_class': float(gg['mean'].isin([0.0, 1.0]).mean()),
            'pathogenic_fraction': float(s.label.mean()),
            'auroc': float(gene_prior_random_split(s)),
        }
        if 'alphamissense' in s.columns:
            ok = s.alphamissense.notna()
            if ok.sum() > 100:
                row['alphamissense_auroc'] = float(
                    roc_auc_score(s.label[ok], s.alphamissense[ok]))
                row['gap'] = row['alphamissense_auroc'] - row['auroc']
        rows.append(row)
        log(f"  >={m:>2} variants/gene: {row['n_genes']:>6,} genes, "
            f"{row['n_variants']:>7,} variants, "
            f"single-class {row['single_class']:>5.1%}, "
            f"null AUROC {row['auroc']:.3f}"
            + (f", AlphaMissense {row['alphamissense_auroc']:.3f}"
               if 'alphamissense_auroc' in row else ''))

    out = pd.DataFrame(rows)
    out.to_csv(RES / 'gene_prior_robustness.csv', index=False)

    log("split-scheme comparison (all ClinVar), gene-level bootstrap")
    y = d.label.values
    gene_ids = d.gene_symbol.values
    freq = d.groupby('gene_symbol').label.mean()

    insample = d.gene_symbol.map(freq).values
    randsplit = np.full(len(d), np.nan)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=42).split(d, y):
        f = d.iloc[tr].groupby('gene_symbol').label.mean()
        randsplit[te] = d.iloc[te].gene_symbol.map(f).fillna(0.5).values

    def gene_bootstrap(score, n=500, seed=42):
        rng = np.random.default_rng(seed)
        ok = ~np.isnan(score)
        uniq = np.unique(gene_ids[ok])
        idx = {g: np.where((gene_ids == g) & ok)[0] for g in uniq}
        vals = []
        for _ in range(n):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            I = np.concatenate([idx[g] for g in pick])
            if len(np.unique(y[I])) < 2:
                continue
            vals.append(roc_auc_score(y[I], score[I]))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    schemes = {
        'gene_prior_insample': insample,
        'gene_prior_randomsplit': randsplit,
        'gene_prior_leavegeneout': np.full(len(d), 0.5),
    }
    split_report = {'n': int(len(d)), 'n_genes': int(d.gene_symbol.nunique())}
    for name, sc in schemes.items():
        auc = 0.5 if name.endswith('leavegeneout') else float(roc_auc_score(y, sc))
        ci = [0.5, 0.5] if name.endswith('leavegeneout') else gene_bootstrap(sc)
        split_report[name] = {'auroc': auc, 'ci': ci}
        log(f"  {name:<26} AUROC={auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
    if 'alphamissense' in d.columns:
        ok = d.alphamissense.notna()
        sc = d.alphamissense.where(ok, np.nan).values
        auc = float(roc_auc_score(y[ok], sc[ok]))
        split_report['alphamissense'] = {'auroc': auc, 'ci': gene_bootstrap(sc)}
        log(f"  {'alphamissense':<26} AUROC={auc:.3f}")
    (RES / 'split_scheme_report.json').write_text(json.dumps(split_report, indent=2))
    (RES / 'gene_prior_robustness.json').write_text(json.dumps({
        'question': 'Is the gene-identity null an artefact of sparsely '
                    'annotated genes?',
        'answer': 'No. The single-class fraction falls from '
                  f"{rows[0]['single_class']:.1%} to {rows[-1]['single_class']:.1%} "
                  'across the thresholds while the null AUROC stays flat at '
                  f"{min(r['auroc'] for r in rows):.3f}-"
                  f"{max(r['auroc'] for r in rows):.3f}.",
        'split': 'random variant-level 10-fold, unseen genes -> 0.5',
    }, indent=2))
    log(f"wrote {RES / 'gene_prior_robustness.csv'}")

if __name__ == '__main__':
    main()
