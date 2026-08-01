from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

__all__ = ['gene_identity_null', 'benchmark_report', 'NullResult']

UNSEEN_CONSTANT = 0.5

@dataclass
class NullResult:
    n_variants: int
    n_genes: int
    pathogenic_fraction: float
    in_sample: float
    random_split: float
    leave_gene_out: float
    frac_genes_single_class: float

    def __str__(self) -> str:
        return (
            f"gene-identity null on {self.n_variants:,} variants in "
            f"{self.n_genes:,} genes ({self.pathogenic_fraction:.1%} pathogenic)\n"
            f"  in-sample        AUROC {self.in_sample:.4f}\n"
            f"  random 10-fold   AUROC {self.random_split:.4f}   "
            f"<- compare your reported number against this\n"
            f"  leave-gene-out   AUROC {self.leave_gene_out:.4f}\n"
            f"  {self.frac_genes_single_class:.1%} of genes carry only one label class"
        )

def _prior_random_split(df, gene, label, n_splits=10, seed=42):
    y = df[label].to_numpy()
    out = np.full(len(df), np.nan)
    for tr, te in StratifiedKFold(n_splits, shuffle=True,
                                  random_state=seed).split(df, y):
        freq = df.iloc[tr].groupby(gene)[label].mean()
        out[te] = df.iloc[te][gene].map(freq).fillna(UNSEEN_CONSTANT).to_numpy()
    return out

def _prior_grouped(df, gene, label, groups, n_splits=10):
    y = df[label].to_numpy()
    out = np.full(len(df), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return np.full(len(df), UNSEEN_CONSTANT)
    for tr, te in GroupKFold(n_splits).split(df, y, groups):
        freq = df.iloc[tr].groupby(gene)[label].mean()
        out[te] = df.iloc[te][gene].map(freq).fillna(UNSEEN_CONSTANT).to_numpy()
    return out

def gene_identity_null(df: pd.DataFrame, gene: str = 'gene_symbol',
                       label: str = 'label', seed: int = 42) -> NullResult:
    for col in (gene, label):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in dataframe")
    d = df[[gene, label]].dropna().reset_index(drop=True)
    y = d[label].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("label column must contain both classes")

    freq = d.groupby(gene)[label].mean()
    in_sample = float(roc_auc_score(y, d[gene].map(freq)))
    rnd = float(roc_auc_score(y, _prior_random_split(d, gene, label, seed=seed)))
    lgo = float(roc_auc_score(
        y, _prior_grouped(d, gene, label, d[gene].to_numpy())))

    return NullResult(
        n_variants=int(len(d)), n_genes=int(d[gene].nunique()),
        pathogenic_fraction=float(y.mean()),
        in_sample=in_sample, random_split=rnd, leave_gene_out=lgo,
        frac_genes_single_class=float(freq.isin([0.0, 1.0]).mean()),
    )

def within_gene_auroc(df: pd.DataFrame, score: str,
                      gene: str = 'gene_symbol', label: str = 'label',
                      min_per_class: int = 3) -> pd.DataFrame:
    for col in (score, gene, label):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in dataframe")
    d = df.dropna(subset=[score, gene, label])
    rows = []
    for g, grp in d.groupby(gene):
        n_pos = int((grp[label] == 1).sum())
        n_neg = int((grp[label] == 0).sum())
        if n_pos < min_per_class or n_neg < min_per_class:
            continue
        rows.append({
            'gene': g, 'n': int(len(grp)), 'n_pos': n_pos, 'n_neg': n_neg,
            'auroc': float(roc_auc_score(grp[label], grp[score])),
        })
    return pd.DataFrame(rows)

def benchmark_report(df: pd.DataFrame, score: str,
                     gene: str = 'gene_symbol', label: str = 'label',
                     group: str | None = None, seed: int = 42,
                     min_per_class: int = 3) -> dict:
    d = df.dropna(subset=[score, gene, label]).reset_index(drop=True)
    y = d[label].to_numpy()
    null = gene_identity_null(d, gene=gene, label=label, seed=seed)
    auc = float(roc_auc_score(y, d[score]))

    out = {
        'n_variants': int(len(d)),
        'n_genes': int(d[gene].nunique()),
        'predictor': score,
        'predictor_auroc': auc,
        'null': asdict(null),
        'margin_over_null_random_split': auc - null.random_split,
        'interpretation': (
            f"{score} exceeds a model with no variant-level information by "
            f"{auc - null.random_split:.4f} AUROC under a random split."
        ),
    }
    wg = within_gene_auroc(d, score, gene=gene, label=label,
                           min_per_class=min_per_class)
    if len(wg):
        out['within_gene'] = {
            'n_genes_evaluable': int(len(wg)),
            'n_variants_covered': int(wg.n.sum()),
            'min_per_class': int(min_per_class),
            'mean_auroc': float(wg.auroc.mean()),
            'variant_weighted_mean_auroc': float(
                np.average(wg.auroc, weights=wg.n)),
            'median_auroc': float(wg.auroc.median()),
        }
        out['interpretation'] += (
            f" Within genes, where the null carries no information at all, it "
            f"reaches {wg.auroc.mean():.4f} (mean over {len(wg):,} evaluable "
            f"genes)."
        )

    if group and group in d.columns:
        g = d[group].to_numpy()
        n_splits = min(10, len(np.unique(g)))
        if n_splits >= 2:
            out['null_leave_cluster_out'] = float(roc_auc_score(
                y, _prior_grouped(d, gene, label, g)))
    return out

def _main():
    ap = argparse.ArgumentParser(
        description='Gene-identity null model for variant effect benchmarks.')
    ap.add_argument('table', help='parquet or csv with gene, label, and optionally score')
    ap.add_argument('--gene', default='gene_symbol')
    ap.add_argument('--label', default='label')
    ap.add_argument('--score', default=None, help='predictor column to compare')
    ap.add_argument('--group', default=None, help='homology cluster column')
    ap.add_argument('--json', action='store_true', help='emit JSON')
    a = ap.parse_args()

    df = (pd.read_parquet(a.table) if a.table.endswith(('.parquet', '.pq'))
          else pd.read_csv(a.table))

    if a.score:
        rep = benchmark_report(df, a.score, gene=a.gene, label=a.label,
                               group=a.group)
        if a.json:
            print(json.dumps(rep, indent=2))
        else:
            n = rep['null']
            print(f"{a.score}: AUROC {rep['predictor_auroc']:.4f}")
            print(f"gene-identity null (random split): {n['random_split']:.4f}")
            print(f"margin: {rep['margin_over_null_random_split']:+.4f}")
            if 'null_leave_cluster_out' in rep:
                print(f"null under leave-cluster-out: "
                      f"{rep['null_leave_cluster_out']:.4f}")
    else:
        res = gene_identity_null(df, gene=a.gene, label=a.label)
        print(json.dumps(asdict(res), indent=2) if a.json else res)

if __name__ == '__main__':
    _main()
