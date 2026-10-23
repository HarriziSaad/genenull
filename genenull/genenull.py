"""
genenull - the gene-identity null model for variant effect benchmarks.

A variant effect predictor benchmarked on ClinVar is competing against a
baseline most papers never report: a model that knows nothing about the variant
and only which gene it sits in. On 197,904 ClinVar missense variants that
baseline reaches AUROC 0.921 under a random 10-fold split, within 0.035 of
AlphaMissense.

This module computes that baseline for any benchmark, so it can be reported
alongside the headline number. It has no dependencies beyond numpy, pandas and
scikit-learn, and runs in seconds on a genome-scale table.

    from genenull import gene_identity_null, benchmark_report

    # the number your benchmark should be compared against
    result = gene_identity_null(df, gene='gene_symbol', label='label')
    print(result)

    # your predictor against the null, under every split scheme
    print(benchmark_report(df, score='my_predictor', gene='gene_symbol'))

Command line:

    python genenull.py variants.parquet --gene gene_symbol --label label \\
        --score my_predictor

Splitting matters more than anything else here:

  in_sample     the gene frequency computed over the whole dataset. Leaky by
                construction; the upper bound on what gene identity can buy.
  random        recomputed on each training fold of a random variant-level
                split. This is how much of a conventionally reported benchmark
                score is attributable to gene identity.
  leave_gene_out  held-out genes are unseen, so every test variant receives the
                same constant and the null is exactly 0.5. Any performance
                above chance under this scheme is genuinely variant-level.

The null itself is not new: it is the protein majority vote of Grimm et al.
(Hum Mutat 2015), and Cheng et al. (Science 2023, supplement) computed it on
ClinVar. This module makes it a one-line report, and adds two diagnostics
(version 1.1):

  auroc_decomposition    global AUROC on a fixed variant set is a pair-weighted
                         mixture of same-gene and cross-gene comparisons. On
                         ClinVar almost all pathogenic-benign pairs are
                         cross-gene, so global AUROC is close to a cross-gene
                         statistic. Reports both components.
  gene_offset_alignment  whether a predictor's scores carry the gene-level label
                         mix: label-adjusted gene offsets correlated with each
                         gene's pathogenic fraction. ~0 for a predictor with no
                         gene-level knowledge of the labels, whatever its
                         accuracy.

Reference: Harrizi et al., bioRxiv 10.64898/2026.08.21.746180.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

__all__ = ['gene_identity_null', 'benchmark_report', 'NullResult',
           'within_gene_auroc', 'auroc_decomposition', 'gene_offset_alignment']

UNSEEN_CONSTANT = 0.5


@dataclass
class NullResult:
    """AUROC of the gene-identity null under each split scheme."""
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
    """
    Compute the gene-identity null for a benchmark table.

    df must contain a gene column and a binary label column (1 = pathogenic).
    Returns a NullResult; print it for a readable summary.
    """
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
    """
    AUROC computed inside each gene, for one predictor.

    This is the gene-controlled measure of discrimination. Every comparison is
    between two variants sharing a gene, so the gene prior is constant and
    carries no information: whatever signal remains is variant-level.

    Global AUROC mixes two abilities - separating pathogenic genes from benign
    genes, and separating pathogenic variants from benign ones within a gene.
    This isolates the second. The two can rank predictors differently, so both
    are worth reporting.

    Only genes with at least `min_per_class` of each label are evaluable, so the
    result describes well-characterised genes rather than the whole table.

    Returns one row per evaluable gene, with columns:
        gene, n, n_pos, n_neg, auroc
    """
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


def _evaluable(d, gene, label, min_per_class):
    c = d.groupby(gene)[label].agg(['sum', 'size'])
    keep = c[(c['sum'] >= min_per_class) & (c['size'] - c['sum'] >= min_per_class)].index
    return d[d[gene].isin(keep)].reset_index(drop=True)


def _per_gene_u(codes, y, s, n_genes):
    """Per-gene Mann-Whitney U (ties count one half) and n_pos * n_neg."""
    r = pd.Series(s).groupby(codes).rank(method='average').to_numpy()
    pos = np.bincount(codes, weights=np.where(y == 1, r, 0.0), minlength=n_genes)
    p = np.bincount(codes, weights=y, minlength=n_genes)
    t = np.bincount(codes, minlength=n_genes)
    return pos - p * (p + 1) / 2, p * (t - p)


def auroc_decomposition(df: pd.DataFrame, score: str, gene: str = 'gene_symbol',
                        label: str = 'label', min_per_class: int = 3,
                        evaluable_only: bool = True) -> dict:
    """
    Split global AUROC into its same-gene and cross-gene parts.

    On a fixed variant set, global AUROC averages over every pathogenic-benign
    pair. Some pairs share a gene and some do not, so

        global = w * within_pooled + (1 - w) * cross_gene

    where w is the share of pairs that fall within one gene. Gene-level label
    structure can only act through the cross-gene pairs. On ClinVar w is tiny
    (about 0.001 even among genes carrying both labels), so global AUROC is in
    practice a cross-gene statistic.

    With evaluable_only=True (default) everything is computed on the variants
    of genes carrying >= min_per_class of each label, so global and within-gene
    AUROC are measured on identical variants. Returns a dict.
    """
    for col in (score, gene, label):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in dataframe")
    d = df.dropna(subset=[score, gene, label]).reset_index(drop=True)
    if evaluable_only:
        d = _evaluable(d, gene, label, min_per_class)
    if d.empty or d[label].nunique() < 2:
        raise ValueError("no evaluable genes; lower min_per_class")
    y = d[label].to_numpy().astype(int)
    s = d[score].to_numpy(dtype=float)
    codes, uniq = pd.factorize(d[gene])
    u, pn = _per_gene_u(codes, y, s, len(uniq))
    n_pos = int(y.sum())
    all_pairs = float(n_pos) * float(len(y) - n_pos)
    glob = float(roc_auc_score(y, s))
    has = pn > 0
    within_pooled = float(u.sum() / pn.sum()) if pn.sum() else float('nan')
    cross = (glob * all_pairs - u.sum()) / (all_pairs - pn.sum())
    return {
        'n_variants': int(len(d)), 'n_genes': int(len(uniq)),
        'evaluable_only': bool(evaluable_only), 'min_per_class': int(min_per_class),
        'global_auroc': glob,
        'within_gene_pooled_auroc': within_pooled,
        'within_gene_mean_auroc': float((u[has] / pn[has]).mean()),
        'cross_gene_auroc': float(cross),
        'same_gene_pair_share': float(pn.sum() / all_pairs),
        'change_within_minus_global': within_pooled - glob,
    }


def _spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def gene_offset_alignment(df: pd.DataFrame, score: str, gene: str = 'gene_symbol',
                          label: str = 'label', min_per_class: int = 3,
                          n_boot: int = 500, seed: int = 0,
                          return_table: bool = False):
    """
    Does a predictor's score carry the gene-level label mix of the benchmark?

    Each variant's score is converted to a percentile over the analysed set and
    its class mean is subtracted (so a gene's own label mix cannot move its
    offset mechanically). The mean residual per gene is that gene's offset.
    The alignment is the Spearman correlation, across genes, between offset
    and the gene's pathogenic fraction.

    A predictor that scores variants without gene-level knowledge of the labels
    has alignment near 0 regardless of how accurate it is. A predictor that
    has absorbed the label mix - for example by training on the same archive -
    scores benign variants higher in pathogenic-rich genes and pathogenic
    variants lower in benign-rich ones, and has positive alignment.

    Uses genes with >= min_per_class variants of each label. Interval is a
    bootstrap over genes. Returns a dict, and the per-gene table if
    return_table=True.
    """
    for col in (score, gene, label):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in dataframe")
    d = _evaluable(df.dropna(subset=[score, gene, label]).reset_index(drop=True),
                   gene, label, min_per_class)
    if d[gene].nunique() < 20:
        raise ValueError("fewer than 20 evaluable genes")
    y = d[label].to_numpy().astype(int)
    pct = pd.Series(d[score].to_numpy(dtype=float)).rank().to_numpy() / len(d)
    resid = pct - np.where(y == 1, pct[y == 1].mean(), pct[y == 0].mean())
    t = (pd.DataFrame({'gene': d[gene].to_numpy(), 'y': y, 'r': resid})
         .groupby('gene').agg(n=('y', 'size'), n_pos=('y', 'sum'), offset=('r', 'mean')))
    t['pathogenic_fraction'] = t.n_pos / t.n
    rho = _spearman(t.offset, t.pathogenic_fraction)
    rng = np.random.default_rng(seed)
    off, pf = t.offset.to_numpy(), t.pathogenic_fraction.to_numpy()
    boots = [_spearman(off[i], pf[i])
             for i in (rng.integers(0, len(t), len(t)) for _ in range(n_boot))]
    out = {'alignment': rho,
           'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           'n_genes': int(len(t)), 'n_variants': int(len(d)),
           'min_per_class': int(min_per_class)}
    return (out, t.reset_index()) if return_table else out


def benchmark_report(df: pd.DataFrame, score: str,
                     gene: str = 'gene_symbol', label: str = 'label',
                     group: str | None = None, seed: int = 42,
                     min_per_class: int = 3) -> dict:
    """
    Report a predictor against the gene-identity null.

    `score`: column of predictor scores, higher meaning more pathogenic.
    `group`: optional homology-cluster column. If given, the predictor is also
             evaluated under leave-cluster-out, which is the split under which
             the null carries no information at all.
    """
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
        dec = auroc_decomposition(d, score, gene=gene, label=label,
                                  min_per_class=min_per_class)
        out['decomposition_evaluable_genes'] = dec
        out['interpretation'] += (
            f" On the {dec['n_variants']:,} variants of the {dec['n_genes']:,} "
            f"genes carrying both labels, global AUROC is "
            f"{dec['global_auroc']:.4f}, within-gene (pooled) "
            f"{dec['within_gene_pooled_auroc']:.4f}; only "
            f"{dec['same_gene_pair_share']:.2%} of pathogenic-benign pairs "
            f"there share a gene."
        )
        if wg.gene.nunique() >= 20:
            al = gene_offset_alignment(d, score, gene=gene, label=label,
                                       min_per_class=min_per_class, seed=seed)
            out['gene_offset_alignment'] = al
            out['interpretation'] += (
                f" Gene-offset alignment {al['alignment']:+.3f} "
                f"[{al['ci95'][0]:+.3f}, {al['ci95'][1]:+.3f}]; values near 0 "
                f"mean the scores carry no gene-level label information."
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
            if 'decomposition_evaluable_genes' in rep:
                dc = rep['decomposition_evaluable_genes']
                print(f"genes with both labels ({dc['n_genes']:,}): global "
                      f"{dc['global_auroc']:.4f}, within-gene pooled "
                      f"{dc['within_gene_pooled_auroc']:.4f}, cross-gene "
                      f"{dc['cross_gene_auroc']:.4f}, same-gene pair share "
                      f"{dc['same_gene_pair_share']:.4%}")
            if 'gene_offset_alignment' in rep:
                al = rep['gene_offset_alignment']
                print(f"gene-offset alignment: {al['alignment']:+.3f} "
                      f"[{al['ci95'][0]:+.3f}, {al['ci95'][1]:+.3f}]")
            if 'null_leave_cluster_out' in rep:
                print(f"null under leave-cluster-out: "
                      f"{rep['null_leave_cluster_out']:.4f}")
    else:
        res = gene_identity_null(df, gene=a.gene, label=a.label)
        print(json.dumps(asdict(res), indent=2) if a.json else res)


if __name__ == '__main__':
    _main()
