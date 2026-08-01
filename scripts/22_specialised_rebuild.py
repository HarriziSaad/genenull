import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
abl = __import__('09_ablation')

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def main():
    df = pd.read_parquet(DATA / 'variants_mito.parquet')
    cl = pd.read_parquet(DATA / 'protein_clusters.parquet')[['uniprot_acc', 'cluster']]
    df = df.merge(cl, on='uniprot_acc', how='left')
    df['cluster'] = df.cluster.fillna(-1).astype(int)
    df = df[df.stars >= 2].reset_index(drop=True)

    e = pd.read_parquet(RES / 'esm2_zeroshot_mito.parquet')
    keys = ['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa']
    assert 'esm2_score' in e.columns, list(e.columns)
    df = df.merge(e[keys + ['esm2_score']].rename(columns={'esm2_score': 'esm2'}),
                  on=keys, how='left')
    log(f">=2-star mitochondrial: {len(df):,} variants, "
        f"{df.cluster.nunique()} clusters, ESM-2 on {df.esm2.notna().sum():,}")

    df = df[df.esm2.notna()].reset_index(drop=True)
    u = pd.read_parquet(BASE / "data" / "raw" / "uniprot_human_reviewed.parquet")
    seqs = {a: s for a, s in zip(u.accession, u.sequence) if isinstance(s, str)}

    log("computing interpretable features from sequence")
    F = abl.featurise(df, seqs).replace([np.inf, -np.inf], 0).fillna(0)
    y = df.label.values
    groups = df.cluster.values

    feats = list(abl.ALL_FEATS)
    res = {}
    res['interpretable only'] = abl.cv_auc(F[feats].values, y, groups)
    F2 = F.copy()
    F2['esm2'] = df.esm2.values
    res['interpretable + ESM-2 zero-shot'] = abl.cv_auc(
        F2[feats + ['esm2']].values, y, groups)
    res['ESM-2 zero-shot alone (no training)'] = float(
        roc_auc_score(y, df.esm2.values))

    am = pd.read_parquet(DATA / 'alphamissense_scores.parquet')
    am['position_1'] = am.protein_variant.str[1:-1].astype(int)
    am['wt_aa'] = am.protein_variant.str[0]
    am['mut_aa'] = am.protein_variant.str[-1]
    j = df.merge(am[keys + ['alphamissense']], on=keys, how='left')
    ok = j.alphamissense.notna()
    res['AlphaMissense'] = float(roc_auc_score(j.label[ok], j.alphamissense[ok]))

    gp = df.groupby('gene_symbol').label.mean()
    res['gene-identity null, in-sample'] = float(
        roc_auc_score(y, df.gene_symbol.map(gp).values))

    log("\nSPECIALISED PIPELINE REBUILT ON CORRECTED DATA")
    log("  (leave-cluster-out, MMseqs2 30% identity)")
    for k, v in sorted(res.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0)):
        log(f"    {k:<40} {v:.4f}")

    out = {
        'n_variants': int(len(df)),
        'n_clusters': int(df.cluster.nunique()),
        'n_proteins': int(df.uniprot_acc.nunique()),
        'split': 'leave-cluster-out over MMseqs2 clusters, 30% identity, 80% coverage',
        'alphamissense_coverage': float(ok.mean()),
        'results': {k: (float(v) if v == v else None) for k, v in res.items()},
        'differences_from_original': [
            'The original used 128 principal components of ESM-2 embeddings; the '
            'language model enters here as the zero-shot masked-marginal score.',
            'Splitting is leave-cluster-out rather than random, which is correct '
            'and also harder.',
            'The original was trained on a variant set that was 79% '
            'non-mitochondrial; this uses MitoCarta 3.0 membership alone.',
        ],
        'note': 'Replaces the previously reported AUROC 0.890, which came from '
                'the contaminated pipeline and was not recomputable here.',
    }
    (RES / 'specialised_rebuild.json').write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'specialised_rebuild.json'}")

if __name__ == '__main__':
    main()
