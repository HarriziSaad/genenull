import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"

NAMES = {'varity': 'VARITY_R_LOO', 'alphamissense': 'AlphaMissense',
         'gmvp': 'gMVP', 'esm2_score': 'ESM-2 zero-shot'}
COLS = list(NAMES)
SUPERVISED = {'varity', 'gmvp'}
MIN_PER_CLASS = 3
N_BOOT = 2000
RNG = np.random.default_rng(42)

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def main():
    frames = []
    for s in ('mito', 'control'):
        d = pd.read_parquet(DATA / f'variants_{s}.parquet')
        d['set'] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)

    esm = pd.concat([
        pd.read_parquet(RES / f'esm2_zeroshot_{s}.parquet')[
            ['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa', 'esm2_score']]
        for s in ('mito', 'control')
        if (RES / f'esm2_zeroshot_{s}.parquet').exists()])
    d = d.merge(esm, on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                how='inner')

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
    d = d.dropna(subset=COLS).reset_index(drop=True)
    log(f"four-predictor intersection: {len(d):,} variants, "
        f"{d.gene_symbol.nunique()} genes")

    glob = {c: float(roc_auc_score(d.label, d[c])) for c in COLS}
    log("\nglobal AUROC:")
    for c in sorted(COLS, key=lambda c: -glob[c]):
        log(f"  {NAMES[c]:<18} {glob[c]:.4f}")

    rows = []
    for gene, grp in d.groupby('gene_symbol'):
        if (grp.label == 1).sum() < MIN_PER_CLASS or \
           (grp.label == 0).sum() < MIN_PER_CLASS:
            continue
        rows.append({'gene': gene, 'n': len(grp),
                     **{c: roc_auc_score(grp.label, grp[c]) for c in COLS}})
    w = pd.DataFrame(rows)
    if len(w) < 20:
        log("too few evaluable genes; stopping")
        return
    within = {c: float(w[c].mean()) for c in COLS}
    log(f"\nwithin-gene AUROC ({len(w)} genes, {int(w.n.sum()):,} variants):")
    for c in sorted(COLS, key=lambda c: -within[c]):
        log(f"  {NAMES[c]:<18} {within[c]:.4f}")

    log("\nchange from global to within-gene (the directional test):")
    delta = {c: within[c] - glob[c] for c in COLS}
    for c in sorted(COLS, key=lambda c: -delta[c]):
        tag = 'supervised on ClinVar' if c in SUPERVISED else ''
        if c == 'esm2_score':
            tag = 'never saw clinical labels'
        log(f"  {NAMES[c]:<18} {delta[c]:+.4f}   {tag}")

    idx = np.arange(len(w))
    wins = 0
    for _ in range(N_BOOT):
        s = w.iloc[RNG.choice(idx, len(idx), replace=True)]
        de = s.esm2_score.mean() - glob['esm2_score']
        ds = max(s[c].mean() - glob[c] for c in SUPERVISED)
        if de > ds:
            wins += 1
    frac = wins / N_BOOT

    report = {
        'n_variants': int(len(d)),
        'n_genes': int(d.gene_symbol.nunique()),
        'n_genes_evaluable': int(len(w)),
        'global_auroc': {NAMES[c]: glob[c] for c in COLS},
        'within_gene_auroc': {NAMES[c]: within[c] for c in COLS},
        'change': {NAMES[c]: delta[c] for c in COLS},
        'esm2_gain_exceeds_supervised_gain_frac_bootstrap': frac,
        'directional_prediction_supported': bool(
            delta['esm2_score'] == max(delta.values())),
        'scope_limit': (
            'ESM-2 was scored only on the mitochondrial and matched control '
            f'sets, so only {len(w)} genes are evaluable here against 1,860 in '
            'the main analysis. This tests the direction of change, not the '
            'ranking reversal, which is not reproducible at this sample size '
            'and is not claimed.'),
    }
    log(f"\nESM-2 gain exceeds the best supervised gain in {frac:.1%} of "
        f"{N_BOOT} gene bootstraps")
    (RES / 'esm2_within_gene.json').write_text(json.dumps(report, indent=2))
    log(f"wrote {RES / 'esm2_within_gene.json'}")

if __name__ == '__main__':
    main()
