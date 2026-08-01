import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
SL_FILE = SCRATCH / "uniprot_human_sl.tsv.gz"

COMPARTMENTS = [
    ('Mitochondrion',   r'\bmitochondri'),
    ('Peroxisome',      r'\bperoxisom'),
    ('Lysosome',        r'\blysosom'),
    ('Endoplasmic ret.', r'endoplasmic reticulum'),
    ('Golgi',           r'\bgolgi'),
    ('Nucleus',         r'\bnucleus\b|\bnuclear\b'),
    ('Cell membrane',   r'cell membrane|plasma membrane'),
    ('Secreted',        r'\bsecreted\b'),
    ('Cytoskeleton',    r'cytoskelet'),
    ('Cytoplasm',       r'\bcytoplasm'),
]

MIN_VARIANTS = 150
MIN_GENES = 15

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def safe_auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    ok = ~pd.isna(s)
    if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))

def assign_compartment(text):
    if not isinstance(text, str) or not text:
        return None
    t = text.lower()
    for name, pat in COMPARTMENTS:
        if re.search(pat, t):
            return name
    return None

def main():
    if not SL_FILE.exists():
        log(f"missing {SL_FILE}; download UniProt subcellular locations first")
        return

    log("loading UniProt subcellular locations")
    sl = pd.read_csv(SL_FILE, sep='\t', low_memory=False)
    acc_col = [c for c in sl.columns if c.lower().startswith('entry')][0]
    loc_col = [c for c in sl.columns if 'subcellular' in c.lower()][0]
    sl = sl[[acc_col, loc_col]].rename(
        columns={acc_col: 'uniprot_acc', loc_col: 'sl_text'})
    sl['compartment'] = sl.sl_text.map(assign_compartment)
    log(f"  {sl.compartment.notna().sum():,} of {len(sl):,} proteins assigned")

    ev = RES / 'evaluation_table.parquet'
    src = ev if ev.exists() else DATA / 'variants_all.parquet'
    log(f"loading variants from {src.name}")
    df = pd.read_parquet(src)
    df = df.merge(sl[['uniprot_acc', 'compartment']], on='uniprot_acc', how='left')

    if 'gene_prior_insample' not in df.columns:
        freq = df.groupby('gene_symbol').label.mean()
        df['gene_prior_insample'] = df.gene_symbol.map(freq)

    predictors = [c for c in ['esm2_score', 'alphamissense', 'varity', 'gmvp',
                              'polyphen2', 'sift'] if c in df.columns]
    log(f"predictors: {predictors or '(none yet - run 02/04 first)'}")

    rows = []
    for name, _ in COMPARTMENTS:
        d = df[df.compartment == name]
        if len(d) < MIN_VARIANTS or d.gene_symbol.nunique() < MIN_GENES:
            continue
        row = {
            'compartment': name,
            'n_variants': int(len(d)),
            'n_genes': int(d.gene_symbol.nunique()),
            'pathogenic_fraction': float(d.label.mean()),
            'gene_prior_auroc': safe_auc(d.label, d.gene_prior_insample),
            'frac_single_class_genes': float(
                d.groupby('gene_symbol').label.mean().isin([0.0, 1.0]).mean()),
        }
        for p in predictors:
            row[p] = safe_auc(d.label, d[p])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('n_variants', ascending=False)
    pd.set_option('display.width', 200)
    log("\n" + out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    corr = {}
    for p in predictors:
        sub = out[['gene_prior_auroc', p]].dropna()
        if len(sub) >= 4:
            corr[p] = {
                'pearson_r': float(np.corrcoef(sub.gene_prior_auroc, sub[p])[0, 1]),
                'spearman_r': float(sub.corr(method='spearman').iloc[0, 1]),
                'n_compartments': int(len(sub)),
            }
    if corr:
        log("\npredictor AUROC vs gene-prior AUROC across compartments:")
        log(json.dumps(corr, indent=2))

    out.to_csv(RES / 'compartment_generalization.csv', index=False)
    (RES / 'compartment_correlation.json').write_text(json.dumps(corr, indent=2))
    log(f"\nwrote {RES}")

if __name__ == '__main__':
    main()
