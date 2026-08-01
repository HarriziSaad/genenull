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
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
CLINICAL = SCRATCH / "pgym_clinical.parquet"

NAMES = {'varity': 'VARITY_R_LOO', 'alphamissense': 'AlphaMissense',
         'gmvp': 'gMVP'}
COLS = list(NAMES)
MIN_PER_CLASS = 3
N_BOOT = 2000
RNG = np.random.default_rng(42)

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def map_proteins():
    d = pd.read_parquet(CLINICAL)
    log(f"ProteinGym clinical: {len(d):,} variants, "
        f"{d.protein_id.nunique():,} proteins")

    u = pd.read_parquet(UNIPROT)
    seq2acc = {}
    for acc, s in zip(u.accession, u.sequence):
        if isinstance(s, str):
            seq2acc.setdefault(s, acc)
    log(f"UniProt index: {len(seq2acc):,} distinct sequences")

    ref = d[['protein_id', 'target_seq']].drop_duplicates('protein_id')
    ref['uniprot_acc'] = ref.target_seq.map(seq2acc)
    matched = ref.uniprot_acc.notna()
    log(f"exact sequence match: {int(matched.sum()):,}/{len(ref):,} proteins "
        f"({matched.mean():.1%})")

    d = d.merge(ref[['protein_id', 'uniprot_acc']], on='protein_id', how='left')
    d = d[d.uniprot_acc.notna()].copy()

    d['wt_aa'] = d.mutant.str[0]
    d['mut_aa'] = d.mutant.str[-1]
    d['position_1'] = d.mutant.str[1:-1].astype(int)
    d['label'] = (d.annotation == 'Pathogenic').astype(int)

    acc2seq = {a: s for a, s in zip(u.accession, u.sequence) if isinstance(s, str)}
    ok = [0 <= p - 1 < len(acc2seq.get(a, '')) and acc2seq[a][p - 1] == w
          for a, p, w in zip(d.uniprot_acc, d.position_1, d.wt_aa)]
    log(f"wild-type residue validated: {int(np.sum(ok)):,}/{len(d):,} "
        f"({np.mean(ok):.1%})")
    d = d[ok]

    acc2gene = {}
    for acc, gn in zip(u.accession, u.gene_names):
        if isinstance(gn, str) and gn.split():
            acc2gene[acc] = gn.split()[0].upper()
    d['gene_symbol'] = d.uniprot_acc.map(acc2gene)

    keep = ['uniprot_acc', 'gene_symbol', 'protein_id', 'position_1', 'wt_aa',
            'mut_aa', 'label']
    d = d[keep].drop_duplicates(
        subset=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'])
    log(f"final: {len(d):,} variants, {d.uniprot_acc.nunique():,} proteins, "
        f"{d.label.mean():.1%} pathogenic")
    d.to_parquet(DATA / 'pgym_clinical_mapped.parquet', index=False)
    return d

def attach_scores(d):
    for fname, col, key in (
            ('replication_scores_alphamissense.parquet', 'alphamissense', 'uniprot_acc'),
            ('replication_scores_varity.parquet', 'varity', 'uniprot_acc'),
            ('replication_scores_gmvp.parquet', 'gmvp', 'gene_symbol')):
        p = DATA / fname
        if not p.exists():
            log(f"missing {fname}; run 20_replication_scores.py")
            continue
        t = pd.read_parquet(p)
        on = [key, 'position_1', 'wt_aa', 'mut_aa']
        if key not in d.columns:
            continue
        d = d.merge(t[on + [col]], on=on, how='left')
        log(f"{col}: {d[col].notna().mean():.1%} coverage")
    return d

def within_gene(d, cols, group='uniprot_acc', min_per_class=MIN_PER_CLASS):
    rows = []
    for g, grp in d.groupby(group):
        if (grp.label == 1).sum() < min_per_class or \
           (grp.label == 0).sum() < min_per_class:
            continue
        rows.append({group: g, 'n': len(grp),
                     **{c: roc_auc_score(grp.label, grp[c]) for c in cols}})
    return pd.DataFrame(rows)

def main():
    d = map_proteins()
    d = attach_scores(d)
    have = [c for c in COLS if c in d.columns and d[c].notna().any()]
    if len(have) < 2:
        log("fewer than two predictors scored; stopping")
        return
    d = d.dropna(subset=have).reset_index(drop=True)
    log(f"\ncommon intersection: {len(d):,} variants, "
        f"{d.uniprot_acc.nunique():,} proteins")

    glob = {c: float(roc_auc_score(d.label, d[c])) for c in have}
    grank = sorted(have, key=lambda c: -glob[c])
    log("\nGLOBAL AUROC:")
    for i, c in enumerate(grank, 1):
        log(f"  {i}. {NAMES[c]:<16} {glob[c]:.4f}")

    w = within_gene(d, have)
    log(f"\n{len(w):,} proteins evaluable")
    win = {c: float(w[c].mean()) for c in have}
    wrank = sorted(have, key=lambda c: -win[c])
    log("MEAN WITHIN-PROTEIN AUROC:")
    for i, c in enumerate(wrank, 1):
        log(f"  {i}. {NAMES[c]:<16} {win[c]:.4f}")

    pairs = {}
    log("\nPAIRWISE:")
    for a, b in combinations(have, 2):
        gd = glob[a] - glob[b]
        diff = (w[a] - w[b]).values
        boots = np.array([diff[RNG.choice(len(diff), len(diff), replace=True)].mean()
                          for _ in range(N_BOOT)])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        rev = bool(np.sign(gd) != np.sign(diff.mean()))
        pairs[f'{NAMES[a]} vs {NAMES[b]}'] = {
            'global_diff': gd, 'within_diff': float(diff.mean()),
            'ci': [float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5))],
            'p': float(min(p, 1.0)), 'reversed': rev}
        log(f"  {NAMES[a]} vs {NAMES[b]}: global {gd:+.4f}, "
            f"within {diff.mean():+.4f}, p={min(p,1.0):.4f}"
            f"{'   REVERSED' if rev else ''}")

    report = {
        'source': 'ProteinGym clinical_substitutions.parquet (independent curation)',
        'n_variants': int(len(d)), 'n_proteins': int(d.uniprot_acc.nunique()),
        'n_proteins_evaluable': int(len(w)),
        'pathogenic_fraction': float(d.label.mean()),
        'global_auroc': {NAMES[c]: glob[c] for c in have},
        'global_ranking': [NAMES[c] for c in grank],
        'within_protein_auroc': {NAMES[c]: win[c] for c in have},
        'within_protein_ranking': [NAMES[c] for c in wrank],
        'ranking_changed': [NAMES[c] for c in grank] != [NAMES[c] for c in wrank],
        'pairwise': pairs,
    }
    (RES / 'independent_replication.json').write_text(json.dumps(report, indent=2))
    log(f"\nranking changed: {report['ranking_changed']}")
    log(f"wrote {RES / 'independent_replication.json'}")

if __name__ == '__main__':
    main()
