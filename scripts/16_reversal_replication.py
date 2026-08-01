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

PREDICTORS = {'varity': 'VARITY_R_LOO', 'alphamissense': 'AlphaMissense',
              'gmvp': 'gMVP'}
COLS = list(PREDICTORS)
N_REPEATS = 200

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
    return d.dropna(subset=COLS).reset_index(drop=True)

def within_gene(d, min_per_class=3):
    rows = []
    for gene, grp in d.groupby('gene_symbol'):
        npos = int((grp.label == 1).sum())
        nneg = int((grp.label == 0).sum())
        if npos < min_per_class or nneg < min_per_class:
            continue
        r = {'gene_symbol': gene, 'n': len(grp)}
        for c in COLS:
            r[c] = roc_auc_score(grp.label, grp[c])
        rows.append(r)
    return pd.DataFrame(rows)

def ranking(vals):
    return [PREDICTORS[c] for c in sorted(COLS, key=lambda c: -vals[c])]

def main():
    d = load()
    log(f"{len(d):,} variants, {d.gene_symbol.nunique():,} genes")

    global_auc = {c: float(roc_auc_score(d.label, d[c])) for c in COLS}
    global_rank = ranking(global_auc)
    log(f"global ranking: {' > '.join(global_rank)}")

    w_full = within_gene(d)
    full_rank = ranking({c: w_full[c].mean() for c in COLS})
    log(f"within-gene ranking (all {len(w_full):,} genes): {' > '.join(full_rank)}")

    report = {'global_ranking': global_rank,
              'within_gene_ranking_full': full_rank,
              'n_genes_evaluable': int(len(w_full))}

    log(f"\nSPLIT-HALF REPLICATION ({N_REPEATS} random gene partitions)")
    genes = w_full.gene_symbol.to_numpy()
    rng = np.random.default_rng(42)
    both_agree = 0
    agree_with_full = 0
    for i in range(N_REPEATS):
        perm = rng.permutation(len(genes))
        h1 = w_full.iloc[perm[:len(perm) // 2]]
        h2 = w_full.iloc[perm[len(perm) // 2:]]
        r1 = ranking({c: h1[c].mean() for c in COLS})
        r2 = ranking({c: h2[c].mean() for c in COLS})
        if r1 == r2:
            both_agree += 1
        if r1 == full_rank and r2 == full_rank:
            agree_with_full += 1
    report['split_half'] = {
        'n_repeats': N_REPEATS,
        'halves_agree_with_each_other': both_agree / N_REPEATS,
        'both_halves_match_full_ranking': agree_with_full / N_REPEATS,
    }
    log(f"  halves agree with each other: {both_agree / N_REPEATS:.1%}")
    log(f"  both halves match full ranking: {agree_with_full / N_REPEATS:.1%}")

    log("\nTHRESHOLD SENSITIVITY")
    thr = {}
    for m in (2, 3, 5, 10, 20):
        w = within_gene(d, min_per_class=m)
        if len(w) < 30:
            continue
        r = ranking({c: w[c].mean() for c in COLS})
        thr[f'min_per_class_{m}'] = {
            'n_genes': int(len(w)), 'ranking': r,
            'matches_full': r == full_rank,
            'means': {PREDICTORS[c]: float(w[c].mean()) for c in COLS},
        }
        log(f"  >={m:>2}/class  {len(w):>5} genes  {' > '.join(r)}"
            f"{'' if r == full_rank else '   <-- DIFFERS'}")
    report['threshold_sensitivity'] = thr

    log("\nLEAVE-ONE-GENE-OUT INFLUENCE")
    flips = 0
    for i in range(len(w_full)):
        sub = w_full.drop(w_full.index[i])
        if ranking({c: sub[c].mean() for c in COLS}) != full_rank:
            flips += 1
    report['leave_one_gene_out'] = {
        'n_genes': int(len(w_full)),
        'n_removals_that_change_ranking': int(flips),
        'fraction': flips / len(w_full),
    }
    log(f"  removing any single gene changes the ranking in "
        f"{flips}/{len(w_full)} cases ({flips / len(w_full):.2%})")

    pair = {}
    for a, b in combinations(COLS, 2):
        gd = global_auc[a] - global_auc[b]
        wd = float((w_full[a] - w_full[b]).mean())
        pair[f'{PREDICTORS[a]} vs {PREDICTORS[b]}'] = {
            'global_diff': gd, 'within_gene_diff': wd,
            'reversed': bool(np.sign(gd) != np.sign(wd)),
        }
    report['pairwise'] = pair

    (RES / 'reversal_replication.json').write_text(json.dumps(report, indent=2))
    log(f"\nwrote {RES / 'reversal_replication.json'}")

if __name__ == '__main__':
    main()
