import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
PG = BASE / "data" / "proteingym"
RES = BASE / "results" / "rebuild"

MIN_SUBS = 100
MIN_COVERAGE = 0.50

GLOBAL_RANK = ['gMVP', 'VARITY_R_LOO', 'AlphaMissense']
WITHIN_RANK = ['AlphaMissense', 'VARITY_R_LOO', 'gMVP']

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def main():
    ref = pd.read_csv(DATA / 'DMS_substitutions_reference.csv')
    human = ref[ref.source_organism.astype(str).str.contains('Homo sapiens',
                                                             na=False)]
    log(f"{len(human)} human assays of {len(ref)} in ProteinGym")

    dms = pd.read_parquet(PG / 'DMS_substitutions.parquet')
    dms = dms[dms.DMS_id.isin(set(human.DMS_id))]
    dms = dms[~dms.mutant.str.contains(':', na=False)].copy()
    dms['wt_aa'] = dms.mutant.str[0]
    dms['position_1'] = pd.to_numeric(dms.mutant.str[1:-1], errors='coerce')
    dms['mut_aa'] = dms.mutant.str[-1]
    dms = dms.dropna(subset=['position_1'])
    dms['position_1'] = dms.position_1.astype(int)
    log(f"{len(dms):,} single substitutions across {dms.DMS_id.nunique()} assays")

    up = pd.read_parquet(BASE / 'data' / 'raw' / 'uniprot_human_reviewed.parquet')
    seq2acc = {}
    for a, s in zip(up.accession, up.sequence):
        if isinstance(s, str):
            seq2acc.setdefault(s, a)
    tgt = human.set_index('DMS_id').target_seq.to_dict()

    need = ['human_assay_scores_alphamissense.parquet',
            'human_assay_scores_varity.parquet',
            'human_assay_scores_gmvp.parquet']
    missing = [n for n in need if not (DATA / n).exists()]
    if missing:
        raise SystemExit(f"run 26_human_assay_scores.py first; missing {missing}")

    am = pd.read_parquet(DATA / 'human_assay_scores_alphamissense.parquet')
    va = pd.read_parquet(DATA / 'human_assay_scores_varity.parquet')
    gm = pd.read_parquet(DATA / 'human_assay_scores_gmvp.parquet')

    def to_common(d, nm):
        expect = ['key', 'position_1', 'wt_aa', 'mut_aa', 'score']
        if list(d.columns) == expect:
            return d
        cols = {c.lower(): c for c in d.columns}
        kc = next((cols[k] for k in ('key', 'uniprot_acc', 'p_vid', 'genename')
                   if k in cols), None)
        sc = next((cols[s] for s in ('score', 'alphamissense', 'varity', 'gmvp')
                   if s in cols), None)
        assert kc and sc, (nm, list(d.columns))
        o = d[[kc, 'position_1', 'wt_aa', 'mut_aa', sc]].copy()
        o.columns = expect
        o['key'] = o.key.astype(str)
        return o

    am, va, gm = (to_common(am, 'AlphaMissense'), to_common(va, 'VARITY'),
                  to_common(gm, 'gMVP'))
    log(f"per-protein score tables: AlphaMissense {len(am):,}, "
        f"VARITY {len(va):,}, gMVP {len(gm):,}")

    tg = pd.read_csv(DATA / 'human_assay_targets.csv')
    sym = dict(zip(tg.uniprot_acc, tg.gene_symbol))

    rows = []
    for dms_id, grp in dms.groupby('DMS_id'):
        if len(grp) < MIN_SUBS:
            continue
        acc = seq2acc.get(tgt.get(dms_id, ''))
        if acc is None:
            continue
        rec = {'DMS_id': dms_id, 'uniprot_acc': acc, 'n_subs': len(grp)}
        g = grp[['position_1', 'wt_aa', 'mut_aa', 'DMS_score']]

        s = sym.get(acc)
        j = g.copy()
        for tbl, k, col in ((am, acc, 'alphamissense'), (va, acc, 'varity'),
                            (gm, s, 'gmvp')):
            sub = (tbl[tbl.key == k][['position_1', 'wt_aa', 'mut_aa', 'score']]
                   .rename(columns={'score': col})
                   if k is not None else
                   pd.DataFrame(columns=['position_1', 'wt_aa', 'mut_aa', col]))
            j = j.merge(sub, on=['position_1', 'wt_aa', 'mut_aa'], how='left')

        cols3 = ['alphamissense', 'varity', 'gmvp']
        for col, name in zip(cols3, ('AlphaMissense', 'VARITY_R_LOO', 'gMVP')):
            rec[f'{name}_cov'] = float(j[col].notna().mean())
        common = j.dropna(subset=cols3)
        rec['n_common'] = int(len(common))
        rec['frac_common'] = float(len(common) / len(j)) if len(j) else 0.0
        for col, name in zip(cols3, ('AlphaMissense', 'VARITY_R_LOO', 'gMVP')):
            rec[f'{name}_rho'] = (float(-spearmanr(common.DMS_score,
                                                   common[col]).statistic)
                                  if len(common) >= MIN_SUBS else np.nan)
        rows.append(rec)

    t = pd.DataFrame(rows)
    preds = ['AlphaMissense', 'VARITY_R_LOO', 'gMVP']
    full = t.dropna(subset=[f'{p}_rho' for p in preds]).reset_index(drop=True)
    log(f"\n{len(t)} assays mapped; {len(full)} with a common intersection of "
        f">={MIN_SUBS} substitutions scored by all three")
    if len(full):
        log(f"  common intersection: median {full.n_common.median():,.0f} "
            f"substitutions ({full.frac_common.median():.0%} of the assay)")
        log("  per-predictor coverage (median): " + ", ".join(
            f"{p} {full[f'{p}_cov'].median():.0%}" for p in preds))

    if len(full) < 10:
        raise SystemExit(
            f"FATAL: only {len(full)} assays are covered by all three predictors "
            f"at >={MIN_COVERAGE:.0%}; the arbitration needs at least 10.\n"
            "The predictor tables here are ClinVar-restricted, while a DMS assay "
            "covers every substitution in a protein, so most assay positions have "
            "no score. Extract scores per protein from the full releases first "
            "(see 12_assay_protein_scores.py) and rerun.")

    log("\nMEAN SPEARMAN AGAINST EXPERIMENT")
    means = {p: float(full[f'{p}_rho'].mean()) for p in preds}
    dms_rank = sorted(preds, key=lambda p: -means[p])
    for i, p in enumerate(dms_rank, 1):
        log(f"  {i}. {p:<16} {means[p]:.4f}")

    log("\nPAIRWISE, PAIRED ACROSS ASSAYS")
    pairs = {}
    for a, b in combinations(preds, 2):
        d_ = full[f'{a}_rho'] - full[f'{b}_rho']
        p = float(wilcoxon(full[f'{a}_rho'], full[f'{b}_rho']).pvalue)
        wins = int((d_ > 0).sum())
        pairs[f'{a} vs {b}'] = {'mean_diff': float(d_.mean()), 'wins': wins,
                                'n': int(len(d_)), 'p': p}
        log(f"  {a:<14} vs {b:<14} mean {d_.mean():+.4f}  "
            f"wins {wins}/{len(d_)}  p={p:.2e}")

    log("\nWHICH ClinVar RANKING DOES EXPERIMENT AGREE WITH?")
    log(f"  global ClinVar ranking      {' > '.join(GLOBAL_RANK)}")
    log(f"  within-gene ClinVar ranking {' > '.join(WITHIN_RANK)}")
    log(f"  DMS ranking                 {' > '.join(dms_rank)}")
    rho_g = spearmanr([GLOBAL_RANK.index(p) for p in preds],
                      [dms_rank.index(p) for p in preds]).statistic
    rho_w = spearmanr([WITHIN_RANK.index(p) for p in preds],
                      [dms_rank.index(p) for p in preds]).statistic
    log(f"\n  rank agreement with GLOBAL      rho = {rho_g:+.3f}")
    log(f"  rank agreement with WITHIN-GENE rho = {rho_w:+.3f}")
    verdict = ("within-gene" if rho_w > rho_g else
               "global" if rho_g > rho_w else "neither/tied")
    log(f"  -> experiment agrees with the {verdict.upper()} ranking")

    out = {
        'n_human_assays_total': int(len(human)),
        'n_assays_mapped': int(len(t)),
        'n_assays_all_three': int(len(full)),
        'min_substitutions': MIN_SUBS,
        'min_coverage': MIN_COVERAGE,
        'mean_spearman': means,
        'dms_ranking': dms_rank,
        'global_clinvar_ranking': GLOBAL_RANK,
        'within_gene_clinvar_ranking': WITHIN_RANK,
        'rank_agreement_global': float(rho_g),
        'rank_agreement_within_gene': float(rho_w),
        'verdict': verdict,
        'pairwise': pairs,
        'note': 'DMS agreement is structurally a within-protein comparison: every '
                'substitution is in one protein, so no between-gene label '
                'structure exists to exploit and the gene-identity null is '
                'undefined. Spearman signs are flipped so higher always means '
                'better agreement.',
        'caveat': 'Assays differ in what they measure (abundance, binding, '
                  'activity), and agreement with a functional assay is not the '
                  'same quantity as clinical pathogenicity. This arbitrates '
                  'between two rankings; it does not validate either metric '
                  'as a measure of clinical utility.',
    }
    full.to_csv(RES / 'dms_arbitration.csv', index=False)
    (RES / 'dms_arbitration.json').write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'dms_arbitration.csv'}")

if __name__ == '__main__':
    main()
