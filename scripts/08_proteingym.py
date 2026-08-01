import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
RES.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
DMS_PARQUET = SCRATCH / "pgym_dms.parquet"
DMS_URL = "https://proteingym.s3.us-east-2.amazonaws.com/DMS_substitutions.parquet"
REF_URL = ("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
           "reference_files/DMS_substitutions.csv")
MITOCARTA = SCRATCH / "mitocarta.xls"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def mitocarta_symbols():
    mc = pd.ExcelFile(MITOCARTA).parse('A Human MitoCarta3.0')
    return set(mc.Symbol.astype(str).str.upper())

def load_reference():
    f = DATA / 'DMS_substitutions_reference.csv'
    if not f.exists():
        r = requests.get(REF_URL, timeout=120)
        r.raise_for_status()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(r.content)
    return pd.read_csv(f)

def find_mito_assays(ref):
    syms = mitocarta_symbols()
    ref = ref.copy()
    ref['gene_guess'] = ref['DMS_id'].astype(str).str.split('_').str[0].str.upper()
    if 'UniProt_ID' in ref.columns:
        is_human = ref['UniProt_ID'].astype(str).str.endswith('_HUMAN')
    else:
        is_human = ref['DMS_id'].astype(str).str.contains('_HUMAN_')
    hit = ref[is_human & ref.gene_guess.isin(syms)]
    rejected = ref[~is_human & ref.gene_guess.isin(syms)]
    return hit, rejected

def fetch_dms():
    if not DMS_PARQUET.exists():
        log(f"downloading DMS substitutions from {DMS_URL}")
        with requests.get(DMS_URL, stream=True, timeout=3600) as r:
            r.raise_for_status()
            with open(DMS_PARQUET, 'wb') as f:
                for c in r.iter_content(1 << 20):
                    f.write(c)
    return pd.read_parquet(DMS_PARQUET)

def align_positions(target, canonical, min_identity=0.95):
    from Bio import Align
    from Bio.Align import substitution_matrices

    off = canonical.find(target)
    if off >= 0:
        return {i + 1: i + 1 + off for i in range(len(target))}, 1.0

    aligner = Align.PairwiseAligner(
        mode='global',
        substitution_matrix=substitution_matrices.load('BLOSUM62'),
        open_gap_score=-11, extend_gap_score=-1)
    try:
        aln = aligner.align(canonical, target)[0]
    except (ValueError, MemoryError, OverflowError):
        return None, 0.0

    A, B = aln[0], aln[1]
    ci = ti = 0
    matches = aligned = 0
    pos_map = {}
    for a, b in zip(A, B):
        if a != '-' and b != '-':
            ci += 1; ti += 1
            aligned += 1
            if a == b:
                matches += 1
                pos_map[ti] = ci
        elif a != '-':
            ci += 1
        elif b != '-':
            ti += 1
    identity = matches / aligned if aligned else 0.0
    if identity < min_identity:
        return None, identity
    return pos_map, identity

def uniprot_sequences():
    u = pd.read_parquet(UNIPROT)
    seq, g2a = {}, {}
    for acc, genes, s in zip(u.accession, u.gene_names, u.sequence):
        if not isinstance(s, str):
            continue
        seq[acc] = s
        if isinstance(genes, str):
            for g in genes.split():
                g2a.setdefault(g.upper(), acc)
    return seq, g2a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()

    ref = load_reference()
    log(f"ProteinGym reference: {len(ref)} assays")
    hit, rejected = find_mito_assays(ref)

    cols = [c for c in ['DMS_id', 'UniProt_ID', 'taxon', 'seq_len',
                        'DMS_total_number_mutants'] if c in hit.columns]
    if len(rejected):
        log(f"\n{len(rejected)} symbol matches REJECTED as non-human:")
        log("\n" + rejected[cols].to_string(index=False))
    log(f"\n{len(hit)} human assays on MitoCarta 3.0 genes:")
    log("\n" + hit[cols].to_string(index=False))
    hit.to_csv(RES / 'proteingym_mito_assays.csv', index=False)

    if len(hit) < 3:
        log("\nFewer than 3 assays: reported as a limitation, not worked around.")
    if not args.evaluate:
        log("\nrun with --evaluate to score predictors")
        return

    dms = fetch_dms()
    log(f"DMS table: {len(dms):,} rows, {dms.DMS_id.nunique()} assays")
    seqs, g2a = uniprot_sequences()

    scores = {}
    am_full = DATA / 'assay_protein_scores_alphamissense.parquet'
    am = am_full if am_full.exists() else DATA / 'alphamissense_scores.parquet'
    if am.exists():
        a = pd.read_parquet(am)
        if 'protein_variant' in a.columns:
            a['position_1'] = a.protein_variant.str[1:-1].astype(int)
            a['wt_aa'] = a.protein_variant.str[0]
            a['mut_aa'] = a.protein_variant.str[-1]
        scores['AlphaMissense'] = (a[['uniprot_acc', 'position_1', 'wt_aa',
                                      'mut_aa', 'alphamissense']]
                                   .rename(columns={'alphamissense': 'score'}),
                                   'uniprot_acc')
        log(f"AlphaMissense table: {am.name} ({len(a):,} rows)")
    gm_full = DATA / 'assay_protein_scores_gmvp.parquet'
    gm = gm_full if gm_full.exists() else DATA / 'gmvp_scores.parquet'
    if gm.exists():
        g = pd.read_parquet(gm).rename(columns={'gmvp': 'score'})
        scores['gMVP'] = (g, 'gene_symbol')
        log(f"gMVP table: {gm.name} ({len(g):,} rows)")
    va_full = DATA / 'assay_protein_scores_varity.parquet'
    va = va_full if va_full.exists() else DATA / 'varity_scores.parquet'
    if va.exists():
        v = pd.read_parquet(va)
        if 'varity' in v.columns:
            scores['VARITY'] = (v.rename(columns={'varity': 'score'}),
                                'uniprot_acc')
            log(f"VARITY table: {va.name} ({len(v):,} rows, VARITY_R_LOO)")
    esm_full = RES / 'esm2_assay_proteins.parquet'
    esm = esm_full if esm_full.exists() else RES / 'esm2_zeroshot_mito.parquet'
    if esm.exists():
        e = pd.read_parquet(esm)[['uniprot_acc', 'position_1', 'wt_aa',
                                  'mut_aa', 'esm2_score']]
        scores['ESM-2 zero-shot'] = (e.rename(columns={'esm2_score': 'score'}),
                                     'uniprot_acc')
        log(f"ESM-2 table: {esm.name} ({len(e):,} rows)")
    log(f"predictors available: {list(scores)}")

    rows, skipped = [], []
    for _, a in hit.iterrows():
        dms_id = a.DMS_id
        d = dms[dms.DMS_id == dms_id].copy()
        if not len(d):
            skipped.append((dms_id, 'no rows in DMS table'))
            continue
        gene = str(dms_id).split('_')[0].upper()
        acc = g2a.get(gene)
        if acc is None:
            skipped.append((dms_id, 'gene not in UniProt index'))
            continue
        canonical = seqs[acc]
        target = d.target_seq.iloc[0]

        pos_map, identity = align_positions(target, canonical)
        if pos_map is None:
            skipped.append((dms_id, 'assay sequence does not align to UniProt '
                                    'canonical'))
            continue

        d['wt_aa'] = d.mutant.str[0]
        d['mut_aa'] = d.mutant.str[-1]
        d['assay_position'] = d.mutant.str[1:-1].astype(int)
        d['position_1'] = d.assay_position.map(pos_map)
        d = d[d.position_1.notna()]
        d['position_1'] = d.position_1.astype(int)

        ok = [0 <= p - 1 < len(canonical) and canonical[p - 1] == w
              for p, w in zip(d.position_1, d.wt_aa)]
        frac_ok = float(np.mean(ok)) if len(ok) else 0.0
        if frac_ok < 0.95:
            skipped.append((dms_id, f'position mapping validated for only '
                                    f'{frac_ok:.0%} of substitutions'))
            continue
        d = d[ok]
        d['uniprot_acc'] = acc
        d['gene_symbol'] = gene

        row = {'DMS_id': dms_id, 'gene': gene, 'uniprot_acc': acc,
               'n_substitutions': int(len(d)),
               'alignment_identity': round(identity, 4),
               'position_mapping_ok': round(frac_ok, 4)}
        for nm, (tbl, key) in scores.items():
            j = d.merge(tbl, on=[key, 'position_1', 'wt_aa', 'mut_aa'],
                        how='inner')
            if len(j) > 30:
                rho = spearmanr(j.DMS_score, j.score).statistic
                row[f'{nm}_spearman'] = float(-rho)
                row[f'{nm}_n'] = int(len(j))
        rows.append(row)
        log(f"  {dms_id}: {len(d):,} substitutions, "
            f"alignment identity {identity:.1%}, mapping {frac_ok:.0%}")

    out = pd.DataFrame(rows)
    if len(out):
        pd.set_option('display.width', 200)
        log("\n" + out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    for s in skipped:
        log(f"  SKIPPED {s[0]}: {s[1]}")

    out.to_csv(RES / 'proteingym_results.csv', index=False)
    (RES / 'proteingym_summary.json').write_text(json.dumps({
        'n_assays_available': int(len(hit)),
        'n_assays_scored': int(len(out)),
        'skipped': [{'DMS_id': a, 'reason': b} for a, b in skipped],
        'metric': 'Spearman correlation against DMS fitness, sign-flipped so '
                  'that a higher value means better agreement with the assay',
        'source': 's3://proteingym/DMS_substitutions.parquet',
    }, indent=2))
    log(f"wrote {RES / 'proteingym_results.csv'}")

if __name__ == '__main__':
    main()
