import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
AM = SCRATCH / "alphamissense_aa.tsv.gz"
GMVP = SCRATCH / "gmvp.csv.gz"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
CHUNK = 5_000_000

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def assay_proteins():
    f = RES / 'proteingym_mito_assays.csv'
    if not f.exists():
        log("run 08_proteingym.py --list first")
        return None, None
    hit = pd.read_csv(f)
    genes = sorted({str(d).split('_')[0].upper() for d in hit.DMS_id})
    u = pd.read_parquet(UNIPROT)
    g2a = {}
    for acc, gn in zip(u.accession, u.gene_names):
        if isinstance(gn, str):
            for g in gn.split():
                g2a.setdefault(g.upper(), acc)
    accs = {g: g2a.get(g) for g in genes}
    log(f"assay proteins: {accs}")
    return genes, {a for a in accs.values() if a}

def scan_alphamissense(accs):
    if not AM.exists():
        log(f"missing {AM}")
        return
    hits, seen = [], 0
    reader = pd.read_csv(AM, sep='\t', comment='#', chunksize=CHUNK,
                         low_memory=False,
                         names=['uniprot_id', 'protein_variant',
                                'am_pathogenicity', 'am_class'], header=0)
    for i, ch in enumerate(reader):
        seen += len(ch)
        ch = ch[ch.uniprot_id.isin(accs)]
        if len(ch):
            hits.append(ch)
        if i % 10 == 0:
            log(f"  AlphaMissense {seen:,} rows, "
                f"{sum(len(h) for h in hits):,} matched")
    if not hits:
        log("AlphaMissense: nothing matched")
        return
    a = pd.concat(hits, ignore_index=True)
    a = a.rename(columns={'uniprot_id': 'uniprot_acc',
                          'am_pathogenicity': 'alphamissense'})
    a['position_1'] = a.protein_variant.str[1:-1].astype(int)
    a['wt_aa'] = a.protein_variant.str[0]
    a['mut_aa'] = a.protein_variant.str[-1]
    a[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa', 'alphamissense']].to_parquet(
        DATA / 'assay_protein_scores_alphamissense.parquet', index=False)
    log(f"AlphaMissense: {len(a):,} substitutions across {a.uniprot_acc.nunique()} proteins")
    return int(len(a))

def scan_gmvp(genes):
    if not GMVP.exists():
        log(f"missing {GMVP}")
        return
    hits, seen = [], 0
    reader = pd.read_csv(GMVP, sep='\t', chunksize=2_000_000, low_memory=False,
                         usecols=['gene_symbol', 'protein_position', 'ref_aa',
                                  'alt_aa', 'gMVP'])
    gset = set(genes)
    for i, ch in enumerate(reader):
        seen += len(ch)
        ch = ch[ch.gene_symbol.isin(gset)]
        if len(ch):
            hits.append(ch)
        if i % 15 == 0:
            log(f"  gMVP {seen:,} rows, {sum(len(h) for h in hits):,} matched")
    if not hits:
        log("gMVP: nothing matched")
        return
    g = pd.concat(hits, ignore_index=True).rename(
        columns={'protein_position': 'position_1', 'ref_aa': 'wt_aa',
                 'alt_aa': 'mut_aa', 'gMVP': 'gmvp'})
    g = g.dropna(subset=['position_1'])
    g['position_1'] = pd.to_numeric(g.position_1, errors='coerce').astype('Int64')
    g = g.dropna(subset=['position_1'])
    g['position_1'] = g.position_1.astype(int)
    g = g.drop_duplicates(subset=['gene_symbol', 'position_1', 'wt_aa', 'mut_aa'])
    g[['gene_symbol', 'position_1', 'wt_aa', 'mut_aa', 'gmvp']].to_parquet(
        DATA / 'assay_protein_scores_gmvp.parquet', index=False)
    log(f"gMVP: {len(g):,} substitutions across {g.gene_symbol.nunique()} genes")
    return int(len(g))

def scan_varity(accs):
    varity = SCRATCH / "varity.tar.gz"
    if not varity.exists():
        log(f"missing {varity}")
        return
    import tarfile
    t = tarfile.open(varity)
    members = [m for m in t.getmembers() if m.isfile()]
    if not members:
        return
    hits, seen = [], 0
    reader = pd.read_csv(
        t.extractfile(members[0]), sep='\t', chunksize=2_000_000,
        low_memory=False,
        usecols=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'VARITY_R_LOO'])
    for i, ch in enumerate(reader):
        seen += len(ch)
        ch = ch[ch.p_vid.isin(accs)]
        if len(ch):
            hits.append(ch)
        if i % 10 == 0:
            log(f"  VARITY {seen:,} rows, {sum(len(h) for h in hits):,} matched")
    if not hits:
        log("VARITY: nothing matched")
        return
    v = pd.concat(hits, ignore_index=True).rename(columns={
        'p_vid': 'uniprot_acc', 'aa_pos': 'position_1', 'aa_ref': 'wt_aa',
        'aa_alt': 'mut_aa', 'VARITY_R_LOO': 'varity'})
    v = v.dropna(subset=['position_1'])
    v['position_1'] = pd.to_numeric(v.position_1, errors='coerce').astype('Int64')
    v = v.dropna(subset=['position_1'])
    v['position_1'] = v.position_1.astype(int)
    v = v.drop_duplicates(subset=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'])
    v.to_parquet(DATA / 'assay_protein_scores_varity.parquet', index=False)
    log(f"VARITY: {len(v):,} substitutions across {v.uniprot_acc.nunique()} proteins")
    return int(len(v))

def main():
    genes, accs = assay_proteins()
    if not genes:
        return
    out = {'genes': genes, 'accessions': sorted(accs)}
    if not (DATA / 'assay_protein_scores_alphamissense.parquet').exists():
        n = scan_alphamissense(accs)
        if n:
            out['alphamissense_substitutions'] = n
    if not (DATA / 'assay_protein_scores_gmvp.parquet').exists():
        n = scan_gmvp(genes)
        if n:
            out['gmvp_substitutions'] = n
    n = scan_varity(accs)
    if n:
        out['varity_substitutions'] = n
    (DATA / 'assay_protein_scores_metadata.json').write_text(
        json.dumps(out, indent=2))
    log("done")

if __name__ == '__main__':
    main()
