import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
AM = SCRATCH / "alphamissense_aa.tsv.gz"
VARITY = SCRATCH / "varity.tar.gz"
GMVP = SCRATCH / "gmvp.csv.gz"
CHUNK = 5_000_000

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def targets():
    p = DATA / 'pgym_clinical_mapped.parquet'
    if not p.exists():
        log("run 19_independent_replication.py first")
        return None, None, None
    d = pd.read_parquet(p)
    accs = set(d.uniprot_acc)
    keys = set(zip(d.uniprot_acc, d.position_1, d.wt_aa, d.mut_aa))
    u = pd.read_parquet(UNIPROT)
    a2g = {}
    for acc, gn in zip(u.accession, u.gene_names):
        if isinstance(gn, str) and gn.split():
            a2g[acc] = gn.split()[0].upper()
    genes = {a2g[a] for a in accs if a in a2g}
    log(f"{len(accs):,} proteins, {len(keys):,} substitutions, "
        f"{len(genes):,} gene symbols")
    return accs, keys, genes

def scan_am(accs, keys):
    out = DATA / 'replication_scores_alphamissense.parquet'
    if out.exists():
        log("AlphaMissense already extracted"); return
    hits, seen = [], 0
    for i, ch in enumerate(pd.read_csv(
            AM, sep='\t', comment='#', chunksize=CHUNK, low_memory=False,
            names=['uniprot_id', 'protein_variant', 'am_pathogenicity',
                   'am_class'], header=0)):
        seen += len(ch)
        ch = ch[ch.uniprot_id.isin(accs)]
        if len(ch):
            pos = ch.protein_variant.str[1:-1]
            ok = pos.str.isdigit()
            ch = ch[ok]
            if len(ch):
                ch = ch.assign(position_1=pos[ok].astype(int),
                               wt_aa=ch.protein_variant.str[0],
                               mut_aa=ch.protein_variant.str[-1])
                ch = ch[[k in keys for k in zip(ch.uniprot_id, ch.position_1,
                                                ch.wt_aa, ch.mut_aa)]]
                if len(ch):
                    hits.append(ch[['uniprot_id', 'position_1', 'wt_aa',
                                    'mut_aa', 'am_pathogenicity']])
        if i % 12 == 0:
            log(f"  AM {seen:,} rows, {sum(len(h) for h in hits):,} kept")
    a = pd.concat(hits, ignore_index=True).rename(
        columns={'uniprot_id': 'uniprot_acc', 'am_pathogenicity': 'alphamissense'})
    a.drop_duplicates().to_parquet(out, index=False)
    log(f"AlphaMissense: {len(a):,} matched")

def scan_varity(accs, keys):
    out = DATA / 'replication_scores_varity.parquet'
    if out.exists():
        log("VARITY already extracted"); return
    import tarfile
    t = tarfile.open(VARITY)
    m = [x for x in t.getmembers() if x.isfile()][0]
    hits, seen = [], 0
    for i, ch in enumerate(pd.read_csv(
            t.extractfile(m), sep='\t', chunksize=2_000_000, low_memory=False,
            usecols=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'VARITY_R_LOO'])):
        seen += len(ch)
        ch = ch[ch.p_vid.isin(accs)]
        if len(ch):
            ch = ch.dropna(subset=['aa_pos'])
            ch = ch.assign(aa_pos=pd.to_numeric(ch.aa_pos, errors='coerce'))
            ch = ch.dropna(subset=['aa_pos'])
            ch = ch.assign(aa_pos=ch.aa_pos.astype(int))
            ch = ch[[k in keys for k in zip(ch.p_vid, ch.aa_pos,
                                            ch.aa_ref, ch.aa_alt)]]
            if len(ch):
                hits.append(ch)
        if i % 10 == 0:
            log(f"  VARITY {seen:,} rows, {sum(len(h) for h in hits):,} kept")
    v = pd.concat(hits, ignore_index=True).rename(columns={
        'p_vid': 'uniprot_acc', 'aa_pos': 'position_1', 'aa_ref': 'wt_aa',
        'aa_alt': 'mut_aa', 'VARITY_R_LOO': 'varity'})
    v[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa', 'varity']] \
        .drop_duplicates().to_parquet(out, index=False)
    log(f"VARITY: {len(v):,} matched")

def scan_gmvp(genes):
    out = DATA / 'replication_scores_gmvp.parquet'
    if out.exists():
        log("gMVP already extracted"); return
    hits, seen = [], 0
    for i, ch in enumerate(pd.read_csv(
            GMVP, sep='\t', chunksize=2_000_000, low_memory=False,
            usecols=['gene_symbol', 'protein_position', 'ref_aa', 'alt_aa',
                     'gMVP'])):
        seen += len(ch)
        ch = ch[ch.gene_symbol.isin(genes)]
        if len(ch):
            ch = ch.dropna(subset=['protein_position'])
            ch = ch.assign(protein_position=pd.to_numeric(
                ch.protein_position, errors='coerce'))
            ch = ch.dropna(subset=['protein_position'])
            hits.append(ch)
        if i % 15 == 0:
            log(f"  gMVP {seen:,} rows, {sum(len(h) for h in hits):,} kept")
    g = pd.concat(hits, ignore_index=True).rename(columns={
        'protein_position': 'position_1', 'ref_aa': 'wt_aa',
        'alt_aa': 'mut_aa', 'gMVP': 'gmvp'})
    g = g.dropna(subset=['position_1'])
    g['position_1'] = pd.to_numeric(g.position_1, errors='coerce').astype('Int64')
    g = g.dropna(subset=['position_1'])
    g['position_1'] = g.position_1.astype(int)
    g[['gene_symbol', 'position_1', 'wt_aa', 'mut_aa', 'gmvp']] \
        .drop_duplicates().to_parquet(out, index=False)
    log(f"gMVP: {len(g):,} rows kept")

def main():
    accs, keys, genes = targets()
    if not accs:
        return
    scan_am(accs, keys)
    scan_varity(accs, keys)
    scan_gmvp(genes)
    (DATA / 'replication_scores_metadata.json').write_text(json.dumps({
        'n_proteins': len(accs), 'n_substitutions': len(keys),
        'note': 'scores extracted for the ProteinGym clinical replication set',
    }, indent=2))
    log("done")

if __name__ == '__main__':
    main()
