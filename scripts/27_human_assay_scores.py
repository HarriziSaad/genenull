import json
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
PG = BASE / "data" / "proteingym"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
AM = SCRATCH / "alphamissense_aa.tsv.gz"
GMVP = SCRATCH / "gmvp.csv.gz"
VARITY = SCRATCH / "varity.tar.gz"
CHUNK = 5_000_000

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def slim(d, score_names, key_names):
    cols = {c.lower(): c for c in d.columns}
    pick = lambda names: next((cols[n] for n in names if n in cols), None)
    sc, kc = pick(score_names), pick(key_names)
    pc = pick(['protein_position', 'aapos', 'aa_pos', 'position_1', 'pos'])
    wc = pick(['ref_aa', 'wt_aa', 'aa_ref', 'aaref'])
    mc = pick(['alt_aa', 'mut_aa', 'aa_alt', 'aaalt'])
    assert 'pos' not in (str(pc).lower(),) or pc == 'position_1', (
        f"refusing to use a genomic position column: {pc}")
    assert None not in (sc, kc, pc, wc, mc), (
        f"cannot map columns: score={sc} key={kc} pos={pc} wt={wc} mut={mc}; "
        f"available {list(d.columns)[:20]}")
    o = d[[kc, pc, wc, mc, sc]].copy()
    o.columns = ['key', 'position_1', 'wt_aa', 'mut_aa', 'score']
    o['position_1'] = pd.to_numeric(o.position_1, errors='coerce')
    o = o.dropna(subset=['position_1'])
    o['position_1'] = o.position_1.astype(int)
    o['key'] = o.key.astype(str)
    return o

def targets():
    ref = pd.read_csv(DATA / 'DMS_substitutions_reference.csv')
    human = ref[ref.source_organism.astype(str).str.contains('Homo sapiens',
                                                             na=False)]
    u = pd.read_parquet(UNIPROT)
    seq2acc = {}
    for a, s in zip(u.accession, u.sequence):
        if isinstance(s, str):
            seq2acc.setdefault(s, a)
    acc2sym = {}
    for a, gn in zip(u.accession, u.gene_names):
        if isinstance(gn, str) and gn.split():
            acc2sym[a] = gn.split()[0].upper()

    rows = []
    for dms_id, tseq in zip(human.DMS_id, human.target_seq):
        acc = seq2acc.get(tseq)
        if acc:
            rows.append({'DMS_id': dms_id, 'uniprot_acc': acc,
                         'gene_symbol': acc2sym.get(acc)})
    t = pd.DataFrame(rows)
    log(f"{len(human)} human assays -> {len(t)} matched to a canonical UniProt "
        f"sequence ({t.uniprot_acc.nunique()} distinct proteins)")
    return t

def scan_alphamissense(accs, out):
    if out.exists():
        log(f"{out.name} exists, skipping")
        return
    log(f"scanning AlphaMissense for {len(accs)} accessions")
    keep, n = [], 0
    reader = pd.read_csv(AM, sep='\t', comment='#', chunksize=CHUNK,
                         usecols=['uniprot_id', 'protein_variant',
                                  'am_pathogenicity'])
    for i, ch in enumerate(reader, 1):
        h = ch[ch.uniprot_id.isin(accs)]
        n += len(ch)
        if len(h):
            keep.append(h)
        if i % 5 == 0:
            log(f"  {n:,} rows, {sum(map(len, keep)):,} kept")
    d = pd.concat(keep, ignore_index=True)
    d = d.rename(columns={'uniprot_id': 'uniprot_acc',
                          'am_pathogenicity': 'alphamissense'})
    d['position_1'] = d.protein_variant.str[1:-1].astype(int)
    d['wt_aa'] = d.protein_variant.str[0]
    d['mut_aa'] = d.protein_variant.str[-1]
    d[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa',
       'alphamissense']].to_parquet(out, index=False)
    log(f"wrote {out.name}: {len(d):,} substitutions")

def scan_gmvp(genes, out):
    if out.exists():
        log(f"{out.name} exists, skipping")
        return
    log(f"scanning gMVP for {len(genes)} gene symbols")
    keep = []
    reader = pd.read_csv(GMVP, sep='\t', chunksize=2_000_000, low_memory=False)
    for i, ch in enumerate(reader, 1):
        cols = {c.lower(): c for c in ch.columns}
        gc = cols.get('genename') or cols.get('gene') or cols.get('gene_symbol')
        if gc is None:
            log(f"  gMVP columns unexpected: {list(ch.columns)[:12]}")
            return
        h = ch[ch[gc].astype(str).str.upper().isin(genes)]
        if len(h):
            keep.append(h)
        if i % 5 == 0:
            log(f"  chunk {i}, {sum(map(len, keep)):,} kept")
    if not keep:
        log("no gMVP rows matched")
        return
    d = slim(pd.concat(keep, ignore_index=True),
             ['gmvp'], ['genename', 'gene', 'gene_symbol'])
    d.to_parquet(out, index=False)
    log(f"wrote {out.name}: {len(d):,} rows, columns {list(d.columns)}")

def scan_varity(accs, out):
    if out.exists():
        log(f"{out.name} exists, skipping")
        return
    log(f"scanning VARITY for {len(accs)} accessions")
    keep = []
    with tarfile.open(VARITY, 'r:gz') as t:
        members = [m for m in t.getmembers() if m.isfile()]
        reader = pd.read_csv(t.extractfile(members[0]), sep='\t',
                             chunksize=2_000_000, low_memory=False)
        for i, ch in enumerate(reader, 1):
            cols = {c.lower(): c for c in ch.columns}
            ac = cols.get('p_vid') or cols.get('uniprot_id') or cols.get('uniprot_acc')
            if ac is None:
                log(f"  VARITY columns unexpected: {list(ch.columns)[:12]}")
                return
            h = ch[ch[ac].isin(accs)]
            if len(h):
                keep.append(h)
            if i % 5 == 0:
                log(f"  chunk {i}, {sum(map(len, keep)):,} kept")
    if not keep:
        log("no VARITY rows matched")
        return
    d = slim(pd.concat(keep, ignore_index=True),
             ['varity_r_loo', 'varity_r', 'varity'], ['p_vid', 'uniprot_acc'])
    d.to_parquet(out, index=False)
    log(f"wrote {out.name}: {len(d):,} rows, columns {list(d.columns)}")

def main():
    t = targets()
    t.to_csv(DATA / 'human_assay_targets.csv', index=False)
    accs = set(t.uniprot_acc.dropna())
    genes = set(t.gene_symbol.dropna())
    scan_alphamissense(accs, DATA / 'human_assay_scores_alphamissense.parquet')
    scan_varity(accs, DATA / 'human_assay_scores_varity.parquet')
    scan_gmvp(genes, DATA / 'human_assay_scores_gmvp.parquet')
    (DATA / 'human_assay_scores_metadata.json').write_text(json.dumps({
        'n_assays': int(len(t)), 'n_proteins': int(t.uniprot_acc.nunique()),
        'source': 'full predictor releases, not the ClinVar-restricted tables',
    }, indent=2))
    log("done")

if __name__ == '__main__':
    main()
