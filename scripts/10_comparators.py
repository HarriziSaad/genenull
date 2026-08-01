import gzip
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
GMVP = SCRATCH / "gmvp.csv.gz"
VARITY = SCRATCH / "varity.tar.gz"
CHUNK = 2_000_000

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def wanted():
    frames = []
    for name in ('mito', 'control', 'all'):
        p = DATA / f'variants_{name}.parquet'
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=['gene_symbol', 'uniprot_acc', 'position_1',
                            'wt_aa', 'mut_aa']))
    w = pd.concat(frames).drop_duplicates(
        subset=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'])
    log(f"{len(w):,} unique substitutions to look up")
    return w

def run_gmvp(w):
    if not GMVP.exists():
        log(f"gMVP file missing at {GMVP}")
        return
    keys = set(zip(w.gene_symbol, w.position_1, w.wt_aa, w.mut_aa))
    pos_keys = set(zip(w.gene_symbol, w.position_1, w.mut_aa))
    genes = set(w.gene_symbol)

    hits, mismatch, scanned = [], 0, 0
    reader = pd.read_csv(
        GMVP, sep='\t', chunksize=CHUNK, low_memory=False,
        usecols=['gene_symbol', 'protein_position', 'ref_aa', 'alt_aa', 'gMVP'])
    for i, ch in enumerate(reader):
        scanned += len(ch)
        ch = ch[ch.gene_symbol.isin(genes)]
        if len(ch):
            ch = ch.dropna(subset=['protein_position'])
            ch['protein_position'] = pd.to_numeric(
                ch.protein_position, errors='coerce').astype('Int64')
            ch = ch.dropna(subset=['protein_position'])
            k = list(zip(ch.gene_symbol, ch.protein_position.astype(int),
                         ch.ref_aa, ch.alt_aa))
            keep = [x in keys for x in k]
            pk = list(zip(ch.gene_symbol, ch.protein_position.astype(int),
                          ch.alt_aa))
            mismatch += sum(1 for a, b in zip(keep, pk) if (not a) and b in pos_keys)
            ch = ch[keep]
            if len(ch):
                hits.append(ch)
        if i % 10 == 0:
            log(f"  gMVP chunk {i + 1}: {scanned:,} rows, "
                f"{sum(len(h) for h in hits):,} matched")

    if not hits:
        log("gMVP: no matches")
        return
    g = pd.concat(hits, ignore_index=True)
    g = g.rename(columns={'protein_position': 'position_1', 'ref_aa': 'wt_aa',
                          'alt_aa': 'mut_aa', 'gMVP': 'gmvp'})
    g['position_1'] = g.position_1.astype(int)
    g = g.drop_duplicates(subset=['gene_symbol', 'position_1', 'wt_aa', 'mut_aa'])
    g[['gene_symbol', 'position_1', 'wt_aa', 'mut_aa', 'gmvp']].to_parquet(
        DATA / 'gmvp_scores.parquet', index=False)
    cov = len(g) / len(keys)
    log(f"gMVP: matched {len(g):,} of {len(keys):,} ({cov:.1%}); "
        f"{mismatch:,} dropped on reference-residue mismatch")
    return {'matched': int(len(g)), 'requested': int(len(keys)),
            'coverage': float(cov), 'ref_mismatch_dropped': int(mismatch)}

def run_varity(w):
    if not VARITY.exists():
        log(f"VARITY file missing at {VARITY}")
        return
    import tarfile
    try:
        t = tarfile.open(VARITY)
        members = [m for m in t.getmembers() if m.isfile()]
    except Exception as e:
        log(f"VARITY archive unreadable ({type(e).__name__}); "
            f"download may be incomplete")
        return
    if not members:
        return
    m = members[0]
    log(f"VARITY: streaming {m.name} ({m.size / 1e9:.1f} GB uncompressed)")

    keys = set(zip(w.uniprot_acc, w.position_1, w.wt_aa, w.mut_aa))
    accs = set(w.uniprot_acc)

    hits, scanned = [], 0
    reader = pd.read_csv(
        t.extractfile(m), sep='\t', chunksize=CHUNK, low_memory=False,
        usecols=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt',
                 'VARITY_R', 'VARITY_ER', 'VARITY_R_LOO', 'VARITY_ER_LOO'])
    for i, ch in enumerate(reader):
        scanned += len(ch)
        ch = ch[ch.p_vid.isin(accs)]
        if len(ch):
            ch = ch.dropna(subset=['aa_pos'])
            ch['aa_pos'] = pd.to_numeric(ch.aa_pos, errors='coerce').astype('Int64')
            ch = ch.dropna(subset=['aa_pos'])
            k = list(zip(ch.p_vid, ch.aa_pos.astype(int), ch.aa_ref, ch.aa_alt))
            ch = ch[[x in keys for x in k]]
            if len(ch):
                hits.append(ch)
        if i % 5 == 0:
            log(f"  VARITY chunk {i + 1}: {scanned:,} rows, "
                f"{sum(len(h) for h in hits):,} matched")

    if not hits:
        log("VARITY: no matches")
        return
    v = pd.concat(hits, ignore_index=True).rename(columns={
        'p_vid': 'uniprot_acc', 'aa_pos': 'position_1', 'aa_ref': 'wt_aa',
        'aa_alt': 'mut_aa', 'VARITY_R_LOO': 'varity',
        'VARITY_R': 'varity_r_full', 'VARITY_ER_LOO': 'varity_er_loo',
        'VARITY_ER': 'varity_er_full'})
    v['position_1'] = v.position_1.astype(int)
    v = v.drop_duplicates(subset=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'])
    v.to_parquet(DATA / 'varity_scores.parquet', index=False)
    cov = len(v) / len(keys)
    log(f"VARITY: matched {len(v):,} of {len(keys):,} ({cov:.1%}); "
        f"primary column VARITY_R_LOO (leave-one-out, non-circular)")
    return {'matched': int(len(v)), 'requested': int(len(keys)),
            'coverage': float(cov),
            'primary_column': 'VARITY_R_LOO',
            'why': 'VARITY is trained on ClinVar; the leave-one-out score is '
                   'the only non-circular choice for a ClinVar benchmark'}

def main():
    w = wanted()
    report = {}
    if not (DATA / 'gmvp_scores.parquet').exists():
        r = run_gmvp(w)
        if r:
            report['gmvp'] = r
    else:
        log('gMVP already joined; skipping')
    r = run_varity(w)
    if r:
        report['varity'] = r
    if report:
        (DATA / 'comparator_metadata.json').write_text(json.dumps(report, indent=2))
        log(f"wrote {DATA / 'comparator_metadata.json'}")

if __name__ == '__main__':
    main()
