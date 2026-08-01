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
AM = SCRATCH / "alphamissense_aa.tsv.gz"
CHUNK = 5_000_000

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def main():
    frames = []
    for name in ('mito', 'control', 'all'):
        p = DATA / f'variants_{name}.parquet'
        if p.exists():
            frames.append(pd.read_parquet(
                p, columns=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa']))
    if not frames:
        log("no variant tables; run 01_build_datasets.py first")
        return
    want = pd.concat(frames).drop_duplicates()
    want['variant'] = (want.wt_aa + want.position_1.astype(str) + want.mut_aa)
    keys = set(zip(want.uniprot_acc, want.variant))
    accs = set(want.uniprot_acc)
    log(f"looking for {len(keys):,} substitutions in {len(accs):,} proteins")

    hits, seen = [], 0
    reader = pd.read_csv(AM, sep='\t', comment='#', chunksize=CHUNK,
                         low_memory=False,
                         names=['uniprot_id', 'protein_variant',
                                'am_pathogenicity', 'am_class'],
                         header=0)
    for i, ch in enumerate(reader):
        seen += len(ch)
        ch = ch[ch.uniprot_id.isin(accs)]
        if len(ch):
            m = [(a, v) in keys for a, v in zip(ch.uniprot_id, ch.protein_variant)]
            ch = ch[m]
            if len(ch):
                hits.append(ch)
        log(f"  chunk {i + 1}: {seen:,} rows scanned, "
            f"{sum(len(h) for h in hits):,} matched")

    if not hits:
        log("no matches found")
        return
    am = pd.concat(hits, ignore_index=True).drop_duplicates(
        subset=['uniprot_id', 'protein_variant'])
    am = am.rename(columns={'uniprot_id': 'uniprot_acc',
                            'am_pathogenicity': 'alphamissense'})
    am.to_parquet(DATA / 'alphamissense_scores.parquet', index=False)

    cov = len(am) / len(keys)
    log(f"matched {len(am):,} of {len(keys):,} ({cov:.1%})")
    (DATA / 'alphamissense_metadata.json').write_text(json.dumps({
        'source': 'AlphaMissense_aa_substitutions.tsv.gz',
        'join': 'uniprot accession + protein substitution (no liftover)',
        'requested': int(len(keys)), 'matched': int(len(am)),
        'coverage': float(cov),
    }, indent=2))

if __name__ == '__main__':
    main()
