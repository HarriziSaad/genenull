import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
API = "https://www.ebi.ac.uk/proteins/api/variation?accession={}&size=-1"
BATCH = 40
PAUSE = 0.35

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def wanted_accessions():
    accs = set()
    for name in ('within_gene_ranking_modern.csv', 'within_gene_ranking_all.csv'):
        p = RES / name
        if not p.exists():
            continue
        w = pd.read_csv(p)
        d = pd.read_parquet(DATA / 'variants_all.parquet',
                            columns=['gene_symbol', 'uniprot_acc'])
        g2a = dict(zip(d.gene_symbol, d.uniprot_acc))
        accs |= {g2a[g] for g in w.gene_symbol if g in g2a}
    log(f"{len(accs):,} accessions to query")
    return sorted(accs)

def fetch(batch):
    url = API.format(','.join(batch))
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == 3:
                log(f"  giving up on a batch after 4 attempts: {type(e).__name__}")
                return []
            time.sleep(2 ** attempt)
    return []

def main():
    accs = wanted_accessions()
    if not accs:
        log("no accessions; run 16_within_gene_ranking.py first")
        return

    out = DATA / 'polyphen_sift_scores.parquet'
    if out.exists():
        log(f"{out.name} already exists; delete it to refetch")
        return

    rows, done = [], 0
    for i in range(0, len(accs), BATCH):
        chunk = accs[i:i + BATCH]
        for entry in fetch(chunk):
            acc = entry.get('accession')
            for f in entry.get('features', []):
                wt, mut, pos = (f.get('wildType'), f.get('mutatedType'),
                                f.get('begin'))
                preds = f.get('predictions') or []
                if not (wt and mut and pos and preds):
                    continue
                pp = sift = None
                for p in preds:
                    algo = (p.get('predAlgorithmNameType') or '').lower()
                    if 'polyphen' in algo:
                        pp = p.get('score')
                    elif 'sift' in algo:
                        sift = p.get('score')
                if pp is None and sift is None:
                    continue
                try:
                    pos = int(pos)
                except (TypeError, ValueError):
                    continue
                rows.append({'uniprot_acc': acc, 'position_1': pos,
                             'wt_aa': wt, 'mut_aa': mut,
                             'polyphen2': pp,
                             'sift': None if sift is None else -float(sift)})
        done += len(chunk)
        if (i // BATCH) % 5 == 0:
            log(f"  {done:,}/{len(accs):,} accessions, {len(rows):,} scores")
        time.sleep(PAUSE)

    if not rows:
        log("no scores returned")
        return
    df = pd.DataFrame(rows).drop_duplicates(
        subset=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'])
    df.to_parquet(out, index=False)
    log(f"PolyPhen-2: {df.polyphen2.notna().sum():,} scores; "
        f"SIFT: {df.sift.notna().sum():,} scores "
        f"across {df.uniprot_acc.nunique():,} proteins")
    (DATA / 'polyphen_sift_metadata.json').write_text(json.dumps({
        'source': 'EBI Proteins API /variation',
        'n_proteins': int(df.uniprot_acc.nunique()),
        'n_rows': int(len(df)),
        'orientation': 'polyphen2 higher = more damaging; sift NEGATED so that '
                       'higher = more damaging, matching every other predictor',
    }, indent=2))

if __name__ == '__main__':
    main()
