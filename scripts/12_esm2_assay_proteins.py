import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1022
AAS = 'ACDEFGHIKLMNPQRSTVWY'

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def window(seq, pos1):
    n = len(seq)
    if n <= MAX_LEN:
        return seq, pos1 - 1
    half = MAX_LEN // 2
    start = max(0, min(pos1 - 1 - half, n - MAX_LEN))
    return seq[start:start + MAX_LEN], pos1 - 1 - start

def main():
    f = RES / 'proteingym_mito_assays.csv'
    if not f.exists():
        log("run 08_proteingym.py --list first")
        return
    genes = sorted({str(d).split('_')[0].upper()
                    for d in pd.read_csv(f).DMS_id})

    u = pd.read_parquet(UNIPROT)
    g2a, seqs = {}, {}
    for acc, gn, s in zip(u.accession, u.gene_names, u.sequence):
        if isinstance(s, str):
            seqs[acc] = s
            if isinstance(gn, str):
                for g in gn.split():
                    g2a.setdefault(g.upper(), acc)
    targets = {g: g2a[g] for g in genes if g in g2a}
    log(f"assay proteins: {targets}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    log(f"model on {device}")
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in AAS}

    rows = []
    for gene, acc in targets.items():
        seq = seqs[acc]
        log(f"{gene} ({acc}): {len(seq)} residues")
        batch, meta = [], []
        for pos1 in range(1, len(seq) + 1):
            sub, k = window(seq, pos1)
            batch.append(sub); meta.append((pos1, k, seq[pos1 - 1]))
            if len(batch) == 8 or pos1 == len(seq):
                enc = tok(batch, return_tensors='pt', padding=True).to(device)
                for r, (_, k2, _) in enumerate(meta):
                    enc['input_ids'][r, k2 + 1] = tok.mask_token_id
                with torch.no_grad():
                    out = model(**enc).logits
                for r, (p1, k2, wt) in enumerate(meta):
                    lp = torch.log_softmax(out[r, k2 + 1], dim=-1)
                    lwt = float(lp[aa_ids[wt]]) if wt in aa_ids else np.nan
                    for mut in AAS:
                        if mut == wt:
                            continue
                        rows.append({
                            'gene_symbol': gene, 'uniprot_acc': acc,
                            'position_1': p1, 'wt_aa': wt, 'mut_aa': mut,
                            'esm2_score': -(float(lp[aa_ids[mut]]) - lwt),
                        })
                batch, meta = [], []
        log(f"  {gene}: {sum(1 for r in rows if r['gene_symbol'] == gene):,} substitutions")

    out = pd.DataFrame(rows)
    out.to_parquet(RES / 'esm2_assay_proteins.parquet', index=False)
    (RES / 'esm2_assay_proteins.json').write_text(json.dumps({
        'proteins': targets,
        'n_substitutions': int(len(out)),
        'scoring': 'masked marginal, log p(mut) - log p(wt), sign flipped so '
                   'higher means more damaging',
    }, indent=2))
    log(f"wrote {RES / 'esm2_assay_proteins.parquet'} ({len(out):,} rows)")

if __name__ == '__main__':
    main()
