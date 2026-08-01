import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
OUT = BASE / "results" / "rebuild"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1022
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def load_sequences():
    u = pd.read_parquet(UNIPROT)
    return {a: s for a, s in zip(u.accession, u.sequence) if isinstance(s, str)}

def window(seq, pos1):
    n = len(seq)
    if n <= MAX_LEN:
        return seq, pos1 - 1
    half = MAX_LEN // 2
    start = max(0, min(pos1 - 1 - half, n - MAX_LEN))
    return seq[start:start + MAX_LEN], pos1 - 1 - start

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='mito', choices=['mito', 'control', 'all'])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--min-stars', type=int, default=2,
                    help='ClinVar review-status floor; 2 is the pre-declared '
                         'primary analysis set (see STUDY_DESIGN.md)')
    args = ap.parse_args()

    df = pd.read_parquet(DATA / f'variants_{args.set}.parquet')
    if args.min_stars:
        before = len(df)
        df = df[df.stars >= args.min_stars].reset_index(drop=True)
        log(f"review-status floor >={args.min_stars}*: {before:,} -> {len(df):,}")
    log(f"{args.set}: {len(df):,} variants, {df.uniprot_acc.nunique()} proteins")

    seqs = load_sequences()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    log(f"model on {device}")

    sites = df[['uniprot_acc', 'position_1']].drop_duplicates().values.tolist()

    def window_len(site):
        s = seqs.get(site[0])
        return 0 if s is None else min(len(s), MAX_LEN)

    sites.sort(key=window_len)
    log(f"{len(sites):,} unique masked sites to score (length-sorted batches)")

    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in 'ACDEFGHIKLMNPQRSTVWY'}
    logprobs = {}
    t0 = datetime.now()

    for i in range(0, len(sites), args.batch):
        chunk = sites[i:i + args.batch]
        subseqs, idxs, keys = [], [], []
        for acc, pos1 in chunk:
            s = seqs.get(acc)
            if s is None or not (1 <= pos1 <= len(s)):
                continue
            sub, k = window(s, pos1)
            subseqs.append(sub); idxs.append(k); keys.append((acc, pos1))
        if not subseqs:
            continue

        enc = tok(subseqs, return_tensors='pt', padding=True).to(device)
        for row, k in enumerate(idxs):
            enc['input_ids'][row, k + 1] = tok.mask_token_id

        with torch.no_grad():
            out = model(**enc).logits

        for row, (k, key) in enumerate(zip(idxs, keys)):
            lp = torch.log_softmax(out[row, k + 1], dim=-1)
            logprobs[key] = {a: float(lp[t]) for a, t in aa_ids.items()}

        if i and i % (args.batch * 25) == 0:
            done = i + len(chunk)
            rate = done / max((datetime.now() - t0).total_seconds(), 1e-9)
            eta = (len(sites) - done) / max(rate, 1e-9) / 60
            log(f"  {done:,}/{len(sites):,} sites  {rate:.1f}/s  ETA {eta:.0f} min")

    scores = []
    for acc, pos1, wt, mut in zip(df.uniprot_acc, df.position_1, df.wt_aa, df.mut_aa):
        lp = logprobs.get((acc, pos1))
        scores.append(np.nan if lp is None else lp[mut] - lp[wt])
    df['esm2_masked_marginal'] = scores
    df['esm2_score'] = -df['esm2_masked_marginal']

    cov = df.esm2_score.notna().mean()
    log(f"coverage {cov:.1%}")

    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = df.esm2_score.notna()
    res = {
        'set': args.set,
        'n': int(ok.sum()),
        'coverage': float(cov),
        'auroc': float(roc_auc_score(df.label[ok], df.esm2_score[ok])),
        'aupr': float(average_precision_score(df.label[ok], df.esm2_score[ok])),
    }
    for lo in (1, 2):
        m = ok & (df.stars >= lo)
        if m.sum() > 100 and df.label[m].nunique() > 1:
            res[f'auroc_{lo}star'] = float(roc_auc_score(df.label[m], df.esm2_score[m]))
            res[f'n_{lo}star'] = int(m.sum())
    log(json.dumps(res, indent=2))

    for c in ('submito', 'mitopathway', 'ReviewStatus', 'ClinicalSignificance'):
        if c in df.columns:
            df[c] = df[c].astype('string').fillna('')
    df.to_parquet(OUT / f'esm2_zeroshot_{args.set}.parquet', index=False)
    (OUT / f'esm2_zeroshot_{args.set}.json').write_text(json.dumps(res, indent=2))
    log(f"wrote {OUT}")

if __name__ == '__main__':
    main()
