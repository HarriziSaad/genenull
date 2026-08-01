import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
RES.mkdir(parents=True, exist_ok=True)

N_BINS = 10
MIN_VARIANTS_PER_PROTEIN = 10
AMBIGUOUS = (0.4, 0.6)
ALPHA = 0.1

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def ece_mce(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = mce = 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        w = m.mean()
        gap = abs(acc - conf)
        ece += w * gap
        mce = max(mce, gap)
        rows.append({'bin': b, 'lo': edges[b], 'hi': edges[b + 1],
                     'n': int(m.sum()), 'mean_score': conf,
                     'observed_rate': acc, 'gap': gap})
    return float(ece), float(mce), rows

def temperature_scale(y_tr, p_tr, p_te):
    eps = 1e-6
    lt = np.log(np.clip(p_tr, eps, 1 - eps) / (1 - np.clip(p_tr, eps, 1 - eps)))
    le = np.log(np.clip(p_te, eps, 1 - eps) / (1 - np.clip(p_te, eps, 1 - eps)))
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.2, 5.0, 97):
        q = 1 / (1 + np.exp(-lt / t))
        q = np.clip(q, eps, 1 - eps)
        nll = -np.mean(y_tr * np.log(q) + (1 - y_tr) * np.log(1 - q))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return 1 / (1 + np.exp(-le / best_t)), float(best_t)

def conformal_coverage(y, p, groups, alpha=ALPHA):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return None
    covered, sizes, ambiguous, empty = [], [], 0, 0
    for tr, te in GroupKFold(n_splits).split(p, y, groups):
        s_tr = np.where(y[tr] == 1, 1 - p[tr], p[tr])
        k = int(np.ceil((len(s_tr) + 1) * (1 - alpha)))
        k = min(max(k, 1), len(s_tr))
        qhat = np.sort(s_tr)[k - 1]
        in_pos = (1 - p[te]) <= qhat
        in_neg = p[te] <= qhat
        size = in_pos.astype(int) + in_neg.astype(int)
        truth = np.where(y[te] == 1, in_pos, in_neg)
        covered.append(truth)
        sizes.append(size)
        ambiguous += int((size == 2).sum())
        empty += int((size == 0).sum())
    covered = np.concatenate(covered)
    sizes = np.concatenate(sizes)
    return {
        'target_coverage': 1 - alpha,
        'empirical_coverage': float(covered.mean()),
        'mean_set_size': float(sizes.mean()),
        'frac_both_labels': float(ambiguous / len(sizes)),
        'frac_empty': float(empty / len(sizes)),
    }

def main():
    ev = RES / 'evaluation_table.parquet'
    if not ev.exists():
        log("run 04_evaluate.py first")
        return
    df = pd.read_parquet(ev)
    preds = [c for c in ('alphamissense', 'varity', 'gmvp', 'esm2_score')
             if c in df.columns and df[c].notna().any()]
    log(f"predictors: {preds}")

    cal_rows, summary = [], {}
    for c in preds:
        d = df[df[c].notna()]
        y = d.label.values.astype(float)
        p = d[c].values.astype(float)
        if p.min() < 0 or p.max() > 1:
            p = 1 / (1 + np.exp(-p))
            note = 'sigmoid-mapped from log-ratio before calibration'
        else:
            note = 'native [0,1] score'

        ece, mce, rows = ece_mce(y, p)
        brier = float(brier_score_loss(y, p))

        groups = d.cluster.values
        n_splits = min(5, len(np.unique(groups)))
        p_cal = np.full(len(p), np.nan)
        temps = []
        if n_splits >= 2:
            for tr, te in GroupKFold(n_splits).split(p, y, groups):
                p_cal[te], t = temperature_scale(y[tr], p[tr], p[te])
                temps.append(t)
        ok = ~np.isnan(p_cal)
        ece_c, mce_c, _ = ece_mce(y[ok], p_cal[ok]) if ok.any() else (np.nan,) * 3
        brier_c = float(brier_score_loss(y[ok], p_cal[ok])) if ok.any() else np.nan

        conf = conformal_coverage(y, p, groups)
        summary[c] = {
            'n': int(len(d)), 'score_note': note,
            'auroc': float(roc_auc_score(y, p)),
            'ece': ece, 'mce': mce, 'brier': brier,
            'ece_after_temperature': ece_c,
            'mce_after_temperature': mce_c,
            'brier_after_temperature': brier_c,
            'mean_temperature': float(np.mean(temps)) if temps else None,
            'conformal': conf,
        }
        for r in rows:
            r['predictor'] = c
            cal_rows.append(r)
        log(f"  {c:<16} ECE={ece:.4f} -> {ece_c:.4f} after scaling, "
            f"Brier={brier:.4f} -> {brier_c:.4f}"
            + (f", conformal coverage {conf['empirical_coverage']:.3f} "
               f"(target {conf['target_coverage']:.2f}), "
               f"{conf['frac_both_labels']:.1%} uncallable" if conf else ""))

    pd.DataFrame(cal_rows).to_csv(RES / 'calibration.csv', index=False)

    log("protein-level uncertainty")
    prot_rows = []
    for c in preds:
        d = df[df[c].notna()]
        for acc, g in d.groupby('uniprot_acc'):
            if len(g) < MIN_VARIANTS_PER_PROTEIN or g.label.nunique() < 2:
                continue
            p = g[c].values.astype(float)
            if p.min() < 0 or p.max() > 1:
                p = 1 / (1 + np.exp(-p))
            ben, pat = p[g.label.values == 0], p[g.label.values == 1]
            pooled = np.sqrt((ben.var() + pat.var()) / 2) or np.nan
            prot_rows.append({
                'predictor': c, 'uniprot_acc': acc,
                'gene_symbol': g.gene_symbol.iloc[0],
                'set': g.set.iloc[0] if 'set' in g.columns else '',
                'n_variants': int(len(g)),
                'pathogenic_fraction': float(g.label.mean()),
                'auroc': float(roc_auc_score(g.label, p)),
                'separation': float((pat.mean() - ben.mean()) / pooled)
                if pooled and not np.isnan(pooled) else np.nan,
                'frac_ambiguous': float(((p > AMBIGUOUS[0]) &
                                         (p < AMBIGUOUS[1])).mean()),
            })
    pu = pd.DataFrame(prot_rows)
    pu.to_csv(RES / 'protein_uncertainty.csv', index=False)

    for c in preds:
        s = pu[pu.predictor == c]
        if not len(s):
            continue
        worst = s.nsmallest(5, 'auroc')[['gene_symbol', 'n_variants', 'auroc']]
        summary[c]['protein_level'] = {
            'n_proteins': int(len(s)),
            'median_auroc': float(s.auroc.median()),
            'frac_proteins_below_0.7': float((s.auroc < 0.7).mean()),
            'median_frac_ambiguous': float(s.frac_ambiguous.median()),
            'worst_proteins': worst.to_dict('records'),
        }
        log(f"  {c:<16} {len(s)} proteins with >={MIN_VARIANTS_PER_PROTEIN} "
            f"variants, median per-protein AUROC {s.auroc.median():.3f}, "
            f"{(s.auroc < 0.7).mean():.1%} below 0.70")

    (RES / 'calibration_summary.json').write_text(json.dumps(summary, indent=2))
    log(f"wrote {RES / 'calibration_summary.json'}")

if __name__ == '__main__':
    main()
