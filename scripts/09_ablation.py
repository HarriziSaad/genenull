import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
RES.mkdir(parents=True, exist_ok=True)

WINDOW = 15
N_SPLITS = 10

KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
      'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
      'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
TOP_IDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318,
           'E': 0.736, 'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326,
           'K': 0.586, 'M': -0.397, 'F': -0.697, 'P': 0.987, 'S': 0.341,
           'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121}
CHARGE = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
VOL = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8,
       'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6,
       'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8,
       'Y': 193.6, 'V': 140.0}
AROMATIC = set('FWY')

GROUPS = {
    'Substitution': ['d_hydro', 'd_charge', 'd_volume', 'd_disorder',
                     'd_aromatic', 'abs_d_hydro', 'abs_d_charge',
                     'abs_d_volume', 'abs_d_disorder'],
    'Local context': ['loc_hydro', 'loc_charge', 'loc_disorder', 'loc_aromatic',
                      'loc_proline', 'loc_glycine', 'loc_cysteine',
                      'loc_entropy'],
    'Cysteine / redox': ['cys_gained', 'cys_lost', 'nearby_cys',
                         'ros_vulnerability'],
    'Structural perturbation': ['pro_introduced', 'gly_introduced',
                                'charge_inversion'],
    'Composite mechanistic': ['idp_disruption', 'import_disruption'],
    'Position (relative)': ['pos_normalized', 'is_n_terminal'],
}
ALL_FEATS = [f for g in GROUPS.values() for f in g]

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def entropy(s):
    if not s:
        return 0.0
    _, c = np.unique(list(s), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def featurise(df, seqs):
    rows = []
    for acc, pos1, wt, mut in zip(df.uniprot_acc, df.position_1,
                                  df.wt_aa, df.mut_aa):
        s = seqs.get(acc, '')
        n = len(s)
        i = pos1 - 1
        lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
        win = s[lo:hi]
        wl = max(len(win), 1)

        d_hydro = KD.get(mut, 0) - KD.get(wt, 0)
        d_charge = CHARGE.get(mut, 0) - CHARGE.get(wt, 0)
        d_vol = VOL.get(mut, 0) - VOL.get(wt, 0)
        d_dis = TOP_IDP.get(mut, 0) - TOP_IDP.get(wt, 0)
        d_aro = float(mut in AROMATIC) - float(wt in AROMATIC)
        loc_dis = float(np.mean([TOP_IDP.get(c, 0) for c in win])) if win else 0.0
        is_nterm = float(pos1 <= 50)

        rows.append({
            'd_hydro': d_hydro, 'd_charge': d_charge, 'd_volume': d_vol,
            'd_disorder': d_dis, 'd_aromatic': d_aro,
            'abs_d_hydro': abs(d_hydro), 'abs_d_charge': abs(d_charge),
            'abs_d_volume': abs(d_vol), 'abs_d_disorder': abs(d_dis),
            'loc_hydro': float(np.mean([KD.get(c, 0) for c in win])) if win else 0.0,
            'loc_charge': float(np.mean([CHARGE.get(c, 0) for c in win])) if win else 0.0,
            'loc_disorder': loc_dis,
            'loc_aromatic': sum(c in AROMATIC for c in win) / wl,
            'loc_proline': win.count('P') / wl,
            'loc_glycine': win.count('G') / wl,
            'loc_cysteine': win.count('C') / wl,
            'loc_entropy': entropy(win),
            'cys_gained': float(mut == 'C'), 'cys_lost': float(wt == 'C'),
            'nearby_cys': float(win.count('C')),
            'ros_vulnerability': (win.count('C') / wl) * max(loc_dis, 0),
            'pro_introduced': float(mut == 'P'), 'gly_introduced': float(mut == 'G'),
            'charge_inversion': float(CHARGE.get(wt, 0) * CHARGE.get(mut, 0) < 0),
            'idp_disruption': abs(d_dis) * (1 + max(loc_dis, 0)),
            'import_disruption': is_nterm * abs(d_charge),
            'pos_normalized': pos1 / max(n, 1),
            'is_n_terminal': is_nterm,
        })
    return pd.DataFrame(rows, index=df.index)

def cv_auc(X, y, groups, seed=42):
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2 or X.shape[1] == 0:
        return np.nan
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=0.1,
                               random_state=seed).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    ok = ~np.isnan(pred)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], pred[ok]))

def main():
    df = pd.read_parquet(DATA / 'variants_mito.parquet')
    cl = pd.read_parquet(DATA / 'protein_clusters.parquet')[['uniprot_acc', 'cluster']]
    df = df.merge(cl, on='uniprot_acc', how='left')
    df['cluster'] = df.cluster.fillna(-1).astype(int)
    df = df[df.stars >= 2].reset_index(drop=True)
    log(f"mitochondrial, >=2 star: {len(df):,} variants, "
        f"{df.uniprot_acc.nunique()} proteins, {df.cluster.nunique()} clusters")

    u = pd.read_parquet(UNIPROT)
    seqs = {a: s for a, s in zip(u.accession, u.sequence) if isinstance(s, str)}

    log("computing interpretable features from sequence")
    F = featurise(df, seqs)
    F = F.replace([np.inf, -np.inf], 0).fillna(0)
    y = df.label.values
    groups = df.cluster.values

    base = cv_auc(F[ALL_FEATS].values, y, groups)
    log(f"full interpretable feature set: AUROC {base:.4f}")

    rows = [{'ablation': 'Full interpretable set', 'n_features': len(ALL_FEATS),
             'auroc': base, 'delta': 0.0}]
    for name, feats in GROUPS.items():
        keep = [f for f in ALL_FEATS if f not in feats]
        auc = cv_auc(F[keep].values, y, groups)
        rows.append({'ablation': f'minus {name}', 'n_features': len(keep),
                     'auroc': auc, 'delta': auc - base})
        log(f"  minus {name:<26} AUROC {auc:.4f}  delta {auc - base:+.4f}")

    for name, feats in GROUPS.items():
        use = [f for f in feats if f in F.columns]
        auc = cv_auc(F[use].values, y, groups)
        rows.append({'ablation': f'{name} alone', 'n_features': len(use),
                     'auroc': auc, 'delta': auc - base})
        log(f"  {name + ' alone':<32} AUROC {auc:.4f}")

    amf = DATA / 'alphamissense_scores.parquet'
    am_auc = np.nan
    if amf.exists():
        am = pd.read_parquet(amf)
        am['position_1'] = am.protein_variant.str[1:-1].astype(int)
        am['wt_aa'] = am.protein_variant.str[0]
        am['mut_aa'] = am.protein_variant.str[-1]
        j = df.merge(am[['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa',
                         'alphamissense']],
                     on=['uniprot_acc', 'position_1', 'wt_aa', 'mut_aa'],
                     how='left')
        ok = j.alphamissense.notna()
        if ok.sum() > 100:
            am_auc = float(roc_auc_score(j.label[ok], j.alphamissense[ok]))
            log(f"AlphaMissense on the same variants: AUROC {am_auc:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RES / 'ablation.csv', index=False)
    (RES / 'ablation.json').write_text(json.dumps({
        'question': 'Do mitochondria-specific interpretable features contribute '
                    'predictive signal on a correctly constructed dataset?',
        'set': 'MitoCarta 3.0, ClinVar review status >=2 stars',
        'split': 'leave-cluster-out, MMseqs2 30% identity',
        'n_variants': int(len(df)),
        'full_set_auroc': base,
        'alphamissense_auroc': am_auc,
        'note': 'Length and absolute-position features are excluded by design; '
                'on the contaminated dataset protein_length alone reached '
                'AUROC 0.672 through gene-size confounding.',
    }, indent=2, default=float))
    log(f"wrote {RES / 'ablation.csv'}")

if __name__ == '__main__':
    main()
