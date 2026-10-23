import importlib.util
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
NAMES = {"varity": "VARITY_R_LOO", "alphamissense": "AlphaMissense", "gmvp": "gMVP"}
COLS = list(NAMES)
KEY = ["uniprot_acc", "position_1", "wt_aa", "mut_aa"]
MIN_PER_CLASS = 3
N_BOOT = 2000


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def per_group_u(codes, y, s, G):
    r = pd.Series(s).groupby(codes).rank(method="average").values
    pos = np.bincount(codes, weights=np.where(y == 1, r, 0.0), minlength=G)
    p = np.bincount(codes, weights=y, minlength=G)
    t = np.bincount(codes, minlength=G)
    return pos - p * (p + 1) / 2, p * (t - p)


def compare(d, tag, seed=42, dump=None):
    cnt = d.groupby("uniprot_acc").label.agg(["sum", "size"])
    ev = cnt[(cnt["sum"] >= MIN_PER_CLASS) & (cnt["size"] - cnt["sum"] >= MIN_PER_CLASS)].index
    e = d[d.uniprot_acc.isin(ev)].reset_index(drop=True)
    codes, uniq = pd.factorize(e.uniprot_acc)
    G = len(uniq)
    y = e.label.values.astype(int)
    out = {"n_variants_all": int(len(d)), "n_proteins_all": int(d.uniprot_acc.nunique()),
           "n_proteins_evaluable": int(G), "n_variants_evaluable": int(len(e))}
    if G < 20:
        out["note"] = "fewer than 20 evaluable proteins; not analysed"
        return out
    U, PN = {}, None
    tab = {}
    for c in COLS:
        u, pn = per_group_u(codes, y, e[c].values, G)
        U[c], PN = u, pn
        tab[NAMES[c]] = {"global_unmatched": auc(d.label.values.astype(int), d[c].values),
                         "global_matched": auc(y, e[c].values),
                         "within_mean": float((u / pn).mean()),
                         "within_pooled": float(u.sum() / pn.sum())}
    out["auroc"] = tab
    for k in ("global_unmatched", "global_matched", "within_mean", "within_pooled"):
        out[f"ranking_{k}"] = [n for n, _ in sorted(tab.items(), key=lambda kv: -kv[1][k])]

    rng = np.random.default_rng(seed)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(codes, minlength=G))]
    rows_of = [order[starts[g]:starts[g + 1]] for g in range(G)]
    pairs = list(combinations(COLS, 2))
    boot = {(a, b, k): [] for a, b in pairs for k in ("global", "mean", "pooled")}
    for _ in range(N_BOOT):
        gs = rng.integers(0, G, G)
        idx = np.concatenate([rows_of[g] for g in gs])
        yy = y[idx]
        gl = {c: auc(yy, e[c].values[idx]) for c in COLS}
        mn = {c: (U[c][gs] / PN[gs]).mean() for c in COLS}
        pl = {c: U[c][gs].sum() / PN[gs].sum() for c in COLS}
        for a, b in pairs:
            boot[(a, b, "global")].append(gl[a] - gl[b])
            boot[(a, b, "mean")].append(mn[a] - mn[b])
            boot[(a, b, "pooled")].append(pl[a] - pl[b])
    if dump is not None:
        pd.DataFrame({f"{a}|{b}|{k}": v for (a, b, k), v in boot.items()}).to_csv(
            dump, index=False)
    key_obs = {"global": "global_matched", "mean": "within_mean", "pooled": "within_pooled"}
    comp = []
    for (a, b, k), v in boot.items():
        v = np.array(v)
        obs = tab[NAMES[a]][key_obs[k]] - tab[NAMES[b]][key_obs[k]]
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        comp.append({"comparison": f"{NAMES[a]} - {NAMES[b]}", "metric": k,
                     "diff": float(obs), "ci": [float(np.percentile(v, 2.5)),
                                                float(np.percentile(v, 97.5))],
                     "p": float(min(p, 1.0))})
    out["pairwise"] = comp

    log(f"\n=== {tag}: {len(e):,} variants in {G:,} evaluable proteins "
        f"(of {len(d):,} variants, {d.uniprot_acc.nunique():,} proteins)")
    for n, v in tab.items():
        log(f"  {n:<14} global(unmatched) {v['global_unmatched']:.4f}  global(matched) "
            f"{v['global_matched']:.4f}  within mean {v['within_mean']:.4f}  "
            f"within pooled {v['within_pooled']:.4f}")
    for k in ("global_unmatched", "global_matched", "within_mean", "within_pooled"):
        log(f"  ranking {k:<17}: {' > '.join(out[f'ranking_{k}'])}")
    for r in comp:
        log(f"  {r['comparison']:<30} {r['metric']:<7} {r['diff']:+.4f} "
            f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  p={r['p']:.4f}")
    return out


def main():
    spec = importlib.util.spec_from_file_location(
        "m19", Path(__file__).with_name("21_independent_replication.py"))
    m19 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m19)

    d = pd.read_parquet(DATA / "pgym_clinical_mapped.parquet")
    d = m19.attach_scores(d).dropna(subset=COLS)
    n_rows_submitted = int(len(d))
    n_dup_keys = int(d.loc[d.duplicated(KEY, keep=False), KEY].drop_duplicates().shape[0])
    d = d.drop_duplicates(KEY).reset_index(drop=True)
    log(f"ProteinGym clinical, all three scored: {len(d):,} variants, "
        f"{d.uniprot_acc.nunique():,} proteins (script 19 reported 49,260 / 2,046)")

    ours = pd.read_parquet(DATA / "variants_all.parquet")[KEY + ["label"]] \
        .drop_duplicates(KEY).rename(columns={"label": "label_ours"})
    j = d.merge(ours, on=KEY, how="left")
    inside = j.label_ours.notna()
    agree = (j.loc[inside, "label"] == j.loc[inside, "label_ours"]).mean()
    out = {"n_variants": int(len(d)), "n_proteins": int(d.uniprot_acc.nunique()),
           "n_rows_as_submitted": n_rows_submitted,
           "n_substitutions_counted_more_than_once_as_submitted": n_dup_keys,
           "fraction_in_our_clinvar_table": float(inside.mean()),
           "label_agreement_on_overlap": float(agree),
           "fraction_of_proteins_with_any_overlap":
               float(j[inside].uniprot_acc.nunique() / d.uniprot_acc.nunique())}
    log(f"in our ClinVar table: {inside.mean():.1%} of variants; labels agree on "
        f"{agree:.1%} of those")

    out["all_variants"] = compare(d, "ALL ProteinGym clinical variants",
                                  dump=RES / "replication_bootstrap.csv.gz")
    out["not_in_our_table"] = compare(j[~inside].drop(columns="label_ours")
                                      .reset_index(drop=True),
                                      "ProteinGym variants NOT in our ClinVar table")
    (RES / "replication_matched.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'replication_matched.json'}")


if __name__ == "__main__":
    main()
