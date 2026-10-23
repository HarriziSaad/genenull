import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
DATA = BASE / "data" / "rebuild"
PRED = ["alphamissense", "varity", "gmvp", "esm2_score"]
NULLS = ["gene_prior_insample", "gene_prior_randomsplit", "gene_prior_holdout"]


def auc(y, s):
    ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])), int(ok.sum())


def main():
    ev = json.loads((RES / "evaluation_report.json").read_text().replace("NaN", "null"))
    t = pd.read_parquet(RES / "evaluation_table.parquet")
    esm_n = int(t.esm2_score.notna().sum())
    assert (t.loc[t.esm2_score.notna(), "stars"] >= 2).all(), "ESM-2 outside >=2 star"
    out = {"n_total": int(len(t)), "n_clusters": int(t.cluster.nunique()),
           "esm2_scored_variants": esm_n, "sets": {}, "mito_minus_control": {},
           "minus_insample_null": {}}
    for tag, floor in (("all", 0), ("2star", 2)):
        sub = t[t.stars >= floor]
        out["sets"][tag] = {}
        for s in ("mito", "control"):
            d = sub[sub.set == s]
            y = d.label.values
            row = {"n": int(len(d)), "n_genes": int(d.gene_symbol.nunique()),
                   "n_clusters": int(d.cluster.nunique()),
                   "pathogenic_fraction": float(y.mean())}
            for p in PRED + NULLS:
                if p == "esm2_score" and tag == "all":
                    continue
                a, n = auc(y, d[p].values.astype(float))
                row[p] = {"auroc": a, "n": n,
                          "ci": ev["by_set"][f"{floor}star"][s][p]["ci"]}
            out["sets"][tag][s] = row
        out["mito_minus_control"][tag] = {}
        for p in PRED + NULLS[:2]:
            if p == "esm2_score" and tag == "all":
                continue
            b = ev["mito_vs_control"][f"{floor}star"][p]
            out["mito_minus_control"][tag][p] = {
                "difference": out["sets"][tag]["mito"][p]["auroc"]
                - out["sets"][tag]["control"][p]["auroc"],
                "ci": b["ci"], "p": b["p"], "bootstrap_mean": b["delta_auroc"]}
        m = sub[sub.set == "mito"]
        out["minus_insample_null"][tag] = {}
        for p in PRED:
            if p == "esm2_score" and tag == "all":
                continue
            ok = m[p].notna().values
            y = m.label.values[ok]
            a1 = roc_auc_score(y, m[p].values[ok])
            a0 = roc_auc_score(y, m.gene_prior_insample.values[ok])
            b = ev["vs_gene_prior"][f"{floor}star"][p]
            out["minus_insample_null"][tag][p] = {
                "difference": float(a1 - a0), "predictor": float(a1), "null": float(a0),
                "n": int(ok.sum()), "ci": b["ci"], "p": b["p"],
                "bootstrap_mean": b["delta_auroc"]}

    for tag, v in out["mito_minus_control"].items():
        for p, r in v.items():
            print(f"{tag:<6} mito - control {p:<24} {r['difference']:+.4f} "
                  f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] p={r['p']:.3f} "
                  f"(bootstrap mean {r['bootstrap_mean']:+.4f})")
    for tag, v in out["minus_insample_null"].items():
        for p, r in v.items():
            print(f"{tag:<6} minus null     {p:<24} {r['difference']:+.4f} "
                  f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] p={r['p']:.3f} n={r['n']}")
    (RES / "mito_case_study.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {RES / 'mito_case_study.json'}")


if __name__ == "__main__":
    main()
