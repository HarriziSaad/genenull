import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results" / "rebuild"
PREREG = BASE / "submission" / "predictor_exposure_prereg.json"

MIN_PER_CLASS = 3
MIN_COVERAGE = 0.50

NEGATED = {"SIFT_score", "SIFT4G_score", "PROVEAN_score", "FATHMM_score",
           "ESM1b_score", "popEVE_score", "LRT_score"}
LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_REPEATS = 3

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def within_gene_mean(gene, label, score):
    ok = ~pd.isna(score)
    g, y, s = gene[ok], label[ok], score[ok]
    npos = pd.Series(y).groupby(g).transform("sum").values
    n = pd.Series(y).groupby(g).transform("size").values
    keep = (npos >= MIN_PER_CLASS) & ((n - npos) >= MIN_PER_CLASS)
    if keep.sum() == 0:
        return np.nan, 0
    g, y, s = g[keep], y[keep], s[keep]
    rank = pd.Series(s).groupby(g).rank(method="average").values
    pos_rank = pd.Series(np.where(y == 1, rank, 0.0)).groupby(g).sum()
    p = pd.Series(y).groupby(g).sum()
    t = pd.Series(y).groupby(g).size()
    auc = (pos_rank - p * (p + 1) / 2) / (p * (t - p))
    return float(auc.mean()), int(len(auc))

def main():
    prereg = json.loads(PREREG.read_text(encoding="utf8"))["predictors"]
    src = DATA / "dbnsfp_scores.parquet"
    if not src.exists():
        sys.exit(f"missing {src} - run the Colab extraction first")

    d = pd.read_parquet(DATA / "variants_all.parquet")
    x = pd.read_parquet(src)
    log(f"dbNSFP extract: {len(x):,} rows")

    score_cols = [c for c in x.columns
                  if c.endswith(("_score", "_raw", "_raw_coding"))
                  or c in prereg]
    score_cols = [c for c in score_cols if c in x.columns]

    unknown = [c for c in score_cols if c not in prereg]
    if unknown:
        sys.exit("STOP: predictors absent from the pre-registration:\n  "
                 + "\n  ".join(unknown)
                 + "\n\nClassify each in submission/predictor_exposure_prereg.json, "
                   "recording the basis and reference, then re-run. Do not "
                   "default them.")
    log(f"{len(score_cols)} score columns, all pre-registered")

    x = x.copy()
    x["_row"] = np.arange(len(x))
    x["_wt"] = x["aaref"].astype(str).str.split(";").str[0]
    x["_mut"] = x["aaalt"].astype(str).str.split(";").str[0]
    acc_l = x["Uniprot_acc"].astype(str).str.split(";")
    pos_l = x["aapos"].astype(str).str.split(";")
    ok = acc_l.str.len().eq(pos_l.str.len())
    ex = pd.DataFrame({
        "_row": np.repeat(x["_row"].values[ok.values], acc_l[ok].str.len().values),
        "uniprot_acc": np.concatenate(acc_l[ok].values),
        "position_1": np.concatenate(pos_l[ok].values)})
    ex["position_1"] = pd.to_numeric(ex["position_1"], errors="coerce")
    ex = ex.dropna(subset=["position_1"])
    ex["position_1"] = ex["position_1"].astype("int64")
    ex = ex.merge(x[["_row", "_wt", "_mut"]], on="_row", how="left")
    ex = ex.rename(columns={"_wt": "wt_aa", "_mut": "mut_aa"})

    key = ["uniprot_acc", "position_1", "wt_aa", "mut_aa"]
    ex = ex.drop_duplicates(key)
    sub = ex.merge(x[["_row"] + score_cols], on="_row", how="left")
    m = d.merge(sub[key + score_cols], on=key, how="left")
    assert len(m) == len(d), f"join changed row count: {len(d):,} -> {len(m):,}"
    matched = m[score_cols].notna().any(axis=1).sum()
    log(f"joined: {matched:,} of {len(d):,} variants carry at least one score "
        f"({matched/len(d):.1%})")

    gene = m.gene_symbol.values
    label = m.label.values.astype(np.int64)

    rows = []
    inverted = []
    for c in score_cols:
        s = pd.to_numeric(m[c], errors="coerce").values
        if c in NEGATED:
            s = -s
        cov = float(np.mean(~pd.isna(s)))
        if cov < MIN_COVERAGE:
            rows.append({"predictor": c, "coverage": cov, "used": False})
            continue
        ok = ~pd.isna(s)
        g = float(roc_auc_score(label[ok], s[ok]))
        w, n_eval = within_gene_mean(gene, label, s)
        e = prereg[c]
        rows.append({"predictor": c, "coverage": cov, "used": True,
                     "n_scored": int(ok.sum()), "n_genes_evaluable": n_eval,
                     "global": g, "within": w, "change": w - g,
                     "level": e["level"], "confidence": e["confidence"],
                     "family": e.get("family", c),
                     "representative": bool(e.get("representative", True)),
                     "excluded_reason": e.get("reason", "")})
        if g < 0.5:
            inverted.append((c, g))
        log(f"  {c:<30} cov {cov:5.1%}  global {g:.4f}  change {w - g:+.4f}  "
            f"level {e['level']}{'  [negated]' if c in NEGATED else ''}")

    if inverted:
        sys.exit("STOP: still inverted after applying documented polarity:\n  "
                 + "\n  ".join(f"{c}: global AUROC {g:.4f}" for c, g in inverted)
                 + "\n\nCheck each against the dbNSFP column documentation and add "
                   "it to NEGATED with a note in the pre-registration. Do not "
                   "flip on the basis of which direction scores better.")

    t = pd.DataFrame(rows)
    t.to_csv(RES / "dbnsfp_gradient.csv", index=False)

    use = t[t.used & t.level.isin([0, 1, 2])].copy()
    use["level"] = use["level"].astype(int)
    rep = use[use.representative]
    hi = rep[rep.confidence == "high"]

    out = {"n_predictors_total": len(t),
           "n_used": int(t.used.sum()),
           "n_family_representatives": int(len(rep)),
           "n_excluded_ensemble": int((t.excluded_reason == "ensemble").sum()),
           "n_below_coverage": int((~t.used).sum()),
           "min_coverage": MIN_COVERAGE}

    def corr(df, tag):
        if len(df) < 4:
            out[tag] = None
            return
        r = spearmanr(df.level, df.change)
        rc = spearmanr(df["global"], df.change)
        out[tag] = {"n": len(df),
                    "exposure_vs_change": [float(r.statistic), float(r.pvalue)],
                    "global_vs_change": [float(rc.statistic), float(rc.pvalue)]}
        log(f"\n{tag}: n={len(df)}  exposure rho={r.statistic:+.3f} "
            f"p={r.pvalue:.4g}  |  confound rho={rc.statistic:+.3f} "
            f"p={rc.pvalue:.4g}")

    corr(hi, "primary_representatives_high_confidence")
    corr(rep, "representatives_all_confidence")
    corr(use, "sensitivity_every_column")

    (RES / "dbnsfp_gradient.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'dbnsfp_gradient.csv'}")
    log(f"wrote {RES / 'dbnsfp_gradient.json'}")

if __name__ == "__main__":
    main()
