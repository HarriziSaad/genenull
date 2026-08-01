import json
import sys
from datetime import datetime
from itertools import permutations
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
DEG_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_DEG_REPEATS = 3
N_BOOT = 500

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def within_gene_mean(gene, label, score, min_class=MIN_PER_CLASS):
    ok = ~np.isnan(score)
    g, y, s = gene[ok], label[ok], score[ok]
    npos = pd.Series(y).groupby(g).transform("sum").values
    n = pd.Series(y).groupby(g).transform("size").values
    keep = (npos >= min_class) & ((n - npos) >= min_class)
    if keep.sum() == 0:
        return np.nan, 0, None
    g, y, s = g[keep], y[keep], s[keep]
    rank = pd.Series(s).groupby(g).rank(method="average").values
    pos_rank = pd.Series(np.where(y == 1, rank, 0.0)).groupby(g).sum()
    p = pd.Series(y).groupby(g).sum()
    t = pd.Series(y).groupby(g).size()
    auc = (pos_rank - p * (p + 1) / 2) / (p * (t - p))
    return float(auc.mean()), int(len(auc)), auc

def exact_or_sampled_perm_p(x, y, observed, rng, max_exact=9):
    n = len(x)
    if n <= max_exact:
        stats = [abs(spearmanr(x, list(p)).statistic) for p in permutations(y)]
        return float(np.mean(np.array(stats) >= abs(observed) - 1e-12)), len(stats)
    stats = []
    y = np.asarray(y)
    for _ in range(20000):
        stats.append(abs(spearmanr(x, rng.permutation(y)).statistic))
    return float(np.mean(np.array(stats) >= abs(observed) - 1e-12)), len(stats)

def main():
    rng = np.random.default_rng(42)
    prereg = json.loads(PREREG.read_text(encoding="utf8"))["predictors"]
    src = DATA / "dbnsfp_scores.parquet"
    if not src.exists():
        sys.exit(f"missing {src}")

    d = pd.read_parquet(DATA / "variants_all.parquet")
    x = pd.read_parquet(src)

    score_cols = [c for c in x.columns if c.endswith("_score")]
    unknown = [c for c in score_cols if c not in prereg]
    if unknown:
        sys.exit(f"STOP: not pre-registered: {unknown}")

    x = x.copy()
    x["_row"] = np.arange(len(x))
    x["_wt"] = x["aaref"].astype(str).str.split(";").str[0]
    x["_mut"] = x["aaalt"].astype(str).str.split(";").str[0]
    al = x["Uniprot_acc"].astype(str).str.split(";")
    pl = x["aapos"].astype(str).str.split(";")
    ok = al.str.len().eq(pl.str.len())
    ex = pd.DataFrame({
        "_row": np.repeat(x["_row"].values[ok.values], al[ok].str.len().values),
        "uniprot_acc": np.concatenate(al[ok].values),
        "position_1": np.concatenate(pl[ok].values)})
    ex["position_1"] = pd.to_numeric(ex["position_1"], errors="coerce")
    ex = ex.dropna(subset=["position_1"])
    ex["position_1"] = ex["position_1"].astype("int64")
    ex = ex.merge(x[["_row", "_wt", "_mut"]], on="_row", how="left") \
           .rename(columns={"_wt": "wt_aa", "_mut": "mut_aa"})
    key = ["uniprot_acc", "position_1", "wt_aa", "mut_aa"]
    ex = ex.drop_duplicates(key)
    m = d.merge(ex.merge(x[["_row"] + score_cols], on="_row", how="left")
                  [key + score_cols], on=key, how="left")
    assert len(m) == len(d)

    cand = []
    for c in score_cols:
        e = prereg[c]
        if e["level"] not in (0, 1, 2) or not e.get("representative", True):
            continue
        s = pd.to_numeric(m[c], errors="coerce")
        if s.notna().mean() < MIN_COVERAGE:
            continue
        cand.append(c)
    log(f"{len(cand)} family representatives above {MIN_COVERAGE:.0%} coverage")

    S = pd.DataFrame({c: pd.to_numeric(m[c], errors="coerce")
                      * (-1 if c in NEGATED else 1) for c in cand})
    inter = S.notna().all(axis=1).values
    mm, SS = m[inter].reset_index(drop=True), S[inter].reset_index(drop=True)
    gene = mm.gene_symbol.values
    label = mm.label.values.astype(np.int64)
    log(f"common intersection: {len(mm):,} variants, "
        f"{mm.gene_symbol.nunique():,} genes")

    rows, per_gene = [], {}
    for c in cand:
        s = SS[c].values
        g = float(roc_auc_score(label, s))
        w, n_ev, auc = within_gene_mean(gene, label, s)
        e = prereg[c]
        rows.append({"predictor": c, "level": e["level"],
                     "confidence": e["confidence"], "family": e.get("family", c),
                     "global": g, "within": w, "change": w - g,
                     "n_genes_evaluable": n_ev})
        per_gene[c] = auc
        log(f"  {c:<26} global {g:.4f}  within {w:.4f}  change {w-g:+.4f}  "
            f"level {e['level']}  {e['confidence']}")
    t = pd.DataFrame(rows)
    if (t["global"] < 0.5).any():
        sys.exit(f"STOP: still inverted: {list(t[t['global']<0.5].predictor)}")

    log("\ndegradation sweep on the common intersection")
    sweep = []
    for c in cand:
        base = SS[c].values
        r0 = pd.Series(base).rank(pct=True).values
        for a in DEG_LEVELS:
            gs, ws = [], []
            for rep in range(N_DEG_REPEATS):
                rr = np.random.default_rng(7000 + int(a * 100) * 31 + rep)
                s = (1 - a) * r0 + a * rr.random(len(r0))
                gs.append(float(roc_auc_score(label, s)))
                ws.append(within_gene_mean(gene, label, s)[0])
            sweep.append({"predictor": c, "alpha": float(a),
                          "global": float(np.mean(gs)),
                          "change": float(np.mean(ws) - np.mean(gs))})
    sw = pd.DataFrame(sweep)
    coef = np.polyfit(sw["global"], sw["change"], 2)
    pred = np.poly1d(coef)
    t["headroom_expected"] = pred(t["global"])
    t["residual"] = t["change"] - t["headroom_expected"]
    log(f"headroom null fitted on {len(sw)} degraded points")

    t.to_csv(RES / "dbnsfp_common.csv", index=False)

    out = {"n_variants": int(len(mm)), "n_genes": int(mm.gene_symbol.nunique()),
           "n_predictors": len(cand), "predictors": cand,
           "min_coverage": MIN_COVERAGE,
           "fit_coefficients": [float(v) for v in coef]}

    def test(df, tag):
        if len(df) < 4:
            out[tag] = None
            return
        lv = df.level.astype(int).values
        r_ch = spearmanr(lv, df.change.values)
        r_rs = spearmanr(lv, df.residual.values)
        r_cf = spearmanr(df["global"].values, df.change.values)
        p_ch, n_perm = exact_or_sampled_perm_p(lv, df.change.values,
                                               r_ch.statistic, rng)
        p_rs, _ = exact_or_sampled_perm_p(lv, df.residual.values,
                                          r_rs.statistic, rng)
        out[tag] = {
            "n": int(len(df)),
            "exposure_vs_change": [float(r_ch.statistic), float(r_ch.pvalue)],
            "exposure_vs_change_perm_p": p_ch, "n_permutations": int(n_perm),
            "exposure_vs_residual": [float(r_rs.statistic), float(r_rs.pvalue)],
            "exposure_vs_residual_perm_p": p_rs,
            "global_vs_change": [float(r_cf.statistic), float(r_cf.pvalue)]}
        log(f"\n{tag}  n={len(df)}")
        log(f"   exposure vs change     rho={r_ch.statistic:+.3f}  "
            f"permutation p={p_ch:.4f}")
        log(f"   exposure vs residual   rho={r_rs.statistic:+.3f}  "
            f"permutation p={p_rs:.4f}   <- headroom removed")
        log(f"   global   vs change     rho={r_cf.statistic:+.3f}  "
            f"p={r_cf.pvalue:.4f}       <- the confound")

    test(t[t.confidence == "high"], "primary_high_confidence")
    test(t, "sensitivity_all_representatives")

    lo = [c for c in cand if prereg[c]["level"] == 0]
    hi = [c for c in cand if prereg[c]["level"] == 2]
    if lo and hi:
        gm = {c: per_gene[c] for c in lo + hi}
        common = None
        for c in lo + hi:
            s = set(gm[c].index)
            common = s if common is None else (common & s)
        common = np.array(sorted(common))
        W_lo = np.mean([gm[c].reindex(common).values for c in lo], axis=0)
        W_hi = np.mean([gm[c].reindex(common).values for c in hi], axis=0)
        G_lo = float(t.set_index("predictor").loc[lo, "global"].mean())
        G_hi = float(t.set_index("predictor").loc[hi, "global"].mean())
        gap_global = G_lo - G_hi
        did = (W_lo - W_hi) - gap_global
        obs = float(np.mean(did))
        ii = np.arange(len(common))
        boot = np.array([np.mean(did[rng.choice(ii, len(ii), replace=True)])
                         for _ in range(N_BOOT)])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        out["difference_in_differences"] = {
            "n_genes": int(len(common)), "level0": lo, "level2": hi,
            "within_gene_gap": float(np.mean(W_lo - W_hi)),
            "global_gap": gap_global,
            "difference_in_differences": obs,
            "ci": [float(np.percentile(boot, 2.5)),
                   float(np.percentile(boot, 97.5))],
            "p": float(min(p, 1.0)),
            "note": ("Positive means the never-trained group closes ground on the "
                     "supervised group once gene identity is removed, which is "
                     "what the exposure hypothesis predicts.")}
        log(f"\ndifference-in-differences over {len(common):,} genes")
        log(f"   gap globally      (level0 - level2)  {gap_global:+.4f}")
        log(f"   gap within genes  (level0 - level2)  {np.mean(W_lo-W_hi):+.4f}")
        log(f"   difference        {obs:+.4f} "
            f"[{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  "
            f"p={min(p,1.0):.4f}")

    (RES / "dbnsfp_common.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'dbnsfp_common.csv'}")
    log(f"wrote {RES / 'dbnsfp_common.json'}")

if __name__ == "__main__":
    main()
