import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
GNOMAD = DATA / "gnomad.v4.1.constraint_metrics.tsv"
MIN_GENE_N = 5


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_constraint():
    g = pd.read_csv(GNOMAD, sep="\t", low_memory=False)
    for c in ("mane_select", "canonical"):
        g[c] = g[c].astype(str).str.lower().eq("true")
    g = g[g.transcript_type.eq("protein_coding") | g.transcript_type.isna()]
    g = g.sort_values(["mane_select", "canonical"], ascending=False)
    g = g[g.mane_select | g.canonical].drop_duplicates("gene")
    g = g.rename(columns={"mis.z_score": "mis_z", "lof.oe_ci.upper": "loeuf",
                          "mis.oe": "mis_oe", "lof.pLI": "pli"})
    return g[["gene", "mis_z", "loeuf", "mis_oe", "pli"]]


def gene_prior_random(d, seed=42):
    s = np.full(len(d), np.nan)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for tr, te in skf.split(d, d.label):
        f = d.iloc[tr].groupby("gene_symbol").label.mean()
        s[te] = d.iloc[te].gene_symbol.map(f).fillna(0.5).values
    return s


def stratified_auc(y, s, strata):
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        yy = y[m]
        p = yy.sum()
        q = len(yy) - p
        if p == 0 or q == 0:
            continue
        r = rankdata(s[m])
        num += r[yy == 1].sum() - p * (p + 1) / 2
        den += p * q
    return num / den


def main():
    out = {}
    d = pd.read_parquet(DATA / "variants_all.parquet")
    c = load_constraint()
    log(f"gnomAD v4.1 constraint: {len(c):,} genes (MANE Select or canonical)")

    gt = d.groupby("gene_symbol").label.agg(n="size", n_path="sum").reset_index()
    gt["path_frac"] = gt.n_path / gt.n
    gt = gt.merge(c, left_on="gene_symbol", right_on="gene", how="left")
    matched = gt.mis_z.notna()
    log(f"genes matched to gnomAD: {matched.sum():,} of {len(gt):,} "
        f"({d.gene_symbol.isin(gt[matched].gene_symbol).mean():.1%} of variants)")
    out["n_genes"] = int(len(gt))
    out["n_genes_matched"] = int(matched.sum())

    corr = {}
    for min_n in (MIN_GENE_N, 20):
        sub = gt[matched & (gt.n >= min_n)]
        corr[f"genes_n>={min_n}"] = {
            "n_genes": int(len(sub)),
            **{f"path_frac_vs_{k}": float(spearmanr(sub.path_frac, sub[k],
                                                    nan_policy="omit").statistic)
               for k in ("mis_z", "loeuf", "mis_oe")}}
        log(f"  genes n>={min_n:<3} ({len(sub):,}): rho(path_frac, mis_z) "
            f"{corr[f'genes_n>={min_n}']['path_frac_vs_mis_z']:+.3f}   "
            f"rho(path_frac, LOEUF) {corr[f'genes_n>={min_n}']['path_frac_vs_loeuf']:+.3f}")
    out["gene_level_correlation"] = corr
    gt.to_csv(RES / "gene_constraint_genes.csv", index=False)

    v = d.merge(c, left_on="gene_symbol", right_on="gene", how="inner")
    v = v.dropna(subset=["mis_z", "loeuf", "mis_oe"]).reset_index(drop=True)
    y = v.label.values.astype(int)
    log(f"\nvariants with constraint: {len(v):,}")
    out["n_variants_with_constraint"] = int(len(v))
    out["n_genes_with_constraint"] = int(v.gene_symbol.nunique())
    base = {"mis_z": float(roc_auc_score(y, v.mis_z)),
            "minus_loeuf": float(roc_auc_score(y, -v.loeuf)),
            "minus_mis_oe": float(roc_auc_score(y, -v.mis_oe))}
    X = v[["mis_z", "loeuf", "mis_oe"]].values
    pred = np.full(len(v), np.nan)
    for tr, te in GroupKFold(n_splits=10).split(X, y, v.gene_symbol):
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    base["constraint_model_leave_gene_out"] = float(roc_auc_score(y, pred))
    v["gene_null_random"] = gene_prior_random(v)
    base["gene_identity_null_random_split"] = float(
        roc_auc_score(y, v.gene_null_random))
    for k, val in base.items():
        log(f"  {k:<36} AUROC {val:.4f}")
    out["variant_level_baselines"] = base

    gene_dec = (v.drop_duplicates("gene_symbol").set_index("gene_symbol").mis_z
                .rank(pct=True).mul(10).clip(upper=9.999).astype(int))
    strata = v.gene_symbol.map(gene_dec).values
    within_dec = stratified_auc(y, v.gene_null_random.values, strata)
    within_dec_z = stratified_auc(y, v.mis_z.values, strata)
    out["gene_null_within_constraint_deciles"] = float(within_dec)
    out["mis_z_within_constraint_deciles"] = float(within_dec_z)
    log(f"\n  gene-identity null among pairs sharing a mis_z decile: "
        f"{within_dec:.4f}  (mis_z itself: {within_dec_z:.4f})")

    spec = importlib.util.spec_from_file_location(
        "m33", Path(__file__).with_name("29_dbnsfp_matched.py"))
    m33 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m33)
    prereg = json.loads(m33.PREREG.read_text(encoding="utf8"))["predictors"]
    atlas = m33.read_atlas()
    m, score_cols = m33.load(prereg)
    acc = m33.accounting(prereg, score_cols, m)
    cand = list(acc[acc.status == "in panel"].predictor)
    S = pd.DataFrame({k: pd.to_numeric(m[k], errors="coerce")
                      * (-1 if k in m33.NEGATED else 1) for k in cand})
    inter = S.notna().all(axis=1).values
    mm, SS = m[inter].reset_index(drop=True), S[inter].reset_index(drop=True)
    cnt = mm.groupby("gene_symbol").label.agg(["sum", "size"])
    ev = cnt[(cnt["sum"] >= 3) & (cnt["size"] - cnt["sum"] >= 3)].index
    keep = mm.gene_symbol.isin(ev).values
    e, E = mm[keep].reset_index(drop=True), SS[keep].reset_index(drop=True)
    log(f"\npanel matched set: {len(e):,} variants, {e.gene_symbol.nunique():,} genes")

    gstat = e.groupby("gene_symbol").label.agg(pf="mean").join(
        c.set_index("gene")[["mis_z", "loeuf"]], how="left")
    gstat = gstat.dropna()
    log(f"  genes with constraint: {len(gstat):,}")

    def partial(a, b, ctrl):
        ra, rb, rc = (rankdata(x) for x in (a, b, ctrl))
        res = []
        for r in (ra, rb):
            beta = np.polyfit(rc, r, 1)
            res.append(r - np.polyval(beta, rc))
        return float(np.corrcoef(res[0], res[1])[0, 1])

    rows = []
    brng = np.random.default_rng(42)
    for k in cand:
        pct = rankdata(E[k].values) / len(E)
        df = pd.DataFrame({"g": e.gene_symbol.values, "y": e.label.values, "p": pct})
        ben = df[df.y == 0].groupby("g").p.mean()
        pat = df[df.y == 1].groupby("g").p.mean()
        gg = gstat.join(ben.rename("off_b")).join(pat.rename("off_p")).dropna()
        ob, pf_ = gg.off_b.values, gg.pf.values
        bs = []
        for _ in range(500):
            i = brng.integers(0, len(gg), len(gg))
            bs.append(spearmanr(ob[i], pf_[i]).statistic)
        rows.append({
            "benign_offset_vs_path_frac_ci": [float(np.percentile(bs, 2.5)),
                                              float(np.percentile(bs, 97.5))],
            "predictor": k,
            "atlas_type": atlas[k]["type"] if k in atlas else "not in Atlas",
            "benign_offset_vs_path_frac": float(spearmanr(gg.off_b, gg.pf).statistic),
            "benign_offset_vs_mis_z": float(spearmanr(gg.off_b, gg.mis_z).statistic),
            "benign_offset_vs_path_frac_given_mis_z": partial(gg.off_b, gg.pf, gg.mis_z),
            "benign_offset_vs_mis_z_given_path_frac": partial(gg.off_b, gg.mis_z, gg.pf),
            "pathogenic_offset_vs_path_frac": float(spearmanr(gg.off_p, gg.pf).statistic),
            "pathogenic_offset_vs_mis_z": float(spearmanr(gg.off_p, gg.mis_z).statistic),
            "n_genes": int(len(gg))})
    off = pd.DataFrame(rows)
    mt = pd.read_csv(RES / "dbnsfp_matched.csv")[["predictor", "change_pooled",
                                                  "cross_gene", "within_pooled"]]
    off = off.merge(mt, on="predictor")
    off.to_csv(RES / "gene_constraint_offsets.csv", index=False)

    pd.set_option("display.width", 240)
    cols = ["predictor", "atlas_type", "benign_offset_vs_path_frac",
            "benign_offset_vs_mis_z", "benign_offset_vs_path_frac_given_mis_z",
            "benign_offset_vs_mis_z_given_path_frac", "change_pooled"]
    log("\n" + off[cols].sort_values("change_pooled", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    log("\nbenign offset vs ClinVar pathogenic fraction, 95% gene-bootstrap CI:")
    for _, r in off.sort_values("benign_offset_vs_path_frac", ascending=False).iterrows():
        lo, hi = r.benign_offset_vs_path_frac_ci
        flag = "  *" if lo > 0 else ("  (neg)" if hi < 0 else "")
        log(f"  {r.predictor:<24} {r.atlas_type:<17} {r.benign_offset_vs_path_frac:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}]{flag}")

    grp = off.groupby("atlas_type")[cols[2:]].mean()
    log("\nmeans by Atlas type:\n" + grp.to_string(float_format=lambda x: f"{x:+.3f}"))
    for a, b in [("benign_offset_vs_path_frac", "change_pooled"),
                 ("benign_offset_vs_path_frac_given_mis_z", "change_pooled"),
                 ("benign_offset_vs_mis_z_given_path_frac", "change_pooled")]:
        r = spearmanr(off[a], off[b])
        log(f"  across predictors: rho({a}, {b}) = {r.statistic:+.3f}  p = {r.pvalue:.4f}")
        out.setdefault("across_predictors", {})[f"{a}~{b}"] = {
            "rho": float(r.statistic), "p": float(r.pvalue)}
    out["offset_means_by_atlas_type"] = grp.to_dict(orient="index")

    (RES / "gene_constraint.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'gene_constraint.json'}")


if __name__ == "__main__":
    main()
