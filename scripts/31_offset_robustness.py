import importlib.util
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
N_BOOT = 500


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gene_offsets(genes, y, pct):
    y = np.asarray(y).astype(int)
    pct = np.asarray(pct, float)
    mb, mp = pct[y == 0].mean(), pct[y == 1].mean()
    df = pd.DataFrame({"g": np.asarray(genes), "y": y, "p": pct,
                       "r": pct - np.where(y == 1, mp, mb)})
    agg = df.groupby("g").agg(n=("y", "size"), npos=("y", "sum"),
                              off_adj=("r", "mean"))
    agg["pf"] = agg.npos / agg.n
    agg = agg.join(df[df.y == 0].groupby("g").p.mean().rename("off_b") - mb)
    agg = agg.join(df[df.y == 1].groupby("g").p.mean().rename("off_p") - mp)
    return agg


def align(agg, col="off_adj", min_pos=0, min_neg=0):
    a = agg[(agg.npos >= min_pos) & (agg.n - agg.npos >= min_neg)].dropna(subset=[col])
    if len(a) < 20:
        return np.nan, len(a)
    return float(spearmanr(a[col], a.pf).statistic), int(len(a))


def boot_ci(agg, col="off_adj", seed=0):
    a = agg.dropna(subset=[col])
    x, p = a[col].values, a.pf.values
    rng = np.random.default_rng(seed)
    v = [spearmanr(x[i], p[i]).statistic
         for i in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def partial_spearman(x, y, Z):
    R = np.column_stack([rankdata(c) for c in Z.T] + [np.ones(len(x))])
    res = []
    for v in (rankdata(x), rankdata(y)):
        beta, *_ = np.linalg.lstsq(R, v, rcond=None)
        res.append(v - R @ beta)
    return float(np.corrcoef(res[0], res[1])[0, 1])


def exact_group_test(values, in_group):
    v = np.asarray(values, float)
    k = int(np.sum(in_group))
    obs = v[in_group].mean() - v[~in_group].mean()
    n, tot = len(v), v.sum()
    cnt = ext = 0
    for idx in combinations(range(n), k):
        s = v[list(idx)].sum()
        d = s / k - (tot - s) / (n - k)
        ext += abs(d) >= abs(obs) - 1e-12
        cnt += 1
    return float(obs), float(ext / cnt), cnt


def main():
    out = {"specified_before_running": True}
    m33 = load_module("m33", "29_dbnsfp_matched.py")
    m34 = load_module("m34", "30_gene_constraint.py")
    m16 = load_module("m16", "16_within_gene_ranking.py")

    prereg = json.loads(m33.PREREG.read_text(encoding="utf8"))["predictors"]
    atlas = m33.read_atlas()
    m, score_cols = m33.load(prereg)
    acc = m33.accounting(prereg, score_cols, m)
    cand = list(acc[acc.status == "in panel"].predictor)
    S = pd.DataFrame({k: pd.to_numeric(m[k], errors="coerce")
                      * (-1 if k in m33.NEGATED else 1) for k in cand})
    inter = S.notna().all(axis=1).values
    mm, SS = m[inter].reset_index(drop=True), S[inter].reset_index(drop=True)

    def evaluable(df, k=3):
        c = df.groupby("gene_symbol").label.agg(["sum", "size"])
        return c[(c["sum"] >= k) & (c["size"] - c["sum"] >= k)].index

    ev = evaluable(mm)
    keep = mm.gene_symbol.isin(ev).values
    e, E = mm[keep].reset_index(drop=True), SS[keep].reset_index(drop=True)
    y = e.label.values.astype(int)
    log(f"matched set: {len(e):,} variants, {e.gene_symbol.nunique():,} genes")

    rng = np.random.default_rng(42)
    am = rankdata(E["AlphaMissense_score"].values) / len(e)
    pf_in = e.groupby("gene_symbol").label.transform("mean").values
    calib = {"uniform noise": rng.random(len(e)), "AlphaMissense": am,
             "AlphaMissense + 0.1 x gene pathogenic fraction": am + 0.1 * pf_in,
             "AlphaMissense + 0.3 x gene pathogenic fraction": am + 0.3 * pf_in}
    out["calibration"] = {}
    log("\ncalibration of the alignment statistic:")
    for name, s in calib.items():
        a = gene_offsets(e.gene_symbol, y, rankdata(s) / len(s))
        r, n = align(a)
        out["calibration"][name] = r
        log(f"  {name:<50} alignment {r:+.3f}")

    rows, offs = [], {}
    for k in cand:
        a = gene_offsets(e.gene_symbol, y, rankdata(E[k].values) / len(e))
        offs[k] = a
        r, n = align(a)
        rb, _ = align(a, "off_b")
        rp, _ = align(a, "off_p")
        rows.append({"predictor": k,
                     "atlas_type": atlas[k]["type"] if k in atlas else "not in Atlas",
                     "align_adj": r, "align_adj_ci": boot_ci(a),
                     "align_benign": rb, "align_pathogenic": rp, "n_genes": n,
                     "align_adj_ge5": align(a, min_pos=5, min_neg=5)[0],
                     "align_adj_ge10": align(a, min_pos=10, min_neg=10)[0]})
    t = pd.DataFrame(rows)

    use = t[(t.atlas_type != "not in Atlas") & (t.predictor != "MPC_score")]
    grp = (use.atlas_type == "Clinical-trained").values
    for col in ("align_adj", "align_benign", "align_adj_ge5", "align_adj_ge10"):
        d, p, nsplit = exact_group_test(use[col].values, grp)
        lv = use.atlas_type.map(m33.ATLAS_LEVEL).values
        rho = spearmanr(lv, use[col].values).statistic
        p_ord, _ = m33.exact_perm_p(lv, use[col].values)
        out.setdefault("primary_test", {})[col] = {
            "clinical_minus_rest": d, "p_exact": p, "n_splits": nsplit,
            "n_clinical": int(grp.sum()), "n_rest": int((~grp).sum()),
            "atlas_level_rho": float(rho), "atlas_level_p_exact": p_ord}
    pt = out["primary_test"]["align_adj"]
    log(f"\nPRIMARY: clinical-trained minus rest, mean alignment "
        f"{pt['clinical_minus_rest']:+.3f}, exact p = {pt['p_exact']:.5f} "
        f"({pt['n_splits']:,} splits; {pt['n_clinical']} vs {pt['n_rest']})")
    log(f"         Atlas level vs alignment rho {pt['atlas_level_rho']:+.3f}, "
        f"exact p = {pt['atlas_level_p_exact']:.5f}")
    for col in ("align_benign", "align_adj_ge5", "align_adj_ge10"):
        q = out["primary_test"][col]
        log(f"  same test on {col:<16} {q['clinical_minus_rest']:+.3f}  p = {q['p_exact']:.5f}")

    nb = mm[mm.label == 0].groupby("gene_symbol").size()
    broad_genes = nb[nb >= 3].index
    bmask = mm.gene_symbol.isin(broad_genes).values
    bm, BS = mm[bmask].reset_index(drop=True), SS[bmask].reset_index(drop=True)
    yb = bm.label.values.astype(int)
    log(f"\nS3 broad set: {len(bm):,} variants, {bm.gene_symbol.nunique():,} genes "
        f"with >=3 benign ({(bm.groupby('gene_symbol').label.max() == 0).sum():,} benign-only)")
    t["align_benign_broad"] = [align(gene_offsets(bm.gene_symbol, yb,
                                                  rankdata(BS[k].values) / len(bm)),
                                     "off_b")[0] for k in cand]

    cons = m34.load_constraint().set_index("gene")
    gn = pd.read_csv(m34.GNOMAD, sep="\t", usecols=["gene", "mane_select", "canonical",
                                                     "cds_length"], low_memory=False)
    cds = gn[gn.mane_select.astype(str).str.lower().eq("true")] \
        .groupby("gene").cds_length.max()
    allv = pd.read_parquet(DATA / "variants_all.parquet")
    depth = np.log(allv.groupby("gene_symbol").size())
    cov = cons[["mis_z", "loeuf", "pli"]].join(cds.rename("cds_length")) \
        .join(depth.rename("log_n"))
    COVS = ["mis_z", "loeuf", "pli", "cds_length", "log_n"]
    cov["density"] = cov.log_n - np.log(cov.cds_length)
    BREAKDOWN = {"constraint": ["mis_z", "loeuf", "pli"],
                 "constraint_cds": ["mis_z", "loeuf", "pli", "cds_length"],
                 "depth": ["log_n"], "cds": ["cds_length"],
                 "density": ["density"],
                 "constraint_density": ["mis_z", "loeuf", "pli", "density"]}
    part = {name: [] for name in ["pre-specified five"] + list(BREAKDOWN)}
    for k in cand:
        a = offs[k].join(cov, how="inner").dropna(subset=["off_adj"] + COVS)
        part["pre-specified five"].append(
            partial_spearman(a.off_adj.values, a.pf.values, a[COVS].values))
        for name, cols in BREAKDOWN.items():
            part[name].append(partial_spearman(a.off_adj.values, a.pf.values,
                                               a[cols].values))
    t["align_adj_partial"] = part["pre-specified five"]
    for name in BREAKDOWN:
        t[f"align_adj_partial_{name}"] = part[name]
    out["S4_covariates"] = {
        "n_genes": int(len(a)),
        "rho_path_frac_vs": {c_: float(spearmanr(a.pf, a[c_]).statistic)
                             for c_ in COVS + ["density"]}}
    log(f"S4 covariate-adjusted: {len(a):,} genes with all covariates")

    s2 = (mm.stars >= 2).values
    m2, S2 = mm[s2].reset_index(drop=True), SS[s2].reset_index(drop=True)
    ev2 = evaluable(m2)
    k2 = m2.gene_symbol.isin(ev2).values
    e2, E2 = m2[k2].reset_index(drop=True), S2[k2].reset_index(drop=True)
    log(f"S5 >=2-star: {len(e2):,} variants, {len(ev2):,} genes")
    out["S5_two_star"] = {"n_variants": int(len(e2)), "n_genes": int(len(ev2))}
    t["align_adj_2star"] = [align(gene_offsets(e2.gene_symbol, e2.label.values,
                                               rankdata(E2[k].values) / len(e2)))[0]
                            for k in cand]

    d3, _ = m16.load()
    v = pd.read_parquet(DATA / "varity_scores.parquet")[
        ["uniprot_acc", "position_1", "wt_aa", "mut_aa", "varity_r_full"]]
    d3 = d3.merge(v, on=["uniprot_acc", "position_1", "wt_aa", "mut_aa"], how="left")
    d3 = d3.dropna(subset=["varity", "alphamissense", "gmvp"]).reset_index(drop=True)
    ev3 = evaluable(d3)
    e3 = d3[d3.gene_symbol.isin(ev3)].reset_index(drop=True)
    s6 = {}
    for col, lab in [("alphamissense", "AlphaMissense"), ("varity", "VARITY_R_LOO"),
                     ("gmvp", "gMVP"), ("varity_r_full", "VARITY_R (not LOO)")]:
        sub = e3.dropna(subset=[col])
        a = gene_offsets(sub.gene_symbol, sub.label.values,
                         rankdata(sub[col].values) / len(sub))
        r, n = align(a)
        ac = a.join(cov, how="inner").dropna(subset=["off_adj"] + COVS)
        s6[lab] = {"alignment": r, "ci": boot_ci(a), "n_genes": n,
                   "alignment_covariate_adjusted": partial_spearman(
                       ac.off_adj.values, ac.pf.values, ac[COVS].values),
                   "n_genes_adjusted": int(len(ac))}
    out["S6_three_predictor_set"] = {"n_variants": int(len(e3)),
                                     "n_genes": int(len(ev3)), **s6}
    log(f"\nS6 three-predictor set: {len(e3):,} variants, {len(ev3):,} genes")
    for lab, r in s6.items():
        log(f"  {lab:<22} alignment {r['alignment']:+.3f} "
            f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  ({r['n_genes']} genes)   "
            f"covariate-adjusted {r['alignment_covariate_adjusted']:+.3f} "
            f"({r['n_genes_adjusted']} genes)")

    pg = pd.read_parquet(DATA / "pgym_clinical_mapped.parquet")
    key = ["uniprot_acc", "position_1", "wt_aa", "mut_aa"]
    for f, col in [("replication_scores_alphamissense.parquet", "alphamissense"),
                   ("replication_scores_varity.parquet", "varity")]:
        pg = pg.merge(pd.read_parquet(DATA / f)[key + [col]], on=key, how="left")
    gm = pd.read_parquet(DATA / "replication_scores_gmvp.parquet")
    pg = pg.merge(gm, on=["gene_symbol", "position_1", "wt_aa", "mut_aa"], how="left")
    pg = pg.dropna(subset=["alphamissense", "varity", "gmvp"]).drop_duplicates(key)
    ours = allv[key].drop_duplicates()
    ov = pg.merge(ours, on=key, how="inner")
    lab_ours = allv[key + ["label"]].drop_duplicates(key).rename(columns={"label": "lab_cv"})
    agree = pg.merge(lab_ours, on=key, how="inner")
    cnt = pg.groupby("uniprot_acc").label.agg(["sum", "size"])
    evp = cnt[(cnt["sum"] >= 3) & (cnt["size"] - cnt["sum"] >= 3)].index
    ep = pg[pg.uniprot_acc.isin(evp)].reset_index(drop=True)
    s7 = {"n_variants": int(len(ep)), "n_proteins": int(len(evp)),
          "overlap_with_our_clinvar_variants": float(len(ov) / len(pg)),
          "label_agreement_on_overlap": float((agree.label == agree.lab_cv).mean())}
    for col, lab in [("alphamissense", "AlphaMissense"), ("varity", "VARITY_R_LOO"),
                     ("gmvp", "gMVP")]:
        a = gene_offsets(ep.uniprot_acc, ep.label.values, rankdata(ep[col].values) / len(ep))
        r, n = align(a)
        s7[lab] = {"alignment": r, "ci": boot_ci(a), "n_proteins": n}
    novel = pg.merge(ours.assign(_in=1), on=key, how="left")
    novel = novel[novel._in.isna()].drop(columns="_in")
    cn = novel.groupby("uniprot_acc").label.agg(["sum", "size"])
    evn = cn[(cn["sum"] >= 3) & (cn["size"] - cn["sum"] >= 3)].index
    en = novel[novel.uniprot_acc.isin(evn)].reset_index(drop=True)
    s7["not_in_our_set"] = {"n_variants": int(len(en)), "n_proteins": int(len(evn))}
    if len(evn) >= 20:
        for col, lab in [("alphamissense", "AlphaMissense"), ("varity", "VARITY_R_LOO"),
                         ("gmvp", "gMVP")]:
            a = gene_offsets(en.uniprot_acc, en.label.values,
                             rankdata(en[col].values) / len(en))
            r, n = align(a)
            s7["not_in_our_set"][lab] = {"alignment": r, "ci": boot_ci(a),
                                         "n_proteins": n}
    out["S7_proteingym_replication"] = s7
    log(f"\nS7 ProteinGym clinical: {len(ep):,} variants, {len(evp):,} proteins; "
        f"{s7['overlap_with_our_clinvar_variants']:.1%} of its variants are in our set "
        f"(labels agree on {s7['label_agreement_on_overlap']:.1%} of those)")
    for lab in ("AlphaMissense", "VARITY_R_LOO", "gMVP"):
        r = s7[lab]
        log(f"  {lab:<14} alignment {r['alignment']:+.3f} "
            f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  ({r['n_proteins']} proteins)")
    nn = s7["not_in_our_set"]
    log(f"  variants NOT in our ClinVar set: {nn['n_variants']:,} in {nn['n_proteins']} proteins")
    for lab in ("AlphaMissense", "VARITY_R_LOO", "gMVP"):
        if lab in nn:
            r = nn[lab]
            log(f"    {lab:<14} alignment {r['alignment']:+.3f} "
                f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")

    pd.set_option("display.width", 250)
    cols = ["predictor", "atlas_type", "align_adj", "align_benign", "align_pathogenic",
            "align_adj_ge5", "align_adj_ge10", "align_benign_broad",
            "align_adj_partial", "align_adj_2star"]
    log("\n" + t[cols].sort_values("align_adj", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    log("\nmeans by Atlas type:\n" + t.groupby("atlas_type")[cols[2:]].mean()
        .to_string(float_format=lambda x: f"{x:+.3f}"))
    for col in ("align_benign_broad", "align_adj_partial", "align_adj_2star") + tuple(
            f"align_adj_partial_{n}" for n in ("constraint", "constraint_cds", "depth",
                                               "cds", "density", "constraint_density")):
        d, p, _ = exact_group_test(
            t.set_index("predictor").loc[use.predictor, col].values, grp)
        out.setdefault("robustness_group_tests", {})[col] = {"clinical_minus_rest": d,
                                                             "p_exact": p}
        log(f"  group test on {col:<20} {d:+.3f}  exact p = {p:.5f}")

    t.to_csv(RES / "offset_robustness.csv", index=False)
    out["per_predictor"] = t.to_dict(orient="records")
    (RES / "offset_robustness.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"\nwrote {RES / 'offset_robustness.json'}")


if __name__ == "__main__":
    main()
