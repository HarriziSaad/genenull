import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
OLD = DATA / "external" / "variant_summary_2023-12.txt.gz"
KEY = ["gene_symbol", "wt_aa", "position_1", "mut_aa"]


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m01 = load_module("m01", "01_build_datasets.py")
    m33 = load_module("m33", "29_dbnsfp_matched.py")
    m35 = load_module("m35", "31_offset_robustness.py")
    m16 = load_module("m16", "16_within_gene_ranking.py")

    m01.CLINVAR = OLD
    old = m01.load_clinvar()
    lab_old = old.groupby(KEY).label.agg(["min", "max"])
    lab_old = lab_old[lab_old["min"] == lab_old["max"]]["min"].rename("label_2023")
    log(f"2023-12: {len(lab_old):,} unambiguous missense protein changes")

    allv = pd.read_parquet(DATA / "variants_all.parquet")
    allv = allv.merge(lab_old.reset_index(), on=KEY, how="left")
    status = np.where(allv.label_2023.isna(), "new",
                      np.where(allv.label_2023 == allv.label, "same", "changed"))
    counts = pd.Series(status).value_counts().to_dict()
    log(f"our 2026 variants: {counts}")
    out = {"old_release": "variant_summary_2023-12", "status_counts": counts}
    newkeys = allv.loc[status == "new", KEY]

    hist = old.groupby("gene_symbol").label.agg(n_2023="size", npos_2023="sum")
    hist["pf_2023"] = hist.npos_2023 / hist.n_2023

    prereg = json.loads(m33.PREREG.read_text(encoding="utf8"))["predictors"]
    atlas = m33.read_atlas()
    m, score_cols = m33.load(prereg)
    acc = m33.accounting(prereg, score_cols, m)
    cand = list(acc[acc.status == "in panel"].predictor)
    S = pd.DataFrame({k: pd.to_numeric(m[k], errors="coerce")
                      * (-1 if k in m33.NEGATED else 1) for k in cand})
    inter = S.notna().all(axis=1).values
    mm, SS = m[inter].reset_index(drop=True), S[inter].reset_index(drop=True)
    isnew = mm.merge(newkeys.assign(_n=1), on=KEY, how="left")._n.notna().values
    nm, NS = mm[isnew].reset_index(drop=True), SS[isnew].reset_index(drop=True)
    log(f"panel intersection: {len(mm):,} variants, of which new since 2023-12: {len(nm):,}")
    out["panel_new_variants"] = int(len(nm))

    use_pred = [k for k in cand if k in atlas and k != "MPC_score"]
    grp = np.array([atlas[k]["type"] == "Clinical-trained" for k in use_pred])

    def panel_test(df, sc, pf_source, tag):
        res = {}
        rows = []
        for k in cand:
            a = m35.gene_offsets(df.gene_symbol, df.label.values,
                                 rankdata(sc[k].values) / len(df))
            if pf_source is not None:
                a = a.join(pf_source, how="inner")
                a["pf"] = a[pf_source.name]
            rows.append({"predictor": k, "alignment": m35.align(a)[0],
                         "n_genes": m35.align(a)[1],
                         "atlas_type": atlas[k]["type"] if k in atlas else "not in Atlas"})
        t = pd.DataFrame(rows).set_index("predictor")
        d, p, nsplit = m35.exact_group_test(t.loc[use_pred, "alignment"].values, grp)
        lv = np.array([m33.ATLAS_LEVEL[atlas[k]["type"]] for k in use_pred])
        rho = spearmanr(lv, t.loc[use_pred, "alignment"].values).statistic
        p_ord, _ = m33.exact_perm_p(lv, t.loc[use_pred, "alignment"].values)
        res = {"n_genes": int(t.n_genes.max()), "clinical_minus_rest": d,
               "p_exact": p, "n_splits": nsplit, "atlas_level_rho": float(rho),
               "atlas_level_p_exact": p_ord,
               "by_type": t.groupby("atlas_type").alignment.mean().to_dict(),
               "per_predictor": t.alignment.to_dict()}
        log(f"\n{tag}: {res['n_genes']} genes")
        log(f"  clinical-trained minus rest {d:+.3f}, exact p = {p:.5f};  "
            f"Atlas level rho {rho:+.3f}, exact p = {p_ord:.5f}")
        log("  means by type: " + ", ".join(f"{k} {v:+.3f}" for k, v in res["by_type"].items()))
        for k, r in t.sort_values("alignment", ascending=False).iterrows():
            log(f"    {k:<24} {r.atlas_type:<17} {r.alignment:+.3f}")
        return res

    c = nm.groupby("gene_symbol").label.agg(["sum", "size"])
    ev1 = c[(c["sum"] >= 3) & (c["size"] - c["sum"] >= 3)].index
    k1 = nm.gene_symbol.isin(ev1).values
    out["T1_new_labels"] = panel_test(nm[k1].reset_index(drop=True),
                                      NS[k1].reset_index(drop=True), None,
                                      "T1 new variants, new labels (>=3 new of each class)")

    hev = hist[(hist.npos_2023 >= 3) & (hist.n_2023 - hist.npos_2023 >= 3)]
    nn = nm.groupby("gene_symbol").size()
    ev2 = hev.index.intersection(nn[nn >= 3].index)
    k2 = nm.gene_symbol.isin(ev2).values
    out["T2_historical_composition"] = panel_test(
        nm[k2].reset_index(drop=True), NS[k2].reset_index(drop=True),
        hist["pf_2023"], "T2 new variants vs 2023 gene composition")

    d3, _ = m16.load()
    d3 = d3.dropna(subset=["varity", "alphamissense", "gmvp"]).reset_index(drop=True)
    new3 = d3.merge(newkeys.assign(_n=1), on=KEY, how="left")._n.notna().values
    n3 = d3[new3].reset_index(drop=True)
    c3 = n3.groupby("gene_symbol").label.agg(["sum", "size"])
    ev3 = c3[(c3["sum"] >= 3) & (c3["size"] - c3["sum"] >= 3)].index
    e3 = n3[n3.gene_symbol.isin(ev3)].reset_index(drop=True)
    three = {}
    log(f"\nthree-predictor set, new variants: {len(n3):,}; evaluable genes {len(ev3)}")
    for col, lab in [("varity", "VARITY_R_LOO"), ("gmvp", "gMVP"),
                     ("alphamissense", "AlphaMissense")]:
        a = m35.gene_offsets(e3.gene_symbol, e3.label.values,
                             rankdata(e3[col].values) / len(e3))
        r, n = m35.align(a)
        three[lab] = {"alignment": r, "ci": m35.boot_ci(a), "n_genes": n}
        log(f"  {lab:<14} alignment {r:+.3f} [{three[lab]['ci'][0]:+.3f}, "
            f"{three[lab]['ci'][1]:+.3f}]  ({n} genes)")
    out["three_predictor_new"] = {"n_variants": int(len(n3)), **three}

    (RES / "temporal_split.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"\nwrote {RES / 'temporal_split.json'}")


if __name__ == "__main__":
    main()
