import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
KEY = ["uniprot_acc", "position_1", "wt_aa", "mut_aa"]


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exploded(score_cols):
    x = pd.read_parquet(DATA / "dbnsfp_scores.parquet")
    x["_row"] = np.arange(len(x))
    x["wt_aa"] = x["aaref"].astype(str).str.split(";").str[0]
    x["mut_aa"] = x["aaalt"].astype(str).str.split(";").str[0]
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
    ex = ex.merge(x[["_row", "wt_aa", "mut_aa"]], on="_row", how="left")
    sc = x[["_row"] + score_cols].copy()
    for c in score_cols:
        sc[c] = pd.to_numeric(sc[c], errors="coerce")
    ex = ex.drop_duplicates(KEY + ["_row"]).merge(sc, on="_row", how="left")
    return ex


def pooled(codes, y, s, G, m33):
    u, pn = m33.per_gene_u(codes, y, s, G)
    return float(u.sum() / pn.sum())


def main():
    m33 = load_module("m33", "29_dbnsfp_matched.py")
    m35 = load_module("m35", "31_offset_robustness.py")
    m16 = load_module("m16", "16_within_gene_ranking.py")
    prereg = json.loads(m33.PREREG.read_text(encoding="utf8"))["predictors"]
    atlas = m33.read_atlas()

    m, score_cols = m33.load(prereg)
    acc = m33.accounting(prereg, score_cols, m)
    cand = list(acc[acc.status == "in panel"].predictor)
    sign = {c: (-1 if c in m33.NEGATED else 1) for c in cand}
    S = pd.DataFrame({c: pd.to_numeric(m[c], errors="coerce") * sign[c] for c in cand})
    inter = S.notna().all(axis=1).values
    mm, SS = m[inter].reset_index(drop=True), S[inter].reset_index(drop=True)
    cnt = mm.groupby("gene_symbol").label.agg(["sum", "size"])
    ev = cnt[(cnt["sum"] >= 3) & (cnt["size"] - cnt["sum"] >= 3)].index
    keep = mm.gene_symbol.isin(ev).values
    e, E1 = mm[keep].reset_index(drop=True), SS[keep].reset_index(drop=True)
    assert len(e) == 58569, len(e)

    ex = exploded(cand)
    n_snv = ex.groupby(KEY).size().rename("n_snv")
    mean = ex.groupby(KEY)[cand].mean()
    spread = (ex.groupby(KEY)[cand].max() - ex.groupby(KEY)[cand].min())
    e2 = e[KEY].merge(n_snv, left_on=KEY, right_index=True, how="left")
    Em = e[KEY].merge(mean, left_on=KEY, right_index=True, how="left")
    Sp = e[KEY].merge(spread, left_on=KEY, right_index=True, how="left")
    assert len(Em) == len(e)
    Em = pd.DataFrame({c: Em[c].values * sign[c] for c in cand})
    assert Em.notna().all().all(), "a matched variant lost its score under the mean"

    out = {"n_variants": int(len(e)), "n_genes": int(e.gene_symbol.nunique()),
           "variants_with_more_than_one_snv": int((e2.n_snv > 1).sum()),
           "fraction_with_more_than_one_snv": float((e2.n_snv > 1).mean())}
    log(f"matched panel set: {len(e):,} variants; "
        f"{out['fraction_with_more_than_one_snv']:.1%} encoded by more than one SNV")

    codes, uniq = pd.factorize(e.gene_symbol)
    G, y = len(uniq), e.label.values.astype(np.int64)
    rows = []
    for c in cand:
        differ = (Sp[c].values > 1e-9)
        r = {"predictor": c,
             "fraction_scores_differ_between_snvs": float(differ.mean())}
        for tag, s in (("first", E1[c].values), ("mean", Em[c].values)):
            g = m33.fast_auc(y, s)
            w = pooled(codes, y, s, G, m33)
            a = m35.gene_offsets(e.gene_symbol, y, rankdata(s) / len(s))
            r.update({f"global_{tag}": g, f"within_pooled_{tag}": w,
                      f"change_pooled_{tag}": w - g, f"alignment_{tag}": m35.align(a)[0]})
        rows.append(r)
        log(f"  {c:<24} SNV scores differ {r['fraction_scores_differ_between_snvs']:6.1%}  "
            f"change {r['change_pooled_first']:+.4f} -> {r['change_pooled_mean']:+.4f}  "
            f"alignment {r['alignment_first']:+.3f} -> {r['alignment_mean']:+.3f}")
    t = pd.DataFrame(rows).set_index("predictor")
    out["per_predictor"] = t.reset_index().to_dict(orient="records")

    atlas_lv = {c: m33.ATLAS_LEVEL.get(atlas[c]["type"]) if c in atlas else None
                for c in cand}
    atlas_lv["MPC_score"] = None
    amended = {c: m33.AMENDED.get(c, (prereg[c]["level"],))[0] for c in cand}
    schemes = {"submitted": ({c: prereg[c]["level"] for c in cand}, [0]),
               "amended": (amended, [0]), "atlas": (atlas_lv, [0, 1])}
    out["did_pooled"] = {}
    for name, (lv, lo_lv) in schemes.items():
        lo = [c for c in cand if lv[c] in lo_lv]
        hi = [c for c in cand if lv[c] == 2]
        d = {tag: float(t.loc[lo, f"change_pooled_{tag}"].mean()
                        - t.loc[hi, f"change_pooled_{tag}"].mean())
             for tag in ("first", "mean")}
        out["did_pooled"][name] = d
        log(f"  pooled difference in gain, {name:<9} {d['first']:+.4f} -> {d['mean']:+.4f}")

    use = [c for c in cand if atlas_lv[c] is not None]
    grp = np.array([atlas_lv[c] == 2 for c in use])
    out["alignment_clinical_minus_rest"] = {}
    for tag in ("first", "mean"):
        obs, p, n = m35.exact_group_test(t.loc[use, f"alignment_{tag}"].values, grp)
        out["alignment_clinical_minus_rest"][tag] = {"difference": obs, "p_exact": p,
                                                     "n_splits": n}
        log(f"  alignment, clinical-trained minus rest ({tag}): {obs:+.3f}, p = {p:.5f}")

    d, cols = m16.load()
    d = d.dropna(subset=["varity", "alphamissense", "gmvp"]).reset_index(drop=True)
    c3 = d.groupby("gene_symbol").label.agg(["sum", "size"])
    ev3 = c3[(c3["sum"] >= 3) & (c3["size"] - c3["sum"] >= 3)].index
    d = d[d.gene_symbol.isin(ev3)].reset_index(drop=True)
    assert len(d) == 96142, len(d)
    vm = d[KEY].merge(mean[["VARITY_R_LOO_score"]], left_on=KEY, right_index=True,
                      how="left").VARITY_R_LOO_score.values
    fallback = np.isnan(vm)
    vm = np.where(fallback, d.varity.values, vm)
    vn = d[KEY].merge(n_snv, left_on=KEY, right_index=True, how="left").n_snv
    codes3, uniq3 = pd.factorize(d.gene_symbol)
    G3, y3 = len(uniq3), d.label.values.astype(np.int64)
    res3 = {"n_variants": int(len(d)),
            "fraction_with_more_than_one_snv": float((vn > 1).mean()),
            "varity_mean_unavailable_kept_first": int(fallback.sum())}
    for tag, v in (("first", d.varity.values), ("mean", vm)):
        r = {}
        for nm, s in (("VARITY_R_LOO", v), ("AlphaMissense", d.alphamissense.values),
                      ("gMVP", d.gmvp.values)):
            r[nm] = {"global": m33.fast_auc(y3, s),
                     "within_pooled": pooled(codes3, y3, s, G3, m33)}
        r["VARITY_minus_AlphaMissense"] = {
            k: r["VARITY_R_LOO"][k] - r["AlphaMissense"][k] for k in ("global", "within_pooled")}
        r["VARITY_minus_gMVP"] = {
            k: r["VARITY_R_LOO"][k] - r["gMVP"][k] for k in ("global", "within_pooled")}
        a = m35.gene_offsets(d.gene_symbol, y3, rankdata(v) / len(v))
        r["VARITY_alignment"] = m35.align(a)[0]
        res3[tag] = r
        log(f"  three predictors ({tag}): VARITY - AlphaMissense global "
            f"{r['VARITY_minus_AlphaMissense']['global']:+.4f}, pooled within "
            f"{r['VARITY_minus_AlphaMissense']['within_pooled']:+.4f}; VARITY - gMVP "
            f"global {r['VARITY_minus_gMVP']['global']:+.4f}, pooled within "
            f"{r['VARITY_minus_gMVP']['within_pooled']:+.4f}; VARITY alignment "
            f"{r['VARITY_alignment']:+.3f}")
    out["three_predictor"] = res3

    (RES / "snv_aggregation.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {RES / 'snv_aggregation.json'}")


if __name__ == "__main__":
    main()
