import json
import math
import sys
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
PREREG = BASE / "prereg" / "predictor_exposure.json"
ATLAS = DATA / "atlas_coding_veps.csv"

MIN_PER_CLASS = 3
MIN_COVERAGE = 0.50
NEGATED = {"SIFT_score", "SIFT4G_score", "PROVEAN_score", "FATHMM_score",
           "ESM1b_score", "popEVE_score", "LRT_score"}
DEG_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_DEG_REPEATS = 3
N_BOOT = 1000

ATLAS_NAME = {
    "SIFT_score": "SIFT - Sorting Intolerant From Tolerant",
    "Polyphen2_HVAR_score": "PolyPhen-2 - Polymorphism Phenotyping v2",
    "MutationTaster_score": "MutationTaster2021",
    "MutationAssessor_score": "MutationAssessor",
    "PROVEAN_score": "PROVEAN - PROtein Variant Effect ANalyzer",
    "VEST4_score": "VEST4 - Variant Effect Scoring Tool 4",
    "MutPred2_score": "MutPred2",
    "MVP_score": "MVP - Missense Variant Pathogenicity prediction",
    "gMVP_score": "gMVP - graphical Missense Variant Pathogenicity Predictor",
    "MPC_score": "MPC - Missense badness, PolyPhen-2 and Constraint",
    "PrimateAI_score": "PrimateAI",
    "DEOGEN2_score": "DEOGEN2",
    "LIST-S2_score": "LIST_S2",
    "VARITY_R_LOO_score": "VARITY",
    "ESM1b_score": "ESM-1b",
    "AlphaMissense_score": "AlphaMissense",
    "PHACTboost_score": "PHACTboost",
    "MutFormer_score": "MutFormer",
    "MutScore_score": "MutScore",
    "popEVE_score": "popEVE",
    "DANN_score": "DANN",
}
ATLAS_LEVEL = {"Population-free": 0, "Population-tuned": 1,
               "Clinical-trained": 2}

AMENDED = {
    "PHACTboost_score": (2, "Plan marked it low-confidence pending a check of "
                            "the training target. Dereli et al. 2024 train on "
                            "ClinVar, gnomAD and HSV; the Atlas lists it as "
                            "clinical-trained (R2.7)."),
    "Polyphen2_HVAR_score": (2, "HumVar is UniProt disease-causing versus common "
                                "variants; the plan's own level 2 includes "
                                "curated UniProt disease sets (R2.7)."),
    "MPC_score": ("x", "MPC combines missense badness, PolyPhen-2 and regional "
                       "constraint. PolyPhen-2 is in the panel, so the plan's "
                       "ensemble rule excludes it (R2.7)."),
}

RECOVERED = {
    "CADD_raw": "the Atlas lists PolyPhen-2 output among its features",
    "Eigen-raw_coding": "the Atlas lists SIFT, PolyPhen-2 and MutationAssessor "
                        "among its inputs",
}

ENSEMBLE_NAMED = {"MPC_score", "MVP_score", "DANN_score", "DEOGEN2_score",
                  "MutFormer_score"}
ENSEMBLE_NAMED_PLUS = ENSEMBLE_NAMED | {"VARITY_R_LOO_score", "MutScore_score"}


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def exact_perm_p(x, y):
    x = np.asarray(x)
    n = len(x)
    rx = rankdata(x)
    ry2 = np.rint(2 * rankdata(y)).astype(int)
    levels = sorted(set(x.tolist()))
    if len(levels) > 3:
        raise ValueError("at most three exposure levels")
    sizes = [int((x == L).sum()) for L in levels]
    a = [float(rx[x == L][0]) for L in levels]
    while len(sizes) < 3:
        sizes.append(0)
        a.append(0.0)
    n0, n1, n2 = sizes
    smax = int(ry2.sum())
    D = np.zeros((n0 + 1, n1 + 1, smax + 1, smax + 1))
    D[0, 0, 0, 0] = 1.0
    for v in ry2:
        new = D.copy()
        if n0:
            new[1:, :, v:, :] += D[:-1, :, :smax + 1 - v, :]
        if n1:
            new[:, 1:, :, v:] += D[:, :-1, :, :smax + 1 - v]
        D = new
    C = D[n0, n1]
    total = C.sum()
    expect = math.factorial(n) // (math.factorial(n0) * math.factorial(n1)
                                   * math.factorial(n2))
    assert abs(total - expect) < 0.5, (total, expect)
    s0, s1 = np.meshgrid(np.arange(smax + 1), np.arange(smax + 1), indexing="ij")
    T = (a[0] * s0 + a[1] * s1 + a[2] * (smax - s0 - s1)) / 2.0
    E = rx.sum() * ry2.sum() / 2.0 / n
    t_obs = float((rx * ry2).sum() / 2.0)
    extreme = (C > 0) & (np.abs(T - E) >= abs(t_obs - E) - 1e-9)
    return float(C[extreme].sum() / total), int(expect)


def _self_test():
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.integers(0, 3, 7)
        while len(set(x.tolist())) < 2:
            x = rng.integers(0, 3, 7)
        y = rng.normal(size=7)
        obs = abs(spearmanr(x, y).statistic)
        brute = np.mean([abs(spearmanr(x, p).statistic) >= obs - 1e-12
                         for p in permutations(y)])
        dp, _ = exact_perm_p(x, y)
        assert abs(brute - dp) < 1e-9, (brute, dp)
    log("exact permutation test agrees with brute force at n = 7")


def fast_auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def per_gene_u(codes, y, s, n_genes):
    r = pd.Series(s).groupby(codes).rank(method="average").values
    pos = np.bincount(codes, weights=np.where(y == 1, r, 0.0), minlength=n_genes)
    p = np.bincount(codes, weights=y, minlength=n_genes)
    t = np.bincount(codes, minlength=n_genes)
    return pos - p * (p + 1) / 2, p * (t - p)


def load(prereg):
    d = pd.read_parquet(DATA / "variants_all.parquet")
    x = pd.read_parquet(DATA / "dbnsfp_scores.parquet")
    score_cols = [c for c in x.columns if c.endswith("_score") or c in RECOVERED]
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
    return m, score_cols


def accounting(prereg, score_cols, m):
    rows = []
    for c, e in prereg.items():
        in_extract = c in score_cols
        cov = float(pd.to_numeric(m[c], errors="coerce").notna().mean()) \
            if in_extract else np.nan
        if not in_extract:
            reason = ("dropped at extraction: column name does not end in "
                      "'_score'" if not c.endswith("_score") else
                      "no column of this name among the dbNSFP 5.3.1a score "
                      "columns")
        elif c in RECOVERED:
            reason = ("recovered in revision; excluded by the ensemble rule: "
                      + RECOVERED[c])
        elif e["level"] == "x":
            reason = f"excluded by plan: {e.get('reason', 'ensemble')}"
        elif not e.get("representative", True):
            reason = f"not the family representative ({e.get('family')})"
        elif cov < MIN_COVERAGE:
            reason = "coverage below 50%"
        else:
            reason = "in panel"
        rows.append({"predictor": c, "family": e.get("family", c),
                     "level_plan": e["level"], "confidence": e["confidence"],
                     "representative": e.get("representative", True),
                     "in_extract": in_extract, "coverage": cov,
                     "status": reason})
    return pd.DataFrame(rows)


def read_atlas():
    t = pd.read_csv(ATLAS, header=1)
    t.columns = [c.strip() for c in t.columns]
    t = t.set_index("Predictor")
    out = {}
    for col, name in ATLAS_NAME.items():
        if name not in t.index:
            sys.exit(f"STOP: {name} not found in the Atlas table")
        out[col] = {"type": t.loc[name, "Type"],
                    "other_predictors": str(t.loc[name, "Other Predictors"])
                    .strip().upper() == "TRUE",
                    "training_data": t.loc[name, "Training data"]}
    return out


def main():
    _self_test()
    prereg = json.loads(PREREG.read_text(encoding="utf8"))["predictors"]
    atlas = read_atlas()
    m, score_cols = load(prereg)

    acc = accounting(prereg, score_cols, m)
    acc.to_csv(RES / "dbnsfp_accounting.csv", index=False)
    log(f"accounting: {len(acc)} classified predictors, "
        f"{int(acc.in_extract.sum())} in the extract, "
        f"{int((acc.status == 'in panel').sum())} in the panel")
    for s, g in acc.groupby("status"):
        log(f"   {len(g):>2}  {s}: {', '.join(g.predictor)}")

    cand = list(acc[acc.status == "in panel"].predictor)
    S = pd.DataFrame({c: pd.to_numeric(m[c], errors="coerce")
                      * (-1 if c in NEGATED else 1) for c in cand})
    inter = S.notna().all(axis=1).values
    mm, SS = m[inter].reset_index(drop=True), S[inter].reset_index(drop=True)
    log(f"\ncommon intersection: {len(mm):,} variants, "
        f"{mm.gene_symbol.nunique():,} genes, {len(cand)} predictors")
    assert len(cand) == 22 and len(mm) == 112248, "does not reproduce script 28"

    cnt = mm.groupby("gene_symbol").label.agg(["sum", "size"])
    ev = cnt[(cnt["sum"] >= MIN_PER_CLASS)
             & (cnt["size"] - cnt["sum"] >= MIN_PER_CLASS)].index
    e = mm[mm.gene_symbol.isin(ev)].reset_index(drop=True)
    E = SS[mm.gene_symbol.isin(ev).values].reset_index(drop=True)
    codes, uniq = pd.factorize(e.gene_symbol)
    G = len(uniq)
    y = e.label.values.astype(np.int64)
    log(f"matched set: {G:,} evaluable genes, {len(e):,} variants "
        f"({len(e) / len(mm):.1%} of the intersection)")

    U, PN = {}, None
    rows = []
    for c in cand:
        s = E[c].values
        u, pn = per_gene_u(codes, y, s, G)
        U[c], PN = u, pn
        rows.append({
            "predictor": c, "family": prereg[c].get("family", c),
            "level_submitted": prereg[c]["level"],
            "confidence": prereg[c]["confidence"],
            "level_amended": AMENDED.get(c, (prereg[c]["level"],))[0],
            "atlas_type": atlas[c]["type"] if c in atlas else "not in Atlas",
            "atlas_other_predictors": atlas[c]["other_predictors"]
            if c in atlas else np.nan,
            "global_full": float(roc_auc_score(mm.label, SS[c])),
            "global_matched": fast_auc(y, s),
            "within_mean": float((u / pn).mean()),
            "within_pooled": float(u.sum() / pn.sum())})
    t = pd.DataFrame(rows)
    t["change_submitted"] = t.within_mean - t.global_full
    t["change_mean"] = t.within_mean - t.global_matched
    t["change_pooled"] = t.within_pooled - t.global_matched

    n_pos = int(y.sum())
    all_pairs = n_pos * (len(y) - n_pos)
    w_within = float(PN.sum() / all_pairs)
    t["cross_gene"] = [(g * all_pairs - U[c].sum()) / (all_pairs - PN.sum())
                       for c, g in zip(cand, t.global_matched)]
    prior = e.groupby("gene_symbol").label.transform("mean").values
    up, _ = per_gene_u(codes, y, prior, G)
    null_ref = {"global_matched": fast_auc(y, prior),
                "within_pooled": float(up.sum() / PN.sum()),
                "cross_gene": float((fast_auc(y, prior) * all_pairs - up.sum())
                                    / (all_pairs - PN.sum()))}
    log(f"same-gene share of pathogenic-benign pairs: {w_within:.4f}")
    log(f"in-sample gene prior on the matched set: global "
        f"{null_ref['global_matched']:.4f}, cross-gene {null_ref['cross_gene']:.4f}")

    old = pd.read_csv(RES / "dbnsfp_common.csv").set_index("predictor")
    dg = (t.set_index("predictor").global_full - old["global"]).abs().max()
    dw = (t.set_index("predictor").within_mean - old["within"]).abs().max()
    log(f"reproduces script 28: max |diff| global {dg:.1e}, within {dw:.1e}")
    assert dg < 1e-9 and dw < 1e-9

    log("\ndegradation sweep on the matched set")
    sweep = []
    for c in cand:
        r0 = rankdata(E[c].values) / len(e)
        for a in DEG_LEVELS:
            g_, wm_, wp_ = [], [], []
            for rep in range(N_DEG_REPEATS):
                rr = np.random.default_rng(7000 + int(a * 100) * 31 + rep)
                s = (1 - a) * r0 + a * rr.random(len(r0))
                u, _ = per_gene_u(codes, y, s, G)
                g_.append(fast_auc(y, s))
                wm_.append((u / PN).mean())
                wp_.append(u.sum() / PN.sum())
            sweep.append({"predictor": c, "alpha": float(a),
                          "global": np.mean(g_),
                          "change_mean": np.mean(wm_) - np.mean(g_),
                          "change_pooled": np.mean(wp_) - np.mean(g_)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(RES / "dbnsfp_matched_sweep.csv", index=False)
    fits, fitfn = {}, {}
    for k in ("mean", "pooled"):
        coef = np.polyfit(sw["global"], sw[f"change_{k}"], 2)
        fits[k] = [float(v) for v in coef]
        fitfn[k] = np.poly1d(coef)
        t[f"headroom_{k}"] = fitfn[k](t.global_matched)
        t[f"residual_{k}"] = t[f"change_{k}"] - t[f"headroom_{k}"]
    log(f"   {len(sw)} degraded points")

    log(f"\ngene bootstrap, {N_BOOT} replicates (global AUROC recomputed each time)")
    rng = np.random.default_rng(42)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(codes, minlength=G))]
    gene_rows = [order[starts[g]:starts[g + 1]] for g in range(G)]
    Emat = E[cand].values
    Umat = np.vstack([U[c] for c in cand])
    P = len(cand)
    Gb, Wmb, Wpb = (np.empty((N_BOOT, P)) for _ in range(3))
    for b in range(N_BOOT):
        gs = rng.integers(0, G, G)
        idx = np.concatenate([gene_rows[g] for g in gs])
        yy = y[idx]
        for j in range(P):
            Gb[b, j] = fast_auc(yy, Emat[idx, j])
        Wmb[b] = (Umat[:, gs] / PN[gs]).mean(axis=1)
        Wpb[b] = Umat[:, gs].sum(axis=1) / PN[gs].sum()
        if (b + 1) % 250 == 0:
            log(f"   {b + 1}/{N_BOOT}")

    obsG = t.global_matched.values
    obsW = {"mean": t.within_mean.values, "pooled": t.within_pooled.values}
    bootW = {"mean": Wmb, "pooled": Wpb}

    def interval(v, obs):
        v = np.asarray(v)
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        return {"estimate": float(obs),
                "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "p": float(min(p, 1.0))}

    def did(lo, hi, metric, headroom=False):
        lo, hi = np.array(lo), np.array(hi)
        W, Wb = obsW[metric], bootW[metric]
        f = fitfn[metric]

        def stat(Wv, Gv, l, h):
            ch = Wv - Gv - (f(Gv) if headroom else 0.0)
            return ch[l].mean() - ch[h].mean()

        obs = stat(W, obsG, lo, hi)
        r = np.random.default_rng(11)
        genes = [stat(Wb[b], Gb[b], lo, hi) for b in range(N_BOOT)]
        preds, both = [], []
        for b in range(N_BOOT):
            l_, h_ = r.choice(lo, len(lo)), r.choice(hi, len(hi))
            preds.append(stat(W, obsG, l_, h_))
            both.append(stat(Wb[b], Gb[b], l_, h_))
        gap_g = obsG[lo].mean() - obsG[hi].mean()
        gap_w = W[lo].mean() - W[hi].mean()
        return {"n_low": int(len(lo)), "n_high": int(len(hi)),
                "global_gap": float(gap_g), "within_gap": float(gap_w),
                "did": float(obs),
                "boot_genes": interval(genes, obs),
                "boot_predictors": interval(preds, obs),
                "boot_both": interval(both, obs)}

    def lv(col):
        out = {}
        for i, c in enumerate(cand):
            v = t.loc[i, col]
            out[c] = None if v in ("x", "not in Atlas") else v
        return out

    atlas_lv = {c: ATLAS_LEVEL.get(atlas[c]["type"]) if c in atlas else None
                for c in cand}
    atlas_lv["MPC_score"] = None
    atlas_misfit = dict(atlas_lv, MisFit_S_score=1)
    atlas_nometa = {c: (None if (c in atlas and atlas[c]["other_predictors"])
                        else v) for c, v in atlas_lv.items()}

    schemes = {
        "submitted": {c: int(prereg[c]["level"]) for c in cand},
        "submitted_high_confidence": {c: int(prereg[c]["level"]) for c in cand
                                      if prereg[c]["confidence"] == "high"},
        "amended": {c: int(v) for c, v in lv("level_amended").items()
                    if v is not None},
        "atlas": {c: v for c, v in atlas_lv.items() if v is not None},
        "atlas_misfit_as_tuned": {c: v for c, v in atlas_misfit.items()
                                  if v is not None},
        "atlas_excluding_meta": {c: v for c, v in atlas_nometa.items()
                                 if v is not None},
    }
    amended = schemes["amended"]
    for tag, drop in (("named", ENSEMBLE_NAMED), ("named_plus", ENSEMBLE_NAMED_PLUS)):
        schemes[f"amended_ensemble_{tag}"] = {c: v for c, v in amended.items()
                                              if c not in drop}
        schemes[f"atlas_ensemble_{tag}"] = {c: v for c, v in schemes["atlas"].items()
                                            if c not in drop}
    schemes["amended_ensemble_flagged"] = {
        c: v for c, v in amended.items()
        if not (c in atlas and atlas[c]["other_predictors"])}

    idx_of = {c: i for i, c in enumerate(cand)}
    results = {}
    for name, levels in schemes.items():
        cs = list(levels)
        ii = np.array([idx_of[c] for c in cs])
        lev = np.array([levels[c] for c in cs])
        res = {"n": len(cs), "levels": {c: int(levels[c]) for c in cs},
               "counts": {int(k): int((lev == k).sum()) for k in sorted(set(lev))}}
        for metric in ("submitted", "mean", "pooled"):
            ch = (t.change_submitted if metric == "submitted"
                  else t[f"change_{metric}"]).values[ii]
            rho = spearmanr(lev, ch).statistic
            p, n_arr = exact_perm_p(lev, ch)
            entry = {"exposure_vs_change": {"rho": float(rho), "p_exact": p,
                                            "n_arrangements": n_arr}}
            if metric != "submitted":
                resid = t[f"residual_{metric}"].values[ii]
                rr = spearmanr(lev, resid).statistic
                pr, _ = exact_perm_p(lev, resid)
                entry["exposure_vs_residual"] = {"rho": float(rr), "p_exact": pr}
            res[metric] = entry

        lo_lv = [0] if name.startswith(("submitted", "amended")) else [0, 1]
        lo = [idx_of[c] for c in cs if levels[c] in lo_lv]
        hi = [idx_of[c] for c in cs if levels[c] == 2]
        res["did_groups"] = {"low_levels": lo_lv,
                             "low": [cand[i] for i in lo],
                             "high": [cand[i] for i in hi]}
        if name.startswith("submitted"):
            W, Gf = t.within_mean.values, t.global_full.values
            res["did_submitted_unmatched"] = float(
                (W[lo].mean() - W[hi].mean()) - (Gf[lo].mean() - Gf[hi].mean()))
        res["did_mean"] = did(lo, hi, "mean")
        res["did_pooled"] = did(lo, hi, "pooled")
        res["did_pooled_headroom"] = did(lo, hi, "pooled", headroom=True)
        if name.startswith("atlas"):
            lo0 = [idx_of[c] for c in cs if levels[c] == 0]
            lo1 = [idx_of[c] for c in cs if levels[c] == 1]
            res["did_pooled_free_only"] = did(lo0, hi, "pooled")
            res["did_pooled_tuned_only"] = did(lo1, hi, "pooled")
        results[name] = res

    for name, res in results.items():
        ii = np.array([idx_of[c] for c in res["levels"]])
        for metric in ("submitted", "mean", "pooled"):
            ch = (t.change_submitted if metric == "submitted"
                  else t[f"change_{metric}"]).values[ii]
            g = (t.global_full if metric == "submitted" else t.global_matched).values[ii]
            r = spearmanr(g, ch)
            res[metric]["global_vs_change"] = {"rho": float(r.statistic),
                                               "p_asymptotic": float(r.pvalue)}

    pd.set_option("display.width", 220)
    show = t[["predictor", "level_submitted", "level_amended", "atlas_type",
              "global_full", "global_matched", "within_pooled", "cross_gene",
              "change_submitted", "change_mean", "change_pooled",
              "residual_pooled"]]
    log("\n" + show.sort_values("change_pooled", ascending=False)
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for name, res in results.items():
        log(f"\n=== {name}  n={res['n']}  levels {res['counts']}")
        for metric in ("submitted", "mean", "pooled"):
            r = res[metric]
            line = (f"   {metric:<9} exposure~change rho={r['exposure_vs_change']['rho']:+.3f} "
                    f"p_exact={r['exposure_vs_change']['p_exact']:.4f}   "
                    f"global~change rho={r['global_vs_change']['rho']:+.3f}")
            if "exposure_vs_residual" in r:
                line += (f"   exposure~residual rho={r['exposure_vs_residual']['rho']:+.3f} "
                         f"p_exact={r['exposure_vs_residual']['p_exact']:.4f}")
            log(line)
        if "did_submitted_unmatched" in res:
            log(f"   DiD as submitted (unmatched): {res['did_submitted_unmatched']:+.4f}")
        for k in ("did_mean", "did_pooled", "did_pooled_headroom",
                  "did_pooled_free_only", "did_pooled_tuned_only"):
            if k not in res:
                continue
            d_ = res[k]
            log(f"   {k:<22} {d_['did']:+.4f}  (lo {d_['n_low']} vs hi {d_['n_high']}; "
                f"global gap {d_['global_gap']:+.4f}, within gap {d_['within_gap']:+.4f})")
            for bk in ("boot_genes", "boot_predictors", "boot_both"):
                ci = d_[bk]["ci"]
                log(f"      {bk:<16} [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={d_[bk]['p']:.4f}")

    t.to_csv(RES / "dbnsfp_matched.csv", index=False)
    out = {"n_intersection": int(len(mm)), "n_genes_intersection":
           int(mm.gene_symbol.nunique()), "n_genes_evaluable": int(G),
           "n_variants_matched": int(len(e)), "n_boot": N_BOOT,
           "same_gene_pair_share": w_within, "gene_prior_reference": null_ref,
           "headroom_fit": fits, "amendments": {k: {"level": v[0], "reason": v[1]}
                                                for k, v in AMENDED.items()},
           "atlas_source": json.loads((DATA / "atlas_coding_veps_metadata.json")
                                      .read_text(encoding="utf8")),
           "schemes": results}
    (RES / "dbnsfp_matched.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'dbnsfp_matched.json'}")


if __name__ == "__main__":
    main()
