import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
sys.path.insert(0, str(BASE))
import genenull as gn


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {name:<58} genenull {got:.6f}  script {want:.6f}")
    assert ok, name


def main():
    print(f"genenull {gn.__version__}")
    v = pd.read_parquet(DATA / "variants_all.parquet")

    print("\n1. the null on all 197,904 variants (manuscript: 0.943 / 0.921 / 0.500)")
    r = gn.gene_identity_null(v)
    check("in-sample", r.in_sample, 0.9433, 5e-4)
    check("random 10-fold", r.random_split, 0.9215, 5e-4)
    check("leave-gene-out", r.leave_gene_out, 0.5, 1e-12)

    print("\n2. decomposition, three predictors (script 32)")
    m16 = load_module("m16", "16_within_gene_ranking.py")
    d, _ = m16.load()
    d = d.dropna(subset=["varity", "alphamissense", "gmvp"]).reset_index(drop=True)
    s32 = {r["predictor"]: r for r in json.loads(
        (RES / "same_set_comparison.json").read_text())["table"]}
    for col, name in [("varity", "VARITY_R_LOO"), ("alphamissense", "AlphaMissense"),
                      ("gmvp", "gMVP")]:
        dec = gn.auroc_decomposition(d, col)
        check(f"{name} global, same variants", dec["global_auroc"],
              s32[name]["global_same_set"], 1e-9)
        check(f"{name} within-gene pooled", dec["within_gene_pooled_auroc"],
              s32[name]["within_pooled"], 1e-9)
        check(f"{name} within-gene mean", dec["within_gene_mean_auroc"],
              s32[name]["within_mean"], 1e-9)

    print("\n3. 22-predictor panel (scripts 33, 35)")
    m33 = load_module("m33", "29_dbnsfp_matched.py")
    prereg = json.loads(m33.PREREG.read_text(encoding="utf8"))["predictors"]
    m, sc = m33.load(prereg)
    acc = m33.accounting(prereg, sc, m)
    cand = list(acc[acc.status == "in panel"].predictor)
    S = pd.DataFrame({k: pd.to_numeric(m[k], errors="coerce")
                      * (-1 if k in m33.NEGATED else 1) for k in cand})
    mm = pd.concat([m[["gene_symbol", "label"]], S], axis=1)[S.notna().all(axis=1).values]
    s33 = json.loads((RES / "dbnsfp_matched.json").read_text())
    dec = gn.auroc_decomposition(mm, "AlphaMissense_score")
    check("same-gene pair share", dec["same_gene_pair_share"],
          s33["same_gene_pair_share"], 1e-12)
    m33t = pd.read_csv(RES / "dbnsfp_matched.csv").set_index("predictor")
    s35 = pd.read_csv(RES / "offset_robustness.csv").set_index("predictor")
    for k in ("AlphaMissense_score", "VARITY_R_LOO_score", "MutFormer_score",
              "SIFT_score", "gMVP_score"):
        dk = gn.auroc_decomposition(mm, k)
        check(f"{k} cross-gene AUROC", dk["cross_gene_auroc"],
              m33t.loc[k, "cross_gene"], 1e-9)
        al = gn.gene_offset_alignment(mm, k, n_boot=10)
        check(f"{k} gene-offset alignment", al["alignment"],
              s35.loc[k, "align_adj"], 1e-9)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
