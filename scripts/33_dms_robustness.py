import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
DATA = BASE / "data"
PRED = ["AlphaMissense", "VARITY_R_LOO", "gMVP"]
SUFFIX = "_snvmean" if "--snv-mean" in sys.argv else ""


def log(m=""):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def tests(df, unit):
    out = {"n": int(len(df)),
           "mean_rho": {p: float(df[f"{p}_rho"].mean()) for p in PRED}}
    out["ranking"] = sorted(PRED, key=lambda p: -out["mean_rho"][p])
    for a, b in combinations(PRED, 2):
        d = (df[f"{a}_rho"] - df[f"{b}_rho"]).values
        p = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else np.nan
        out[f"{a} vs {b}"] = {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
                              "n": int(len(d)), "p": p}
    log(f"  {unit:<44} n={len(df):>3}  ranking {' > '.join(out['ranking'])}")
    for a, b in combinations(PRED, 2):
        r = out[f"{a} vs {b}"]
        log(f"      {a} vs {b:<14} {r['mean_diff']:+.4f}  wins {r['wins']}/{r['n']}  "
            f"p = {r['p']:.4f}")
    return out


def main():
    a = pd.read_csv(RES / f"dms_arbitration{SUFFIX}.csv")
    meta = pd.read_csv(DATA / "proteingym" / "DMS_substitutions.csv")[
        ["DMS_id", "coarse_selection_type"]]
    a = a.merge(meta, on="DMS_id", how="left")
    v = pd.read_parquet(DATA / "rebuild" / "variants_all.parquet")
    cv = v.groupby("uniprot_acc").label.agg(n_clinvar="size", n_path="sum")
    a = a.merge(cv, left_on="uniprot_acc", right_index=True, how="left") \
        .fillna({"n_clinvar": 0, "n_path": 0})
    log(f"{len(a)} assays on {a.uniprot_acc.nunique()} proteins")

    out = {}
    log("\nper assay (as submitted):")
    out["per_assay"] = tests(a, "all assays")

    prot = a.groupby("uniprot_acc").agg(
        **{f"{p}_rho": (f"{p}_rho", "mean") for p in PRED},
        n_clinvar=("n_clinvar", "first"), n_path=("n_path", "first"))
    log("\nper protein (assays averaged within protein):")
    out["per_protein"] = tests(prot, "all proteins")

    q75, q50 = prot.n_clinvar.quantile(0.75), prot.n_clinvar.median()
    out["thresholds"] = {"clinvar_q75": float(q75), "clinvar_median": float(q50)}
    log(f"\nheavily studied: ClinVar missense variants per protein, "
        f"75th percentile {q75:.0f}, median {q50:.0f}")
    out["excl_top_quartile"] = tests(prot[prot.n_clinvar < q75],
                                     "proteins below the top ClinVar quartile")
    out["excl_above_median"] = tests(prot[prot.n_clinvar <= q50],
                                     "proteins at or below the ClinVar median")

    log("\ndisease-relevant proteins:")
    out["disease_ge1"] = tests(prot[prot.n_path >= 1], ">=1 ClinVar pathogenic missense")
    out["disease_ge5"] = tests(prot[prot.n_path >= 5], ">=5 ClinVar pathogenic missense")
    out["not_disease"] = tests(prot[prot.n_path == 0], "no ClinVar pathogenic missense")

    log("\nassay type:")
    out["assay_types"] = a.coarse_selection_type.value_counts().to_dict()
    fn = a.coarse_selection_type.isin(["Activity", "OrganismalFitness"])
    out["function_assays"] = tests(a[fn], "Activity / OrganismalFitness assays")
    out["other_assays"] = tests(a[~fn], "Stability / Expression / Binding assays")

    (RES / f"dms_robustness{SUFFIX}.json").write_text(
        json.dumps(out, indent=2, default=float))
    log(f"\nwrote {RES / f'dms_robustness{SUFFIX}.json'}")


if __name__ == "__main__":
    main()
