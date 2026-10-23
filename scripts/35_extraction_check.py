import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
KEY = ["uniprot_acc", "position_1", "wt_aa", "mut_aa"]


def load(name, f):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    m33, m16, m35 = (load("m33", "29_dbnsfp_matched.py"),
                     load("m16", "16_within_gene_ranking.py"),
                     load("m35", "31_offset_robustness.py"))
    prereg = json.loads(m33.PREREG.read_text(encoding="utf8"))["predictors"]
    m, sc = m33.load(prereg)
    d3, _ = m16.load()
    pairs = [("AlphaMissense_score", "alphamissense"),
             ("VARITY_R_LOO_score", "varity"), ("gMVP_score", "gmvp")]
    j = m[KEY + [a for a, _ in pairs]].merge(d3[KEY + [b for _, b in pairs]], on=KEY)
    out = {"agreement": {}}
    for a, b in pairs:
        x = j[[a, b]].dropna()
        dd = (x[a] - x[b]).abs()
        out["agreement"][a] = {"n": int(len(x)),
                               "identical_within_1e-3": float((dd < 1e-3).mean()),
                               "differ_by_more_than_0.05": float((dd > 0.05).mean()),
                               "pearson": float(np.corrcoef(x[a], x[b])[0, 1])}
        print(a, out["agreement"][a])

    acc = m33.accounting(prereg, sc, m)
    cand = list(acc[acc.status == "in panel"].predictor)
    S = pd.DataFrame({k: pd.to_numeric(m[k], errors="coerce")
                      * (-1 if k in m33.NEGATED else 1) for k in cand})
    mm = m[S.notna().all(axis=1).values].reset_index(drop=True)
    c = mm.groupby("gene_symbol").label.agg(["sum", "size"])
    ev = c[(c["sum"] >= 3) & (c["size"] - c["sum"] >= 3)].index
    e = mm[mm.gene_symbol.isin(ev)].reset_index(drop=True)
    e = e.merge(d3[KEY + ["alphamissense", "varity"]], on=KEY, how="left")
    out["alignment_dbnsfp_vs_direct"] = {}
    for a, b in pairs[:2]:
        x = e.dropna(subset=[a, b]).reset_index(drop=True)
        r = {}
        for col in (a, b):
            g = m35.gene_offsets(x.gene_symbol, x.label.values,
                                 rankdata(x[col].values) / len(x))
            r["dbnsfp" if col == a else "direct"] = m35.align(g)[0]
        r["n_variants"] = int(len(x))
        out["alignment_dbnsfp_vs_direct"][a] = r
        print(a, r)
    (RES / "extraction_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
