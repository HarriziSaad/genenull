import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
sys.path.insert(0, str(BASE))
import genenull as gn


def pair_share(df, gene="gene_symbol", label="label"):
    c = df.groupby(gene)[label].agg(["sum", "size"])
    within = float((c["sum"] * (c["size"] - c["sum"])).sum())
    n_pos = float(df[label].sum())
    return within / (n_pos * (len(df) - n_pos))


def main():
    v = pd.read_parquet(DATA / "variants_all_scored.parquet")
    out = {}

    null = gn.gene_identity_null(v)
    am = v.dropna(subset=["alphamissense"])
    out["null_all_variants"] = {"n_variants": null.n_variants,
                                "in_sample": null.in_sample,
                                "random_split": null.random_split,
                                "leave_gene_out": null.leave_gene_out}
    out["alphamissense_all_scored"] = {
        "n_variants": int(len(am)),
        "auroc": float(roc_auc_score(am.label, am.alphamissense)),
        "null_random_split_same_variants":
            gn.gene_identity_null(am).random_split}

    c = v.groupby("gene_symbol").label.agg(["sum", "size"])
    ev = c[(c["sum"] >= 3) & (c["size"] - c["sum"] >= 3)].index
    out["same_gene_pair_share"] = {
        "all_variants": pair_share(v),
        "genes_with_both_labels_ge3": pair_share(v[v.gene_symbol.isin(ev)]),
        "n_genes_all": int(v.gene_symbol.nunique()),
        "n_genes_evaluable": int(len(ev)),
        "fraction_of_genes_label_pure": float(
            ((c["sum"] == 0) | (c["sum"] == c["size"])).mean()),
        "fraction_of_variants_in_label_pure_genes": float(
            v.gene_symbol.isin(c[(c["sum"] == 0) | (c["sum"] == c["size"])].index).mean())}

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m16", Path(__file__).with_name("16_within_gene_ranking.py"))
    m16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m16)
    d3, _ = m16.load()
    d3 = d3.dropna(subset=["varity", "alphamissense", "gmvp"]).reset_index(drop=True)
    dec = gn.auroc_decomposition(d3, "alphamissense")
    out["three_predictor_matched"] = {"n_variants": dec["n_variants"],
                                      "n_genes": dec["n_genes"],
                                      "same_gene_pair_share": dec["same_gene_pair_share"]}

    (RES / "figure_data_extra.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
