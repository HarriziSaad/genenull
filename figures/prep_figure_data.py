import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"
RES = BASE / "results"
OUT = BASE / "figures" / "output" / "data"
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(BASE))
from genenull import genenull as gmod


def thin(fpr, tpr, keep=2500):
    if len(fpr) <= keep:
        return fpr, tpr
    idx = np.unique(np.r_[0, np.linspace(0, len(fpr) - 1, keep).astype(int), len(fpr) - 1])
    return fpr[idx], tpr[idx]


def roc_rows(name, y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=True)
    fpr, tpr = thin(fpr, tpr)
    return pd.DataFrame({"curve": name, "fpr": fpr, "tpr": tpr})


def main():
    sp = json.loads((RES / "split_scheme_report.json").read_text())
    fx = json.loads((RES / "figure_data_extra.json").read_text())

    d = pd.read_parquet(DATA / "variants_all.parquet")
    y = d.label.to_numpy()
    insample = d.gene_symbol.map(d.groupby("gene_symbol").label.mean()).to_numpy()
    rand = gmod._prior_random_split(d, "gene_symbol", "label", seed=42)
    a_in, a_rand = roc_auc_score(y, insample), roc_auc_score(y, rand)
    assert abs(a_in - sp["gene_prior_insample"]["auroc"]) < 1e-9, a_in
    assert abs(a_rand - sp["gene_prior_randomsplit"]["auroc"]) < 1e-9, a_rand

    s = pd.read_parquet(DATA / "variants_all_scored.parquet")
    ok = s.alphamissense.notna()
    a_am = roc_auc_score(s.label[ok], s.alphamissense[ok])
    assert abs(a_am - sp["alphamissense"]["auroc"]) < 1e-9, a_am
    roc = pd.concat([roc_rows("null_insample", y, insample),
                     roc_rows("null_random", y, rand),
                     roc_rows("alphamissense", s.label[ok].to_numpy(),
                              s.alphamissense[ok].to_numpy())])
    roc.to_csv(OUT / "roc_curves.csv", index=False)
    print(f"ROC: null in-sample {a_in:.4f}, random {a_rand:.4f}, AlphaMissense {a_am:.4f}")

    edges = np.linspace(0, 1, 26)
    rows = []
    for name, sc in (("In-sample", insample), ("Random 10-fold", rand),
                     ("Leave-gene-out", np.full(len(y), 0.5))):
        k = np.clip(np.digitize(sc, edges[1:-1]), 0, 24)
        for lab in (1, 0):
            cnt = np.bincount(k[y == lab], minlength=25)
            rows += [{"scheme": name, "label": lab, "lo": edges[i], "hi": edges[i + 1],
                      "share": cnt[i] / (y == lab).sum()} for i in range(25)]
    hist = pd.DataFrame(rows)
    assert np.allclose(hist.groupby(["scheme", "label"]).share.sum(), 1.0)
    hist.to_csv(OUT / "null_score_hist.csv", index=False)

    g = (d.groupby("gene_symbol").label.agg(p="sum", n="size")
         .assign(b=lambda t: t.n - t.p).reset_index().sort_values("gene_symbol"))
    nP, nB = int(g.p.sum()), int(g.b.sum())
    g["x0"] = (g.p.cumsum() - g.p) / nP
    g["x1"] = g.p.cumsum() / nP
    g["y0"] = (g.b.cumsum() - g.b) / nB
    g["y1"] = g.b.cumsum() / nB
    share = float((g.p * g.b).sum() / (nP * nB))
    assert abs(share - fx["same_gene_pair_share"]["all_variants"]) < 1e-12, share
    g[["gene_symbol", "p", "b", "x0", "x1", "y0", "y1"]].to_csv(OUT / "pair_blocks.csv", index=False)
    (OUT / "pair_square.json").write_text(json.dumps(
        {"n_pathogenic": nP, "n_benign": nB, "pairs": nP * nB, "same_gene_share": share,
         "n_genes": int(len(g)), "n_genes_both": int(((g.p > 0) & (g.b > 0)).sum())}, indent=2))
    print(f"pairs: {nP:,} x {nB:,} = {nP * nB:,}; same-gene share {share:.5%}")

    c = g.assign(frac=g.p / (g.p + g.b)).sort_values(["frac", "gene_symbol"],
                                                    ascending=[False, True])
    c["x1"] = c.b.cumsum() / nB
    c["x0"] = c.x1 - c.b / nB
    c["y1"] = c.p.cumsum() / nP
    c["y0"] = c.y1 - c.p / nP
    area = float(((c.x1 - c.x0) * (c.y0 + c.y1) / 2).sum())
    assert abs(area - sp["gene_prior_insample"]["auroc"]) < 1e-9, area
    c[["gene_symbol", "p", "b", "frac", "x0", "x1", "y0", "y1"]].to_csv(
        OUT / "null_chain.csv", index=False)
    pure = {"pathogenic_in_pure_pathogenic_genes": float(c.p[c.b == 0].sum() / nP),
            "benign_in_pure_benign_genes": float(c.b[c.p == 0].sum() / nB),
            "area": area}
    (OUT / "null_chain.json").write_text(json.dumps(pure, indent=2))
    print(f"chain: area {area:.4f}; pathogenic variants in all-pathogenic genes "
          f"{pure['pathogenic_in_pure_pathogenic_genes']:.1%}, benign in all-benign "
          f"{pure['benign_in_pure_benign_genes']:.1%}")

    gpr = pd.read_csv(RES / "gene_prior_robustness.csv")
    rows = []
    for k, single in zip(gpr.min_variants, gpr.single_class):
        t = g[g.n >= k]
        pp, pb = float((t.b == 0).mean()), float((t.p == 0).mean())
        assert abs(pp + pb - single) < 1e-12, (k, pp + pb, single)
        rows.append({"min_variants": int(k), "pure_pathogenic": pp, "pure_benign": pb,
                     "n_genes": int(len(t))})
    pd.DataFrame(rows).to_csv(OUT / "purity.csv", index=False)

    rows = []
    for name, f in (("MitoCarta 3.0", "variants_mito"), ("Matched control", "variants_control")):
        v = pd.read_parquet(DATA / f"{f}.parquet")
        t = v.groupby("gene_symbol").label.agg(n="size", path_frac="mean").reset_index()
        t["set"] = name
        rows.append(t)
    sets = pd.concat(rows)
    sets.to_csv(OUT / "mito_sets_genes.csv", index=False)
    print(sets.groupby("set").agg(genes=("n", "size"), variants=("n", "sum")).to_string())


if __name__ == "__main__":
    main()
