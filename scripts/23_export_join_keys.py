from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"

KEYS = ["uniprot_acc", "gene_symbol", "position_1", "wt_aa", "mut_aa"]

def main():
    d = pd.read_parquet(DATA / "variants_all.parquet")
    out = d[KEYS].drop_duplicates().reset_index(drop=True)

    assert out.wt_aa.str.len().eq(1).all(), "wild-type residue must be one letter"
    assert out.mut_aa.str.len().eq(1).all(), "alternate residue must be one letter"
    assert out.position_1.gt(0).all(), "positions are 1-based"
    assert "label" not in out.columns, "labels must not leave the local machine"

    path = DATA / "dbnsfp_join_keys.csv.gz"
    out.to_csv(path, index=False, compression="gzip")
    size_mb = path.stat().st_size / 1e6

    print(f"wrote {path}")
    print(f"  {len(out):,} unique variants, {out.uniprot_acc.nunique():,} accessions, "
          f"{out.gene_symbol.nunique():,} genes")
    print(f"  {size_mb:.1f} MB — upload this to Colab")
    print(f"  columns: {list(out.columns)}  (no labels, by design)")

if __name__ == "__main__":
    main()
