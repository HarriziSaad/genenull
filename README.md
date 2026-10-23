# genenull

Code and data for Harrizi S, Nait Irahal I, Mostafa K, Arnoult D. *Gene-level label structure in ClinVar alters missense predictor rankings and is carried into the scores of clinically trained models.* PLOS Computational Biology.

```python
import pandas as pd
from genenull import benchmark_report
df = pd.read_parquet("data/rebuild/variants_all.parquet")
benchmark_report(df, score="my_predictor", gene="gene_symbol")
```

- `genenull/` reporting module
- `scripts/` analysis, run from the repository root in numeric order
- `figures/` figure scripts (R)
- `notebooks/` dbNSFP extraction (Colab)
- `results/` all reported results
- `data/rebuild/` ClinVar missense variant tables and homology clusters
- `prereg/` prospective analysis plan

Third-party inputs are not redistributed. Place them in `data/downloads/`: ClinVar `variant_summary.txt.gz`, MitoCarta 3.0 `mitocarta.xls`, AlphaMissense `alphamissense_aa.tsv.gz`, VARITY `varity.tar.gz`, gMVP `gmvp.csv.gz`, ProteinGym `pgym_dms.parquet` and `pgym_clinical.parquet`, UniProt `uniprot_human_sl.tsv.gz`. dbNSFP 5.3.1a scores go to `data/rebuild/dbnsfp_scores.parquet`.

```
pip install -r requirements.txt
```

MIT licence.
