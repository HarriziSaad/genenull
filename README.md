# Gene identity in ClinVar benchmarks

Code and result tables for *Gene identity, not variant effect, dominates ClinVar
benchmarks of missense pathogenicity predictors*.

A model scored only by the pathogenic fraction of a variant's own gene — no
sequence, no structure, no conservation — reaches AUROC 0.921 on 197,904 ClinVar
missense variants under a random 10-fold split, against 0.955–0.963 for four
current predictors. Under within-gene evaluation the predictor ranking inverts,
and across 22 dbNSFP predictors roughly half the advantage conventional
benchmarks give clinically supervised models is gene identity rather than
variant-level discrimination.

## `genenull`

The null as a single module, so reporting it costs one call.

```python
from genenull import benchmark_report
report = benchmark_report(df, score='my_predictor', gene='gene_symbol')
```

Computes the null under in-sample, random-split and leave-gene-out schemes,
reports a predictor's margin over it, and returns within-gene AUROC. Also
available as `python -m genenull TABLE` for a parquet or CSV file.

## Layout

```
genenull/     the null model, released as a standalone module
scripts/      analysis pipeline, numbered in run order
figures/      figure scripts (R, ggplot2 + patchwork)
notebooks/    dbNSFP extraction (Colab; streams the archive without storing it)
prereg/       predictor exposure classification, fixed before any score existed
results/      every table and statistic the manuscript reports
```

## Reproducing

Scripts run in numeric order and write to `results/`. Each is independent apart
from `21_headroom_control.py` and `22_specialised_rebuild.py`, which import
`15_within_gene_ranking.py` and `09_ablation.py` respectively.

Figures need no raw data — they read the shipped `results/` tables. Run from the
repository root, e.g. `Rscript figures/fig5_dbnsfp_gradient.R`; output goes to
`figures/output/`. Script number matches figure number.

Inputs are not included — all are public and none may be redistributed here:

| Source | Used for |
|---|---|
| ClinVar `variant_summary.txt.gz` (NCBI) | variants and labels |
| UniProt reviewed human proteome | canonical sequences, localisation |
| MitoCarta 3.0 (Broad) | mitochondrial gene membership |
| AlphaMissense bulk substitutions | predictor scores |
| VARITY, gMVP precomputed releases | predictor scores |
| ProteinGym (`s3://proteingym`) | DMS assays, clinical substitutions |
| dbNSFP 5.3.1a (academic licence) | the 37-predictor panel |

The ClinVar release date and MD5 are recorded in the build metadata written by
`01_build_datasets.py`. Random seeds are fixed at 42 throughout.

Python 3.14 with numpy, pandas, scikit-learn, scipy, pyarrow, Biopython, PyTorch
and transformers; R 4.5.2 with ggplot2 and patchwork; MMseqs2 for homology
clustering. See `requirements.txt`.

## Pre-registration

`prereg/predictor_exposure.json` classifies every dbNSFP predictor by its
exposure to clinical labels, from published training descriptions, before any
score was extracted. It also fixes the analysis plan, the model-family
representatives, and the score-polarity record. Every later change is a dated
amendment stating whether it preceded a result.

`24_dbnsfp_gradient.py` and `25_dbnsfp_common_intersection.py` refuse to run if
the extract contains a predictor the file does not classify.

## Licence

MIT for the code. The data sources above carry their own terms; dbNSFP academic
data is CC BY-NC-ND and is not redistributed here.
