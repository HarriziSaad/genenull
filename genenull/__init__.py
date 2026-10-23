"""
genenull - the gene-identity null model for ClinVar-style variant benchmarks.

Reporting this baseline should cost one function call:

    from genenull import benchmark_report
    report = benchmark_report(df, score='my_predictor', gene='gene_symbol')

See genenull/README.md for the rationale and the interpretation of each split
scheme.
"""
from .genenull import (
    NullResult,
    auroc_decomposition,
    benchmark_report,
    gene_identity_null,
    gene_offset_alignment,
    within_gene_auroc,
)

__all__ = [
    'NullResult',
    'auroc_decomposition',
    'benchmark_report',
    'gene_identity_null',
    'gene_offset_alignment',
    'within_gene_auroc',
]
__version__ = '1.1.0'
