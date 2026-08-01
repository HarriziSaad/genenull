import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SCRATCH = Path(
    r"C:/Users/MSI/AppData/Local/Temp/claude/c--Users-MSI-Documents-IDP"
    r"/8ab7849a-1242-4d62-8fca-e7d143f8628f/scratchpad"
)
OUT = BASE / "data" / "rebuild"
OUT.mkdir(parents=True, exist_ok=True)

CLINVAR = SCRATCH / "variant_summary.txt.gz"
MITOCARTA = SCRATCH / "mitocarta.xls"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"

AA3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E',
    'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V',
}
AA20 = set(AA3TO1.values())

STARS = {
    'practice guideline': 4,
    'reviewed by expert panel': 3,
    'criteria provided, multiple submitters, no conflicts': 2,
    'criteria provided, single submitter': 1,
    'criteria provided, conflicting classifications': 1,
    'criteria provided, conflicting interpretations': 1,
    'no assertion criteria provided': 0,
    'no classification provided': 0,
    'no assertion provided': 0,
    'no classifications from unflagged records': 0,
    'no classification for the single variant': 0,
}

HGVS_3 = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
HGVS_1 = re.compile(r'p\.([A-Z])(\d+)([A-Z])')

def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_clinvar():
    log("reading ClinVar variant_summary")
    cols = ['GeneSymbol', 'ClinicalSignificance', 'ReviewStatus', 'Name',
            'Assembly', 'Type', 'NumberSubmitters', 'LastEvaluated']
    df = pd.read_csv(CLINVAR, sep='\t', low_memory=False, usecols=lambda c: c in cols)
    log(f"  raw rows: {len(df):,}")

    df = df[df['Assembly'].eq('GRCh38')]
    df = df[df['GeneSymbol'].notna() & df['Name'].notna()]
    log(f"  GRCh38 with gene+name: {len(df):,}")

    sig = df['ClinicalSignificance'].astype(str).str.lower()
    is_p = sig.str.contains('pathogenic') & ~sig.str.contains('benign|conflicting|uncertain')
    is_b = sig.str.contains('benign') & ~sig.str.contains('pathogenic|conflicting|uncertain')
    df = df[is_p | is_b].copy()
    df['label'] = np.where(is_p[df.index], 1, 0)
    log(f"  unambiguous P/LP or B/LB: {len(df):,}")

    wt, pos, mut = [], [], []
    for name in df['Name'].astype(str).values:
        m = HGVS_3.search(name)
        if m:
            a, p, b = m.groups()
            wt.append(AA3TO1.get(a)); pos.append(int(p)); mut.append(AA3TO1.get(b))
            continue
        m = HGVS_1.search(name)
        if m:
            a, p, b = m.groups()
            wt.append(a); pos.append(int(p)); mut.append(b)
            continue
        wt.append(None); pos.append(-1); mut.append(None)
    df['wt_aa'], df['position_1'], df['mut_aa'] = wt, pos, mut

    df = df[df['wt_aa'].isin(AA20) & df['mut_aa'].isin(AA20) & (df['position_1'] > 0)]
    df = df[df['wt_aa'] != df['mut_aa']]
    log(f"  parsed missense: {len(df):,}")

    rs = df['ReviewStatus'].astype(str).str.strip().str.lower()
    df['stars'] = rs.map(STARS)
    unmapped = rs[df['stars'].isna()].value_counts()
    if len(unmapped):
        log(f"  WARNING unmapped review statuses:\n{unmapped}")
    df['stars'] = df['stars'].fillna(0).astype(int)
    df['gene_symbol'] = df['GeneSymbol'].astype(str).str.upper().str.split(';').str[0]
    return df[['gene_symbol', 'wt_aa', 'position_1', 'mut_aa', 'label',
               'stars', 'ReviewStatus', 'ClinicalSignificance']]

def build_uniprot_index():
    log("indexing UniProt reference proteome")
    u = pd.read_parquet(UNIPROT)
    seq, g2a = {}, {}
    for acc, genes, s in zip(u.accession, u.gene_names, u.sequence):
        if not isinstance(s, str) or not s:
            continue
        seq[acc] = s
        if isinstance(genes, str):
            for g in genes.split():
                key = g.upper().strip()
                if key and key not in g2a:
                    g2a[key] = acc
    log(f"  {len(seq):,} sequences, {len(g2a):,} gene symbols")
    return seq, g2a

def validate(df, seq, g2a):
    log("validating against UniProt canonical sequences")
    acc = df['gene_symbol'].map(g2a)
    df = df.assign(uniprot_acc=acc)
    df = df[df.uniprot_acc.notna()]
    lengths = df.uniprot_acc.map(lambda a: len(seq[a]))
    ok_range = df.position_1.between(1, lengths)
    df = df[ok_range]
    ref = [seq[a][p - 1] for a, p in zip(df.uniprot_acc, df.position_1)]
    df = df.assign(ref_aa=ref)
    match = df.ref_aa == df.wt_aa
    log(f"  gene mapped: {len(df):,}   WT match: {int(match.sum()):,} "
        f"({100 * match.mean():.1f}%)")
    df = df[match].drop(columns=['ref_aa'])
    df['protein_length'] = df.uniprot_acc.map(lambda a: len(seq[a]))

    df = (df.sort_values('stars', ascending=False)
            .drop_duplicates(subset=['uniprot_acc', 'position_1', 'mut_aa'], keep='first'))
    log(f"  after de-duplication: {len(df):,}")
    return df

def load_mitocarta():
    log("loading MitoCarta 3.0")
    mc = pd.ExcelFile(MITOCARTA).parse('A Human MitoCarta3.0')
    primary = set(mc.Symbol.astype(str).str.upper())
    allsyms = set(primary)
    for s in mc.Synonyms.dropna().astype(str):
        allsyms.update(x.strip().upper() for x in s.split('|')
                       if x.strip() and x.strip() != '-')
    meta = mc.set_index(mc.Symbol.astype(str).str.upper())[
        ['MitoCarta3.0_SubMitoLocalization', 'MitoCarta3.0_MitoPathways']]
    meta = meta[~meta.index.duplicated()]
    log(f"  {len(primary)} genes, {len(allsyms)} symbols incl. synonyms")
    return primary, allsyms, meta

def matched_control(mito, nonmito, seed=42):
    log("building matched non-mitochondrial control")
    rng = np.random.default_rng(seed)
    mstat = mito.groupby('gene_symbol').agg(n=('label', 'size'), frac=('label', 'mean'))
    cstat = nonmito.groupby('gene_symbol').agg(n=('label', 'size'), frac=('label', 'mean'))

    def nbin(n):
        return int(np.clip(np.floor(np.log2(max(n, 1))), 0, 8))

    def fbin(f):
        return int(np.clip(np.floor(f * 5), 0, 4))

    pool = {}
    for g, r in cstat.iterrows():
        pool.setdefault((nbin(r.n), fbin(r.frac)), []).append(g)
    for v in pool.values():
        rng.shuffle(v)

    chosen, unmatched = [], 0
    for g, r in mstat.iterrows():
        key = (nbin(r.n), fbin(r.frac))
        if pool.get(key):
            chosen.append(pool[key].pop())
        else:
            alt = [k for k in pool if k[0] == key[0] and pool[k]]
            if alt:
                chosen.append(pool[rng.choice(len(alt)) if False else alt[0]].pop())
            else:
                unmatched += 1
    log(f"  matched {len(chosen)}/{len(mstat)} mitochondrial genes "
        f"({unmatched} without a partner)")
    return nonmito[nonmito.gene_symbol.isin(chosen)].copy()

def main():
    seq, g2a = build_uniprot_index()
    cv = load_clinvar()
    cv = validate(cv, seq, g2a)

    primary, allsyms, meta = load_mitocarta()
    cv['is_mito'] = cv.gene_symbol.isin(allsyms)
    cv['submito'] = cv.gene_symbol.map(meta['MitoCarta3.0_SubMitoLocalization'])
    cv['mitopathway'] = cv.gene_symbol.map(meta['MitoCarta3.0_MitoPathways'])

    mito = cv[cv.is_mito].copy()
    nonmito = cv[~cv.is_mito].copy()
    control = matched_control(mito, nonmito)

    log("")
    log(f"{'set':<22}{'n':>8}{'genes':>8}{'patho':>8}")
    for name, d in [('all validated', cv), ('mitochondrial', mito),
                    ('non-mito (all)', nonmito), ('non-mito (matched)', control)]:
        log(f"{name:<22}{len(d):>8,}{d.gene_symbol.nunique():>8}{d.label.mean():>8.1%}")

    log("")
    log("mitochondrial set by review-status floor:")
    for lo in (0, 1, 2, 3):
        s = mito[mito.stars >= lo]
        if not len(s):
            continue
        g = s.groupby('gene_symbol').label.agg(['size', 'mean'])
        b = g[(g['mean'] > 0) & (g['mean'] < 1)]
        log(f"  >={lo}*  n={len(s):>6,}  genes={len(g):>4}  both-class={len(b):>4}"
            f"  n_in_both={int(b['size'].sum()):>6,}  patho={s.label.mean():.1%}")

    for name, d in [('variants_all', cv), ('variants_mito', mito),
                    ('variants_control', control)]:
        d = d.copy()
        for c in ('submito', 'mitopathway', 'ReviewStatus', 'ClinicalSignificance'):
            if c in d.columns:
                d[c] = d[c].astype('string').fillna('')
        d.to_parquet(OUT / f'{name}.parquet', index=False)

    md = {
        'built_utc': datetime.now(timezone.utc).isoformat(),
        'clinvar_md5': md5(CLINVAR),
        'mitocarta_md5': md5(MITOCARTA),
        'mitocarta_genes': len(primary),
        'counts': {n: int(len(d)) for n, d in
                   [('all', cv), ('mito', mito), ('nonmito', nonmito), ('control', control)]},
        'mito_by_stars': {int(lo): int((mito.stars >= lo).sum()) for lo in (0, 1, 2, 3)},
        'inclusion_criterion': 'MitoCarta3.0 symbol or synonym match; no phenotype keywords',
    }
    (OUT / 'build_metadata.json').write_text(json.dumps(md, indent=2))
    log(f"\nwrote {OUT}")

if __name__ == '__main__':
    main()
