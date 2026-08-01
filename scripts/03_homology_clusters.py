import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "rebuild"
UNIPROT = BASE / "data" / "raw" / "uniprot_human_reviewed.parquet"
MMSEQS = "~/mmseqs/bin/mmseqs"

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace('\\', '/')
    return f"/mnt/{s[0].lower()}{s[2:]}"

def wsl(cmd, timeout=3600):
    r = subprocess.run(['wsl', '--', 'bash', '-lc', cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"WSL command failed:\n{cmd}\n{r.stderr[-2000:]}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--identity', type=float, default=0.30)
    ap.add_argument('--coverage', type=float, default=0.80)
    ap.add_argument('--cov-mode', type=int, default=0,
                    help='0 = coverage of query and target (bidirectional)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=4,
                    help='cap MMseqs2 threads; it otherwise uses every core')
    ap.add_argument('--scope', default='mito_control',
                    choices=['all', 'mito_control'],
                    help='cluster every protein carrying a validated variant, '
                         'or only the mitochondrial and control sets')
    ap.add_argument('--linclust', action='store_true',
                    help='use easy-linclust (linear time, far less scratch disk); '
                         'easy-cluster is more sensitive but writes large tmp')
    args = ap.parse_args()

    if args.scope == 'all':
        src = [DATA / 'variants_all.parquet']
    else:
        src = [DATA / 'variants_mito.parquet', DATA / 'variants_control.parquet']
    accs = pd.concat([pd.read_parquet(p, columns=['uniprot_acc'])
                      for p in src if p.exists()])
    accs = sorted(accs.uniprot_acc.dropna().unique())
    log(f"{len(accs):,} proteins to cluster (scope={args.scope})")

    u = pd.read_parquet(UNIPROT)
    seqs = {a: s for a, s in zip(u.accession, u.sequence) if isinstance(s, str)}
    have = [a for a in accs if a in seqs]
    log(f"{len(have):,} with sequences")

    work = DATA / f'mmseqs{args.tag}'
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / 'proteins.fasta'
    with open(fasta, 'w', newline='\n') as f:
        for a in have:
            f.write(f">{a}\n{seqs[a]}\n")
    log(f"wrote {fasta.name}")

    version = wsl(f"{MMSEQS} version").strip()
    mode = 'easy-linclust' if args.linclust else 'easy-cluster'
    remote = f"~/mmseqs_work{args.tag or ''}"
    cmd = (f"rm -rf {remote} && mkdir -p {remote} && "
           f"cp {to_wsl(fasta)} {remote}/proteins.fasta && "
           f"cd {remote} && {MMSEQS} {mode} proteins.fasta clu tmp "
           f"--min-seq-id {args.identity} -c {args.coverage} "
           f"--cov-mode {args.cov_mode} --threads {args.threads} -v 1 && "
           f"cp clu_cluster.tsv {to_wsl(work)}/clu_cluster.tsv && "
           f"du -sh tmp && rm -rf tmp")
    log(f"running MMseqs2 {mode} in WSL ext4 (min-seq-id={args.identity}, "
        f"c={args.coverage}, cov-mode={args.cov_mode}, threads={args.threads})")
    out = wsl(cmd)
    log(f"  scratch used: {out.strip().splitlines()[-1] if out.strip() else 'n/a'}")

    tsv = work / 'clu_cluster.tsv'
    cl = pd.read_csv(tsv, sep='\t', header=None, lineterminator='\n',
                     names=['representative', 'uniprot_acc'])
    for c in ('representative', 'uniprot_acc'):
        cl[c] = cl[c].astype(str).str.strip()
    if cl.uniprot_acc.isna().any() or (cl.uniprot_acc == 'nan').any():
        raise RuntimeError("malformed cluster TSV - check FASTA line endings")
    if len(cl) != len(have):
        raise RuntimeError(f"expected {len(have)} rows in cluster TSV, got {len(cl)}")
    cl['cluster'] = pd.factorize(cl.representative)[0]
    cl = cl[['uniprot_acc', 'cluster', 'representative']]
    cl.to_parquet(DATA / f'protein_clusters{args.tag}.parquet', index=False)

    sizes = cl.cluster.value_counts()
    meta = {
        'tool': f'MMseqs2 {mode}',
        'version': version,
        'command': f'mmseqs {mode} proteins.fasta clu tmp '
                   f'--min-seq-id {args.identity} -c {args.coverage} '
                   f'--cov-mode {args.cov_mode}',
        'identity_threshold': args.identity,
        'coverage_threshold': args.coverage,
        'cov_mode': args.cov_mode,
        'threads': args.threads,
        'scope': args.scope,
        'n_proteins': int(len(cl)),
        'n_clusters': int(cl.cluster.nunique()),
        'n_singletons': int((sizes == 1).sum()),
        'largest_cluster': int(sizes.max()),
        'median_cluster_size': float(sizes.median()),
    }
    log(json.dumps(meta, indent=2))
    (DATA / f'cluster_metadata{args.tag}.json').write_text(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
