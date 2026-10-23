import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "rebuild"

FILES = {
    DATA / "gnomad.v4.1.constraint_metrics.tsv": {
        "source": "gnomAD v4.1 gene constraint metrics",
        "url": "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
               "constraint/gnomad.v4.1.constraint_metrics.tsv",
        "used_by": "scripts/rebuild/30_gene_constraint.py, 31_offset_robustness.py"},
    DATA / "external" / "variant_summary_2023-12.txt.gz": {
        "source": "ClinVar variant_summary, December 2023 monthly archive",
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2023/"
               "variant_summary_2023-12.txt.gz",
        "used_by": "scripts/rebuild/34_temporal_split.py"},
}


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def main():
    for path, meta in FILES.items():
        md5, sha = digests(path)
        st = path.stat()
        out = dict(meta, file=path.name, bytes=st.st_size, md5=md5, sha256=sha,
                   retrieved=datetime.fromtimestamp(st.st_mtime, timezone.utc)
                   .strftime("%Y-%m-%d"))
        dst = path.with_name(path.name.split(".txt")[0].split(".tsv")[0] + "_metadata.json")
        dst.write_text(json.dumps(out, indent=2))
        print(f"{dst.name}: md5 {md5}  retrieved {out['retrieved']}")


if __name__ == "__main__":
    main()
