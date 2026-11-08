"""Download the prepared workloads into data/ (about 260 MB).

    python download.py

The archive is attached to the repository's GitHub release (tag data-v1). To install a copy you
downloaded by hand, run `python download.py --file data.zip`.
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO, TAG, ASSET = "BaranAtalar/Multi-Turn-Semantic-Caching", "data-v1", "data.zip"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
SHA256 = "ab07321cc5d5e4abc1b060bbf2b39f01717f7b18070de10b8c49a4948b31db7a"
FILES = ["states.parquet", "embeddings.npz", "judge_labels.parquet", "risk_training_pairs.parquet"]


def fetch(url, dest):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help="download the archive from this URL")
    p.add_argument("--file", type=Path, help="use an archive you already downloaded")
    args = p.parse_args()
    archive = args.file or ROOT / ASSET
    if args.file is None:
        url = args.url or URL
        try:
            fetch(url, archive)
        except urllib.error.URLError as exc:
            sys.exit(f"Could not download {url} ({exc}). Download data.zip from the repository's Releases page "
                     "and run `python download.py --file data.zip`.")
    if SHA256:
        h = hashlib.sha256()
        with open(archive, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != SHA256:
            sys.exit(f"Checksum mismatch for {archive}: the download is incomplete or not the released archive.")
    with zipfile.ZipFile(archive) as z:
        z.extractall(ROOT)
    missing = [f"data/{w}/{f}" for w in ["taskmaster3", "socrateach"] for f in FILES if not (ROOT / "data" / w / f).exists()]
    if missing:
        sys.exit(f"The archive is missing {missing}.")
    if args.file is None:
        archive.unlink()
    print("Data ready in", ROOT / "data")


if __name__ == "__main__":
    main()
