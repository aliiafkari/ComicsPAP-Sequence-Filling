"""Step 1: inspect the parquet files BEFORE assuming any schema.

Usage:
    python scripts/01_inspect_dataset.py --data_root /path/to/data --n 3

data_root must contain train/, validation/ (or val/), test/ folders with *.parquet files.
Prints columns, dtypes, row counts and a decoded preview of the first rows
(image columns are summarised by size, not printed).
"""
import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image


def describe_value(v, depth=0):
    if isinstance(v, dict) and "bytes" in v:
        try:
            im = Image.open(io.BytesIO(v["bytes"]))
            return f"<image {im.size[0]}x{im.size[1]} {im.mode}>"
        except Exception as e:  # noqa
            return f"<dict with bytes, undecodable: {e}>"
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            seq = list(v)
        except Exception:
            return repr(v)[:80]
        inner = ", ".join(describe_value(x, depth + 1) for x in seq[:3])
        more = "..." if len(seq) > 3 else ""
        return f"list(len={len(seq)})[{inner}{more}]"
    if isinstance(v, (bytes, bytearray)):
        return f"<bytes {len(v)}>"
    return repr(v)[:80]


def find_split_dirs(root: Path):
    names = {"train": ["train"], "validation": ["validation", "val", "valid"], "test": ["test"]}
    out = {}
    for split, cands in names.items():
        for c in cands:
            if (root / c).is_dir():
                out[split] = root / c
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.data_root)
    splits = find_split_dirs(root)
    print("Found splits:", {k: str(v) for k, v in splits.items()})

    for split, d in splits.items():
        files = sorted(d.rglob("*.parquet"))
        print(f"\n=== {split}: {len(files)} parquet file(s) ===")
        if not files:
            continue
        total = 0
        for f in files:
            import pyarrow.parquet as pq
            total += pq.ParquetFile(f).metadata.num_rows
        print("total rows:", total)
        df = pd.read_parquet(files[0]).head(args.n)
        print("columns / dtypes:")
        for c in df.columns:
            print(f"  {c:25s} {df[c].dtype}")
        for i in range(len(df)):
            print(f"--- row {i} ---")
            for c in df.columns:
                print(f"  {c}: {describe_value(df.iloc[i][c])}")

    print("\nCheck specifically:")
    print(" * Is there a task-type column? (Task 1.1 = sequence filling; other tasks may be mixed in.)")
    print(" * Are context/candidate panels separate images (list of images) or one composite image?")
    print(" * Is the label zero-based or one-based? Is the missing position 0- or 1-based?")
    print(" * Is there a book/series id column? (needed to confirm book-level splits)")


if __name__ == "__main__":
    main()
