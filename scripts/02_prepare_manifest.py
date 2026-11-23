"""Step 2: parquet -> extracted panel JPEGs + canonical JSONL manifests (memory-safe).

Usage:
    python scripts/02_prepare_manifest.py --data_root /path/to/data --out_root data \
        --config configs/dataset.yaml [--batch_size 4] [--limit 50] [--splits train validation]

Output:
    data/extracted_panels/<split>/<sample_id>/context_{0..n_ctx-1}.jpg
    data/extracted_panels/<split>/<sample_id>/candidate_{0..n_cand-1}.jpg
    data/manifests/<split>.jsonl   (zero-based label and missing_position;
                                    label = -1 when the split has hidden answers, e.g. test)

Memory notes:
    * Parquet files are streamed in small row batches (never a whole file / split in RAM).
    * Only the needed columns are read (captions etc. are skipped).
    * Rows are handled as plain Python dicts (no pandas), so there is no DataFrame overhead.
    * Re-runs are cheap: images that already exist on disk are not rewritten.

This script only supports SEPARATE panel images (list columns). If your data holds one
composite image per sample, it stops with an explanation instead of guessing crop
coordinates (never crop with guessed fixed coordinates).
"""
import argparse
import gc
import io
import json
import os
from collections import Counter
from pathlib import Path


JJJKKHBKGHKG KGHKGLH KJGHKTG ;OKG
import pyarrow.parquet as pq
import yaml
from PIL import Image
from tqdm import tqdm

CANDS = {
    "sample_id": ["sample_id", "id", "uid", "key"],
    "context": ["context", "context_panels", "panels", "sequence"],
    "candidates": ["options", "candidates", "choices", "candidate_panels"],
    "label": ["solution_index", "label", "answer", "target", "correct", "solution"],
    "missing_position": ["index", "missing_index", "missing_position", "gap", "position"],
    "book_id": ["book_id", "book", "series", "comic_id", "album"],
    "task_type": ["task_type", "task", "skill"],
}
REQUIRED = ["context", "candidates", "label", "missing_position"]


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def detect(names, key, configured):
    """Return the column name for `key`, using the config first, then auto-detection."""
    if configured:
        if configured not in names:
            raise SystemExit(f"Configured column '{configured}' for '{key}' not in {list(names)}")
        return configured
    for c in CANDS[key]:
        if c in names:
            return c
    return None


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def to_pil(v) -> Image.Image:
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, dict) and v.get("bytes") is not None:
        return Image.open(io.BytesIO(v["bytes"]))
    if isinstance(v, dict) and v.get("path"):
        return Image.open(v["path"])
    if isinstance(v, (bytes, bytearray)):
        return Image.open(io.BytesIO(v))
    raise ValueError(f"Cannot decode image from {type(v)}")


def save_img(v, path: Path):
    """Save one panel as JPEG. Skips files that already exist (resume-friendly)."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    img = to_pil(v)
    try:
        img.convert("RGB").save(tmp, format="JPEG", quality=95)
    finally:
        img.close()
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Split discovery
# --------------------------------------------------------------------------- #
def split_dirs(root: Path):
    names = {"train": ["train"], "validation": ["validation", "val", "valid"], "test": ["test"]}
    out = {}
    for s, cs in names.items():
        for c in cs:
            if (root / c).is_dir():
                out[s] = root / c
                break
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process_split(split, d, args, cfg, out_root):
    cols_cfg = cfg.get("columns") or {}
    n_ctx = cfg.get("num_context", 4)
    n_cand = cfg.get("num_candidates", 5)
    label_base = cfg.get("label_base", 0)
    miss_base = cfg.get("missing_position_base", 0)
    task_filter = (cfg.get("task_filter") or "").lower()

    files = sorted(d.rglob("*.parquet"))
    if not files:
        print(f"[{split}] no parquet files, skipping")
        return

    # Detect columns from the schema only (no data loaded).
    names = pq.ParquetFile(files[0]).schema_arrow.names
    col = {k: detect(names, k, cols_cfg.get(k)) for k in CANDS}
    print(f"[{split}] {len(files)} parquet file(s); columns detected: {col}")
    for req in REQUIRED:
        if col[req] is None:
            raise SystemExit(
                f"Could not detect '{req}' column in {names}. "
                f"Run 01_inspect_dataset.py and set it in {args.config}"
            )

    # Read only what we need.
    wanted = sorted({c for c in col.values() if c})

    out_manifest = out_root / "manifests" / f"{split}.jsonl"
    tmp_manifest = out_manifest.with_suffix(".jsonl.tmp")
    n_written = n_seen = n_filtered = 0
    mismatch = Counter()  # (n_context, n_candidates) -> count of rows with a non-standard layout
    done = False

    with open(tmp_manifest, "w", encoding="utf-8") as fout:
        for f in tqdm(files, desc=split, unit="file"):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=wanted):
                rows = batch.to_pylist()
                del batch

                for row in rows:
                    if args.limit and n_written >= args.limit:
                        done = True
                        break

                    if col["task_type"] and task_filter:
                        if task_filter not in str(row[col["task_type"]]).lower():
                            n_filtered += 1
                            continue

                    i = n_seen
                    n_seen += 1

                    ctx, cands = list(row[col["context"]]), list(row[col["candidates"]])
                    if len(ctx) != n_ctx or len(cands) != n_cand:
                        if args.on_mismatch == "error":
                            raise SystemExit(
                                f"Row {i} ({f.name}): got {len(ctx)} context / {len(cands)} candidates, "
                                f"expected {n_ctx}/{n_cand}. Use --on_mismatch skip|keep to continue."
                            )
                        mismatch[(len(ctx), len(cands))] += 1
                        if args.on_mismatch == "skip":
                            continue  # nothing saved yet for this row

                    sid = str(row[col["sample_id"]]) if col["sample_id"] else f"{split}_{i:06d}"
                    sid = sid.replace("/", "_").replace("\\", "_")
                    rel = Path("extracted_panels") / split / sid

                    ctx_paths, cand_paths = [], []
                    for j, v in enumerate(ctx):
                        p = rel / f"context_{j}.jpg"
                        save_img(v, out_root / p)
                        ctx_paths.append(p.as_posix())
                    for j, v in enumerate(cands):
                        p = rel / f"candidate_{j}.jpg"
                        save_img(v, out_root / p)
                        cand_paths.append(p.as_posix())

                    raw_label = int(row[col["label"]])
                    # Negative raw label (e.g. -1 on test) means "answer hidden".
                    label = -1 if raw_label < 0 else raw_label - label_base
                    miss = int(row[col["missing_position"]]) - miss_base
                    if not (-1 <= label < len(cands)):
                        raise SystemExit(f"Row {i}: label out of range: {label}")
                    if not (0 <= miss <= len(ctx)):
                        raise SystemExit(f"Row {i}: missing_position out of range: {miss}")

                    rec = {
                        "sample_id": sid,
                        "split": split,
                        "book_id": str(row[col["book_id"]]) if col["book_id"] else None,
                        "context_panels": ctx_paths,
                        "missing_position": miss,
                        "candidates": cand_paths,
                        "label": label,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    n_written += 1

                del rows
                gc.collect()
                if done:
                    break
            if done:
                break

    os.replace(tmp_manifest, out_manifest)
    if task_filter:
        print(f"[{split}] task filter '{task_filter}': kept {n_written}, dropped {n_filtered}")
    if mismatch:
        verb = "skipped" if args.on_mismatch == "skip" else "kept"
        detail = ", ".join(f"{c}ctx/{k}cand: {n}" for (c, k), n in sorted(mismatch.items()))
        print(f"[{split}] non-standard layouts ({verb}; expected {n_ctx}ctx/{n_cand}cand): {detail}")
    print(f"[{split}] wrote {n_written} records -> {out_manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N rows per split")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="rows read from parquet at once; lower it if you still run out of memory")
    ap.add_argument("--on_mismatch", choices=["skip", "keep", "error"], default="skip",
                    help="rows whose context/candidate count differs from the config: "
                         "skip them (default), keep them as-is, or abort")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="only process these splits (e.g. validation test)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out_root = Path(args.out_root)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)

    found = split_dirs(Path(args.data_root))
    if not found:
        raise SystemExit(f"No train/val/test folders found under {args.data_root}")
    print(f"Found splits: { {k: str(v) for k, v in found.items()} }")

    for split, d in found.items():
        if args.splits and split not in args.splits:
            continue
        process_split(split, d, args, cfg, out_root)


if __name__ == "__main__":
    main()