"""Step 4: run PP-OCRv6 over every panel (4 context + 5 candidates) of each manifest.

RUN THIS IN THE SEPARATE PADDLE ENVIRONMENT (it needs only paddleocr, pillow, tqdm - not torch).

    # 0) sanity check on one image: prints the raw result keys and the parsed detections
    python scripts/04_run_ocr.py --inspect --image D:/new_comic/extracted_panels/validation/<id>/context_0.jpg

    # 1) validation first (small), then test, then train
    python scripts/04_run_ocr.py --data_root D:/new_comic --out_dir D:/new_comic/ocr/ppocrv6_medium ^
        --manifest D:/new_comic/manifests/validation.jsonl

    # all three at once (shared cache):
    python scripts/04_run_ocr.py --data_root D:/new_comic --out_dir D:/new_comic/ocr/ppocrv6_medium ^
        --manifest D:/new_comic/manifests/validation.jsonl D:/new_comic/manifests/test.jsonl D:/new_comic/manifests/train.jsonl

Output (one JSON line per PANEL, the schema of the implementation plan):
    <out_dir>/<split>.jsonl :
    {"sample_id", "panel_id": "context_0"|"candidate_3", "image_path", "width", "height", "hash",
     "ocr": [{"text", "confidence", "bbox": [x_min, y_min, x_max, y_max]}], "failed": false}
    <out_dir>/cache.jsonl   : {"h": <md5 of image bytes>, "ocr": [...]}   (append-only)

Train samples reuse the same panels many times (sliding windows over a page), so OCR is run once
per UNIQUE image (md5 of the file bytes) and shared. The cache makes the script resumable: re-run the
same command after an interruption and only the missing images are processed. All detections are
kept (no confidence filtering here); filtering/ordering is applied later in 05 so it can be ablated.
Use a different --out_dir for each model combination (e.g. PP-OCRv6_small_det/rec).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Result parsing (PaddleOCR 3.x prediction objects -> plain dicts)
# --------------------------------------------------------------------------- #
def _to_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def parse_result(res):
    """One PaddleOCR 3.x result -> [{"text", "confidence", "bbox"}]."""
    d = getattr(res, "json", None)
    d = d if isinstance(d, dict) else res
    if isinstance(d, dict) and isinstance(d.get("res"), dict):
        d = d["res"]
    texts = _to_list(d.get("rec_texts"))
    scores = _to_list(d.get("rec_scores"))
    boxes = _to_list(d.get("rec_boxes"))
    polys = _to_list(d.get("rec_polys"))
    dets = []
    for i, t in enumerate(texts):
        if i < len(boxes):
            bb = [int(round(v)) for v in boxes[i]]
        elif i < len(polys):
            xs, ys = [p[0] for p in polys[i]], [p[1] for p in polys[i]]
            bb = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        else:
            bb = [0, 0, 0, 0]
        conf = float(scores[i]) if i < len(scores) else 0.0
        dets.append({"text": str(t), "confidence": round(conf, 4), "bbox": bb})
    return dets


def make_ocr(args):
    from paddleocr import PaddleOCR  # imported lazily so the helpers above can be tested without paddle
    kw = dict(text_detection_model_name=args.det_model,
              text_recognition_model_name=args.rec_model,
              use_doc_orientation_classify=False,
              use_doc_unwarping=False,
              use_textline_orientation=not args.no_textline_orientation)
    try:
        ocr = PaddleOCR(device=args.device, **kw)
    except (TypeError, ValueError) as e:
        print(f"[warn] PaddleOCR rejected device={args.device!r} ({e}); falling back to its default device")
        ocr = PaddleOCR(**kw)
    return ocr


def run_paths(ocr, paths):
    results = list(ocr.predict(list(paths)))
    if len(results) != len(paths):
        raise RuntimeError(f"expected {len(paths)} results, got {len(results)}")
    return [parse_result(r) for r in results]


# --------------------------------------------------------------------------- #
def md5_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def panel_list(rec):
    """[(panel_id, relative_path)] in the fixed order context_0..3, candidate_0..4."""
    return [(Path(p).stem, p) for p in rec["context_panels"] + rec["candidates"]]


def load_cache(path: Path):
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    cache[r["h"]] = r["ocr"]
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a partially written last line
    return cache


def process_manifest(manifest_path, args, ocr_fn, cache, cache_fh):
    split = Path(manifest_path).stem
    root = Path(args.data_root)
    with open(manifest_path, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh]
    if args.limit:
        records = records[: args.limit]

    panels = []  # (sample_id, panel_id, rel, hash)
    for rec in tqdm(records, desc=f"{split}: hashing panels"):
        for pid, rel in panel_list(rec):
            panels.append((rec["sample_id"], pid, rel, md5_file(root / rel)))

    todo = {}
    for _, _, rel, h in panels:
        if h not in cache and h not in todo:
            todo[h] = str(root / rel)
    n_unique = len({h for *_, h in panels})
    print(f"[{split}] {len(panels)} panels, {n_unique} unique images, {len(todo)} still to OCR "
          f"({n_unique - len(todo)} cached)")

    failed = []
    items = list(todo.items())
    for i in tqdm(range(0, len(items), args.chunk), desc=f"{split}: OCR", disable=not items):
        part = items[i:i + args.chunk]
        try:
            outs = ocr_fn([p for _, p in part])
        except Exception as e:  # fall back to one image at a time to isolate the bad file
            print(f"[warn] chunk failed ({e!r}); retrying image by image")
            outs = []
            for _, p in part:
                try:
                    outs.append(ocr_fn([p])[0])
                except Exception as e2:
                    outs.append(None)
                    failed.append((p, repr(e2)))
        for (h, _), o in zip(part, outs):
            if o is None:
                continue  # not cached on purpose: a re-run retries it
            cache[h] = o
            cache_fh.write(json.dumps({"h": h, "ocr": o}) + "\n")
        cache_fh.flush()

    out_path = Path(args.out_dir) / f"{split}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    n_with_text = n_failed = 0
    with open(tmp, "w", encoding="utf-8") as fout:
        for sid, pid, rel, h in panels:
            with Image.open(root / rel) as im:
                w, hgt = im.size
            ocr = cache.get(h)
            if ocr is None:
                n_failed += 1
            elif ocr:
                n_with_text += 1
            fout.write(json.dumps({"sample_id": sid, "panel_id": pid, "image_path": rel, "width": w,
                                   "height": hgt, "hash": h, "ocr": ocr or [], "failed": ocr is None}) + "\n")
    os.replace(tmp, out_path)
    print(f"[{split}] wrote {len(panels)} panel records -> {out_path}")
    print(f"[{split}] panels with >=1 detection: {n_with_text}/{len(panels)} ({100*n_with_text/max(1,len(panels)):.1f}%)"
          f" | failed: {n_failed}")
    for p, e in failed[:5]:
        print(f"   failed image: {p}  {e}")


def inspect(args):
    ocr = make_ocr(args)
    res = list(ocr.predict(args.image))
    print(f"{len(res)} result object(s); type = {type(res[0]).__name__}")
    j = getattr(res[0], "json", None)
    inner = j.get("res", j) if isinstance(j, dict) else res[0]
    try:
        print("keys:", list(inner.keys()))
    except Exception:
        print("could not list keys:", type(inner))
    dets = parse_result(res[0])
    print(f"{len(dets)} parsed detection(s):")
    for d in dets:
        print("  ", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", help="one or more manifest .jsonl files (splits)")
    ap.add_argument("--data_root", default="data", help="folder that the manifest panel paths are relative to")
    ap.add_argument("--out_dir", default="data/ocr/ppocrv6_medium")
    ap.add_argument("--det_model", default="PP-OCRv6_medium_det")
    ap.add_argument("--rec_model", default="PP-OCRv6_medium_rec")
    ap.add_argument("--device", default="gpu:0", help='"gpu:0" or "cpu"')
    ap.add_argument("--no_textline_orientation", action="store_true", help="skip the text-line orientation classifier")
    ap.add_argument("--chunk", type=int, default=16, help="images per predict() call")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N samples per manifest")
    ap.add_argument("--inspect", action="store_true", help="run one image and print the raw result structure")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    if args.inspect:
        if not args.image:
            raise SystemExit("--inspect needs --image")
        return inspect(args)
    if not args.manifest:
        raise SystemExit("give --manifest (or use --inspect --image ...)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.jsonl"
    cache = load_cache(cache_path)
    print(f"cache: {len(cache)} images already done ({cache_path})")
    ocr = make_ocr(args)
    ocr_fn = lambda paths: run_paths(ocr, paths)
    with open(cache_path, "a", encoding="utf-8") as cache_fh:
        for m in args.manifest:
            process_manifest(m, args, ocr_fn, cache, cache_fh)
    json.dump({"det_model": args.det_model, "rec_model": args.rec_model,
               "textline_orientation": not args.no_textline_orientation},
              open(out_dir / "ocr_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
