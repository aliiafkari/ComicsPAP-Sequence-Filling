"""Step 5: all-MiniLM-L6-v2 embeddings of the OCR text of every panel.

Run in the MAIN (torch) environment:   pip install sentence-transformers

    python scripts/05_generate_text_embeddings.py --ocr_dir D:/new_comic/ocr/ppocrv6_medium ^
        --manifest D:/new_comic/manifests/validation.jsonl D:/new_comic/manifests/test.jsonl D:/new_comic/manifests/train.jsonl ^
        --out_dir D:/new_comic/embeddings/text_ppocrv6_medium

Cache layout (what src/data/features.py reads):  <out_dir>/<sample_id>.pt =
    {"text": Tensor[9, 384], "has_text": Tensor[9], "ocr_confidence": Tensor[9]}
Panel order = manifest order: 0-3 context, 4-8 candidates 0..4 (identical to the visual cache).
A panel with no text (after filtering) gets a zero vector and has_text = 0.

Ablation knobs (change --out_dir with them so the caches do not mix):
    --min_conf 0.0|0.5|0.8   drop detections below this recognition confidence
    --order raw|reading      raw = detector order; reading = top-to-bottom, left-to-right line ordering
    --min_chars 1            drop detections shorter than this (after whitespace cleanup)
Texts are encoded once per UNIQUE string, so repeated panels cost nothing extra.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import panel_confidence, panel_text  # noqa: E402


def load_ocr(ocr_dir, split):
    path = Path(ocr_dir) / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing OCR file {path}: run 04_run_ocr.py on this split first")
    by_key = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_key[(r["sample_id"], r["panel_id"])] = r
    return by_key


def build_panel_texts(records, ocr_by_key, order, min_conf, min_chars):
    """-> [(sample_id, [texts], [confidences])] in manifest panel order; raises on missing OCR records."""
    out, missing, failed = [], 0, 0
    for rec in records:
        texts, confs = [], []
        for p in rec["context_panels"] + rec["candidates"]:
            r = ocr_by_key.get((rec["sample_id"], Path(p).stem))
            if r is None:
                missing += 1
                texts.append(""); confs.append(0.0)
                continue
            failed += int(bool(r.get("failed")))
            texts.append(panel_text(r["ocr"], order, min_conf, min_chars))
            confs.append(panel_confidence(r["ocr"], min_conf, min_chars))
        out.append((rec["sample_id"], texts, confs))
    if missing:
        raise SystemExit(f"{missing} panels have no OCR record: re-run 04_run_ocr.py for this manifest")
    if failed:
        print(f"[warn] {failed} panels had failed OCR and are treated as having no text")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--ocr_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--min_chars", type=int, default=1)
    ap.add_argument("--order", choices=["raw", "reading"], default="raw")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N samples per manifest")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- phase 1: panel texts for every split ----
    per_split = {}
    unique = {}
    for m in args.manifest:
        split = Path(m).stem
        with open(m, encoding="utf-8") as fh:
            records = [json.loads(l) for l in fh]
        if args.limit:
            records = records[: args.limit]
        items = build_panel_texts(records, load_ocr(args.ocr_dir, split), args.order, args.min_conf, args.min_chars)
        per_split[split] = items
        for _, texts, _ in items:
            for t in texts:
                if t and t not in unique:
                    unique[t] = len(unique)
        n_panels = sum(len(t) for _, t, _ in items)
        n_text = sum(1 for _, t, _ in items for x in t if x)
        print(f"[{split}] {len(items)} samples | panels with text: {n_text}/{n_panels} ({100*n_text/max(1,n_panels):.1f}%)")
    print(f"{len(unique)} unique non-empty panel texts to encode")

    # ---- phase 2: encode unique texts once ----
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length
    texts = list(unique.keys())
    emb = (model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)
           if texts else np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32"))
    emb = emb.astype("float32")
    dim = emb.shape[1]
    print(f"embedding dim = {dim}")

    # ---- phase 3: write per-sample caches ----
    for split, items in per_split.items():
        for sid, ptexts, confs in items:
            vec = np.zeros((len(ptexts), dim), dtype="float32")
            has = np.zeros(len(ptexts), dtype="float32")
            for i, t in enumerate(ptexts):
                if t:
                    vec[i] = emb[unique[t]]
                    has[i] = 1.0
            torch.save({"text": torch.from_numpy(vec), "has_text": torch.from_numpy(has),
                        "ocr_confidence": torch.tensor(confs, dtype=torch.float32)}, out / f"{sid}.pt")
        print(f"[{split}] wrote {len(items)} files")
    json.dump({k: v for k, v in vars(args).items() if k != "manifest"} | {"manifests": args.manifest, "dim": dim},
              open(out / "text_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
