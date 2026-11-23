"""Score a saved adapter on the official validation split (same protocol as the end of 07_train_qwen.py)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest_dir", default="data/manifests")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--font", default=None)
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--keep_vision_bf16", action="store_true")
    ap.add_argument("--final_decode", choices=["score", "generate", "both"], default="score")
    ap.add_argument("--ocr_dir", default=None)
    ap.add_argument("--ocr_min_conf", type=float, default=0.0)
    ap.add_argument("--ocr_order", choices=["raw", "reading"], default="raw")
    ap.add_argument("--ocr_max_chars", type=int, default=200)
    args = ap.parse_args()

    from src.data.pap_render import SingleImageRenderer
    from src.models import qwen_pap as Q

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    val = Q.load_split(args.manifest_dir, "validation")
    ocr_cfg = ocr_val = None
    if args.ocr_dir:
        ocr_cfg = dict(order=args.ocr_order, min_conf=args.ocr_min_conf, max_chars=args.ocr_max_chars)
        ocr_val = Q.load_ocr_index(args.ocr_dir, "validation", {r["sample_id"] for r in val})
    renderer = SingleImageRenderer(font_path=args.font)

    model, processor = Q.load_qwen(args.model, None, args.attn, args.min_pixels or None, args.max_pixels or None,
                                   args.load_4bit, keep_vision_bf16=args.keep_vision_bf16)
    model = Q.attach_adapter(model, args.adapter, trainable=False)
    model.eval()
    model.config.use_cache = True

    res = Q.evaluate(model, processor, val, args.data_root, renderer, ocr_val, ocr_cfg, args.final_decode, 1, 64)
    summ = Q.summarize(res)
    summ.pop("_pred_generate", None)
    summ.update({"adapter": args.adapter, "args": vars(args)})
    json.dump(summ, open(out / "results.json", "w"), indent=1)
    if res["logits"] is not None:
        np.savez_compressed(out / "logits.npz", val_logits=res["logits"], val_labels=res["labels"],
                            val_pos=res["pos"], val_sample_ids=np.array(res["sample_ids"]))
    for mode in ("score", "generate"):
        if mode in summ and "acc" in summ[mode]:
            m = summ[mode]
            print(f"{mode:<9} acc {m['acc']:.4f}  95% CI [{m['acc_ci95'][0]:.4f}, {m['acc_ci95'][1]:.4f}]  n={summ['n']}")


if __name__ == "__main__":
    main()