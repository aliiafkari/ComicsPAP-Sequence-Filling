"""Step 7b: evaluate Qwen2.5-VL-3B on ComicsPAP sequence filling (5 options, answers 0-4).

    # sanity check first (3 samples; prints the prompt, image size, tokens, generated text; saves the rendered composite)
    python scripts/07b_eval_qwen.py --smoke --out outputs/qwen_smoke --adapter none ^
        --manifest_dir D:/new_comic/manifests --data_root D:/new_comic

    # E0: zero-shot base model            --adapter none
    # E1: the released ComicsPAP LoRA     --adapter VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP
    # E2 / E8: your own adapter           --adapter outputs/qwen_e2/best_adapter   (E8 also needs --ocr_dir ...)
    python scripts/07b_eval_qwen.py --split validation --adapter VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP ^
        --out outputs/qwen_e1 --manifest_dir D:/new_comic/manifests --data_root D:/new_comic

Two decoding modes are reported (--decode both):
    generate  free generation + parsing of "answer: N" (what the official evaluation does; unparsable answers count as wrong)
    score     forced choice: the model's logits for the digits 0..4 right after "answer: " (never unparsable)
Outputs in --out: results.json, predictions_*.json, logits.npz (loadable by 12_visualize.py calibration), and for the test
split test_submission.json = [{"sample_id", "correct_panel_id"}] (0-based index). I could not verify the competition's exact
upload format; check the challenge page before submitting.

E1 caveats: the published number for this adapter is 62.21% on the full 262-sample validation set; this script scores the 259
clean samples of your manifest. Pass --font C:/Windows/Fonts/arial.ttf (the baselines were trained with Arial).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default="data/manifests")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "holdout", "fit", "train"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="none", help="'none', a Hugging Face id, or a local adapter folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate N evenly spaced samples (0 = all)")
    ap.add_argument("--decode", choices=["score", "generate", "both"], default="both")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0, help="0 = processor default (official setting); lower it if you run out of memory")
    ap.add_argument("--attn", default="sdpa", help="sdpa (default) or flash_attention_2 if installed")
    ap.add_argument("--load_8bit", action="store_true", help="8-bit weights (~4.5 GB; closest to bf16; slower). OK on 8 GB GPUs")
    ap.add_argument("--load_4bit", action="store_true", help="4-bit NF4 weights (~3 GB). Needed if 8-bit does not fit")
    ap.add_argument("--keep_vision_bf16", action="store_true", help="do not quantise the vision tower")
    ap.add_argument("--font", default=None)
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--holdout_gap", type=int, default=200)
    ap.add_argument("--ocr_dir", default=None, help="add the OCR text block to the prompt (E8: only for adapters trained with it)")
    ap.add_argument("--ocr_min_conf", type=float, default=0.0)
    ap.add_argument("--ocr_order", choices=["raw", "reading"], default="raw")
    ap.add_argument("--ocr_max_chars", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="3 samples, verbose, saves the rendered composite image")
    ap.add_argument("--viz_samples", type=int, default=8, help="how many prediction pictures to draw (needs --decode score or both)")
    ap.add_argument("--viz_which", default="mixed", choices=["mixed", "wrong", "correct", "confident_wrong", "uncertain", "random"])
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    from src.data.pap_render import SingleImageRenderer
    from src.models import qwen_pap as Q

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("GPU:", Q.gpu_info())
    records = Q.load_split(args.manifest_dir, args.split, args.holdout, args.holdout_gap)
    if args.smoke:
        args.limit = 3
    records = Q.subsample(records, args.limit)
    ocr_index, ocr_cfg = None, None
    if args.ocr_dir:
        ocr_index = Q.load_ocr_index(args.ocr_dir, Q.ocr_split_name(args.split), {r["sample_id"] for r in records})
        ocr_cfg = dict(order=args.ocr_order, min_conf=args.ocr_min_conf, max_chars=args.ocr_max_chars)
    renderer = SingleImageRenderer(font_path=args.font)
    print(f"{len(records)} samples from '{args.split}' | font {renderer.font_path}")

    adapter = None if args.adapter.lower() == "none" else args.adapter
    t0 = time.time()
    model, processor = Q.load_qwen(args.model, adapter, args.attn, args.min_pixels or None, args.max_pixels or None, args.load_4bit,
                                   load_8bit=args.load_8bit, keep_vision_bf16=args.keep_vision_bf16)
    model.eval()
    print(f"model ready in {time.time() - t0:.0f}s")

    if args.smoke:
        rec = records[0]
        image, block, label = Q.prepare(rec, args.data_root, renderer, ocr_index, ocr_cfg)
        image.save(out / "smoke_render.jpg", quality=92)
        text = Q.prompt_text(processor, block)
        enc = processor(text=[text], images=[image], return_tensors="pt")
        print(f"\n--- smoke test, sample {rec['sample_id']} (label {label}, missing slot {rec['missing_position']}) ---")
        print("composite image:", image.size, "| tokens:", enc["input_ids"].shape[1], "| image_grid_thw:", enc["image_grid_thw"].tolist())
        print("prompt text:\n" + text.replace("<|image_pad|>", "<|image_pad|>...")[:900])
        print(f"rendered image saved to {out / 'smoke_render.jpg'} -- open it and check it looks like two rows with digits on the options")
        print("digit token ids after 'answer: ':", Q.digit_token_ids(processor.tokenizer, 5))

    res = Q.evaluate(model, processor, records, args.data_root, renderer, ocr_index, ocr_cfg, args.decode,
                     args.batch_size, args.max_new_tokens)
    summ = Q.summarize(res)
    summ["args"] = vars(args)
    summ["seconds"] = round(time.time() - t0, 1)
    pred_gen = summ.pop("_pred_generate", None)

    pre = {"validation": "val", "holdout": "hold", "test": "test"}.get(args.split, args.split)
    if res["logits"] is not None:
        np.savez_compressed(out / "logits.npz", **{f"{pre}_logits": res["logits"], f"{pre}_labels": res["labels"],
                                                   f"{pre}_pos": res["pos"], f"{pre}_sample_ids": np.array(res["sample_ids"])})
        json.dump(dict(zip(res["sample_ids"], res["logits"].argmax(1).tolist())), open(out / "predictions_score.json", "w"), indent=1)
    if pred_gen is not None:
        json.dump(dict(zip(res["sample_ids"], pred_gen)), open(out / "predictions_generate.json", "w"), indent=1)
        json.dump([{"sample_id": s, "text": t} for s, t in zip(res["sample_ids"], res["texts"])],
                  open(out / "generated_text.json", "w"), indent=1)
    if args.split == "test":
        preds = res["logits"].argmax(1).tolist() if res["logits"] is not None else pred_gen
        json.dump([{"sample_id": s, "correct_panel_id": int(p)} for s, p in zip(res["sample_ids"], preds)],
                  open(out / "test_submission.json", "w"), indent=1)
    json.dump(summ, open(out / "results.json", "w"), indent=1)
    if not args.no_plots and res["logits"] is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from src.viz import pap_plots as VIZ
            VIZ.plot_predictions(records, res["logits"], args.data_root, out / "pred_viz", args.viz_samples, args.viz_which, 0)
        except Exception as e:
            print(f"[viz] prediction pictures skipped: {e!r}")

    print("\n=========== RESULT ===========")
    for mode in ("score", "generate"):
        if mode in summ:
            m = summ[mode]
            if "acc" in m:
                extra = f" | unparsable {100 * m['unparsed_rate']:.1f}%" if mode == "generate" else ""
                print(f"{mode:<9} acc {m['acc']:.4f}  95% CI [{m['acc_ci95'][0]:.4f}, {m['acc_ci95'][1]:.4f}]  n={summ['n']}{extra}"
                      f" | predicted-option histogram {m['pred_hist']}")
            else:
                print(f"{mode:<9} labels hidden; prediction histogram {m['pred_hist']}")
    if args.smoke and res["texts"]:
        print("\nsample generations:", [t[:80] for t in res["texts"]])
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
