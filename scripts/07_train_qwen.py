"""Step 7: LoRA fine-tuning of Qwen2.5-VL-3B-Instruct on ComicsPAP sequence filling.

    # E2: sequence-only (images only)
    python scripts/07_train_qwen.py --out outputs/qwen_e2 --manifest_dir D:/new_comic/manifests --data_root D:/new_comic ^
        --max_pixels 802816 --train_samples 4000 --epochs 1 --grad_accum 16 --eval_every 50 --eval_samples 200

    # E8: the same + OCR text in the prompt (same flags plus):
        --ocr_dir D:/new_comic/ocr/ppocrv6_medium --ocr_min_conf 0.0 --ocr_order raw

Protocol (same as the fusion models): the last --holdout rows of train (minus a --holdout_gap buffer) are never trained on;
checkpoints are selected on an evenly spaced subset of that holdout (forced-choice scoring), and the best adapter is finally
scored on the official validation split (--decode both). Validation is never used for selection.

How it differs from the released recipe (see src/models/qwen_pap.py): loss only on the answer tokens (they also trained on the
prompt tokens), options are randomly re-ordered every time a sample is shown (--no_shuffle_options to disable; the digits drawn on
the options always match the answer), one task instead of five, and far fewer steps. Their LoRA: r=8, alpha=16, dropout 0.05,
lr 2e-4 constant, batch 128, 5000 steps on an H100 (model card). Defaults here follow the project plan: r=8, alpha=16, lr 1e-4,
cosine schedule, effective batch = --grad_accum, gradient checkpointing, bf16.

COST WARNING: every sample is one ~1.5-3 Mpixel image (~2-4k visual tokens). Measure first with --train_samples 64 --epochs 1
(prints seconds/sample and an ETA). Lower --max_pixels (e.g. 401408 = 512 tokens x 784) to trade accuracy for speed/memory.
Training is interruptible: the best and last adapters are saved at every evaluation; continue with --init_adapter <folder>.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default="data/manifests")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--init_adapter", default=None, help="continue from an adapter folder / HF id instead of a fresh LoRA")
    ap.add_argument("--seed", type=int, default=0)
    # data
    ap.add_argument("--train_samples", type=int, default=0, help="random subset of the fit part (0 = all)")
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--holdout_gap", type=int, default=200)
    ap.add_argument("--no_shuffle_options", action="store_true")
    ap.add_argument("--font", default=None)
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0, help="0 = processor default; try 802816 or 401408 on small GPUs")
    # LoRA / optimisation
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"],
                    help="decoder attention only by default; add gate_proj up_proj down_proj for more capacity")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps (overrides --epochs)")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: 4-bit base model (needs bitsandbytes). REQUIRED on 8 GB GPUs")
    ap.add_argument("--keep_vision_bf16", action="store_true", help="do not quantise the vision tower (+~1.3 GB)")
    ap.add_argument("--full_logits", action="store_true", help="compute logits for every position (default: answer positions only, saves ~1 GB)")
    # evaluation / selection
    ap.add_argument("--eval_every", type=int, default=50, help="optimizer steps between holdout evaluations")
    ap.add_argument("--eval_samples", type=int, default=200)
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without improvement (0 = off)")
    ap.add_argument("--final_decode", choices=["score", "generate", "both"], default="both")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--viz_samples", type=int, default=8, help="validation samples drawn after training (right and wrong picks)")
    ap.add_argument("--no_plots", action="store_true", help="skip training curves and prediction pictures")
    ap.add_argument("--cpu_test", action="store_true", help=argparse.SUPPRESS)   # debugging the loop on CPU with a tiny model
    # OCR (E8)
    ap.add_argument("--ocr_dir", default=None)
    ap.add_argument("--ocr_min_conf", type=float, default=0.0)
    ap.add_argument("--ocr_order", choices=["raw", "reading"], default="raw")
    ap.add_argument("--ocr_max_chars", type=int, default=200)
    args = ap.parse_args()

    import torch
    from transformers import get_cosine_schedule_with_warmup

    from src.data.pap_render import SingleImageRenderer
    from src.models import qwen_pap as Q
    try:                                  # plotting is optional: it must never block training
        import matplotlib
        matplotlib.use("Agg")
        from src.viz import pap_plots as VIZ
    except Exception as e:
        VIZ = None
        print(f"[viz] plotting disabled ({e!r})")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("GPU:", Q.gpu_info())
    if not args.cpu_test and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise SystemExit("training needs a CUDA GPU with bf16 support")

    fit = Q.load_split(args.manifest_dir, "fit", args.holdout, args.holdout_gap)
    hold = Q.subsample(Q.load_split(args.manifest_dir, "holdout", args.holdout, args.holdout_gap), args.eval_samples)
    val = Q.load_split(args.manifest_dir, "validation")
    rng = np.random.default_rng(args.seed)
    if args.train_samples and args.train_samples < len(fit):
        fit = [fit[i] for i in sorted(rng.choice(len(fit), args.train_samples, replace=False))]
    ocr_cfg = ocr_fit = ocr_hold = ocr_val = None
    if args.ocr_dir:
        ocr_cfg = dict(order=args.ocr_order, min_conf=args.ocr_min_conf, max_chars=args.ocr_max_chars)
        ids = {r["sample_id"] for r in fit} | {r["sample_id"] for r in hold}
        ocr_fit = ocr_hold = Q.load_ocr_index(args.ocr_dir, "train", ids)
        ocr_val = Q.load_ocr_index(args.ocr_dir, "validation", {r["sample_id"] for r in val})
    renderer = SingleImageRenderer(font_path=args.font)
    print(f"train {len(fit)} | holdout eval {len(hold)} | validation {len(val)} | OCR text in prompt: {bool(args.ocr_dir)}")

    model, processor = Q.load_qwen(args.model, None, args.attn, args.min_pixels or None, args.max_pixels or None, args.load_4bit,
                                   keep_vision_bf16=args.keep_vision_bf16)
    model.config.use_cache = False
    # (not peft.prepare_model_for_kbit_training: it up-casts every non-quantised weight to fp32, too costly on small GPUs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:
        model = Q.attach_adapter(model, args.init_adapter, trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=args.target_modules, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable parameters: {n_train / 1e6:.2f}M")
    if n_train == 0:
        raise SystemExit("no trainable parameters: check --target_modules / --init_adapter")

    steps_per_epoch = math.ceil(len(fit) / args.grad_accum)
    total_steps = args.max_steps or max(1, int(round(args.epochs * steps_per_epoch)))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(args.warmup_ratio * total_steps)), total_steps)
    print(f"{total_steps} optimizer steps x {args.grad_accum} samples = {total_steps * args.grad_accum} sample passes")

    def holdout_acc():
        model.eval()
        res = Q.evaluate(model, processor, hold, args.data_root, renderer, ocr_hold, ocr_cfg, "score", 1, verbose=False)
        model.train()
        ok = res["logits"].argmax(1) == res["labels"]
        return float(ok.mean()), res

    best, bad, history = -1.0, 0, []
    step_log = []
    log_fh = open(out / "train_log.jsonl", "w", encoding="utf-8")     # one JSON line per optimizer step (plot while training runs)

    def write_curves():
        """training_curves.png with an automatic health check; refreshed at every evaluation, so open it while training runs."""
        if VIZ is None or args.no_plots or not step_log:
            return None
        try:
            return VIZ.plot_qwen_history(step_log, history, out / "training_curves.png", from_adapter=bool(args.init_adapter),
                                         title=f"{out.name}: Qwen LoRA training")
        except Exception as e:
            print(f"[viz] training curves skipped: {e!r}")
            return None
    model.train()
    order = []
    t0, seen = time.time(), 0
    train_secs = eval_secs = 0.0          # kept separately: evaluation time must not inflate the per-sample training speed
    ema = None
    for step in range(1, total_steps + 1):
        t_step = time.time()
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            if not order:
                order = list(rng.permutation(len(fit)))
            rec = fit[order.pop()]
            opt_order = list(rng.permutation(len(rec["candidates"]))) if not args.no_shuffle_options else None
            image, block, label = Q.prepare(rec, args.data_root, renderer, ocr_fit, ocr_cfg, opt_order)
            enc = Q.build_train_example(processor, image, block, label, next(model.parameters()).device)
            loss = Q.answer_loss(model, enc, short=not args.full_logits) / args.grad_accum
            loss.backward()
            loss_sum += loss.detach().item()
            seen += 1
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        lr_used = opt.param_groups[0]["lr"]
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        train_secs += time.time() - t_step
        ema = loss_sum if ema is None else 0.9 * ema + 0.1 * loss_sum
        entry = {"step": step, "loss": loss_sum, "ema": ema, "lr": lr_used, "grad_norm": float(gnorm),
                 "samples_seen": seen, "train_secs": round(train_secs, 1)}
        step_log.append(entry)
        log_fh.write(json.dumps(entry) + "\n"); log_fh.flush()
        if step % args.log_every == 0 or step == total_steps:
            el = time.time() - t0
            per_sample = train_secs / seen
            per_eval = eval_secs / len(history) if history else 0.0
            n_evals_left = max(0, total_steps // args.eval_every - len(history))
            eta = per_sample * args.grad_accum * (total_steps - step) + per_eval * n_evals_left
            print(f"step {step}/{total_steps} loss {loss_sum:.4f} (ema {ema:.4f}) lr {sched.get_last_lr()[0]:.2e} "
                  f"| {per_sample:.1f}s/sample (training only)" + (f", eval {per_eval / 60:.1f}m each" if history else "")
                  + f" | elapsed {el / 60:.1f}m, ETA ~{eta / 60:.1f}m"
                  + (f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if torch.cuda.is_available() else ""), flush=True)
        if step % args.eval_every == 0 or step == total_steps:
            t_ev = time.time()
            acc, _ = holdout_acc()
            eval_secs += time.time() - t_ev
            history.append({"step": step, "loss_ema": ema, "holdout_acc": acc, "n": len(hold), "minutes": (time.time() - t0) / 60})
            json.dump(history, open(out / "history.json", "w"), indent=1)
            write_curves()
            print(f"  >> holdout acc {acc:.4f} (n={len(hold)}, chance 0.20)", flush=True)
            model.save_pretrained(out / "last_adapter")
            if acc > best:
                best, bad = acc, 0
                model.save_pretrained(out / "best_adapter")
                print("  >> new best adapter saved")
            else:
                bad += 1
                if args.patience and bad >= args.patience:
                    print("early stopping")
                    break

    log_fh.close()
    checks = write_curves()
    if checks:
        print("\ntraining health check:\n" + VIZ.format_health(checks))
        json.dump([{"level": l, "message": m} for l, m in checks], open(out / "training_health.json", "w"), indent=1)

    # ---- final: best adapter on the official validation split ----
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights
    set_peft_model_state_dict(model, load_peft_weights(str(out / "best_adapter"), device="cpu"))
    model.eval()
    model.config.use_cache = True
    res = Q.evaluate(model, processor, val, args.data_root, renderer, ocr_val, ocr_cfg, args.final_decode, 1, 64)
    summ = Q.summarize(res)
    summ.pop("_pred_generate", None)
    summ.update({"best_holdout_acc": best, "args": vars(args), "trainable_params": n_train,
                 "train_minutes": (time.time() - t0) / 60})
    json.dump(summ, open(out / "results.json", "w"), indent=1)
    if res["logits"] is not None:
        np.savez_compressed(out / "logits.npz", val_logits=res["logits"], val_labels=res["labels"], val_pos=res["pos"],
                            val_sample_ids=np.array(res["sample_ids"]))
    if VIZ is not None and not args.no_plots and res["logits"] is not None:
        try:
            VIZ.plot_predictions(val, res["logits"], args.data_root, out / "pred_viz_val", args.viz_samples, "mixed", args.seed)
        except Exception as e:
            print(f"[viz] prediction pictures skipped: {e!r}")
    print("\n=========== FINAL (validation, best adapter by holdout) ===========")
    for mode in ("score", "generate"):
        if mode in summ and "acc" in summ[mode]:
            m = summ[mode]
            print(f"{mode:<9} acc {m['acc']:.4f}  95% CI [{m['acc_ci95'][0]:.4f}, {m['acc_ci95'][1]:.4f}]  n={summ['n']}")
    print(f"best holdout acc {best:.4f} | adapter: {out / 'best_adapter'}")


if __name__ == "__main__":
    main()
