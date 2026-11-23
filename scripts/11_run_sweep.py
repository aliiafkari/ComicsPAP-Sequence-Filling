"""Step 11: run 08_train_fusion.py over seeds x configs x feature sets, then aggregate.

Example (PowerShell / cmd, one line):
    python scripts/11_run_sweep.py --manifest_dir D:/new_comic/manifests ^
        --features dinov2=D:/new_comic/embeddings/visual@768 ^
        --configs base reg --seeds 0 1 2 --out_root outputs/sweeps

Feature sets are given as  name=directory@dim  (dim = embedding size; 768 DINOv2 CLS, 1536 DINOv2
cls_mean, 512 CLIP ViT-B, 768 SigLIP-B). Run directories are  <out_root>/<feature>__<config>__s<seed>.

OCR experiments (after 04/05):  add  --text_dir D:/new_comic/embeddings/text_ppocrv6_medium
  --modes visual visual_text text   -> rows "<feature>", "<feature>+ocr" and "ocr-only" in the summary table.
  (the "text" mode trains once, since it does not use the visual encoder.)

Text-usage presets (only with --modes visual_text / text):  --configs base tdrop sim both
  tdrop = text embeddings + 50% text dropout, sim = candidate-to-context text-similarity features only,
  both = embeddings + similarity + dropout.   --text_tag c0.8-reading  names the OCR variant in the run folders.

Every run folder now also gets training_curves.png (+ a health check printed at the end); add --data_root D:/new_comic to also get
pred_viz_val/ and pred_viz_holdout/ (pictures of right and wrong picks).

Finished runs (results.json present) are skipped, so the sweep can be interrupted and resumed.
Re-print the table without training:  python scripts/11_run_sweep.py --aggregate_only --out_root outputs/sweeps

The reported number per run is the validation accuracy of the checkpoint SELECTED ON THE TRAIN HOLDOUT
(not the best validation epoch). Mean +/- sample std (ddof=1) is taken across seeds.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

PRESETS = {
    "base": [],
    "reg": ["--dropout", "0.3", "--weight_decay", "0.1", "--lr", "5e-5", "--layers", "2", "--patience", "8"],
    # ways of using OCR text (need a text mode; they are skipped for visual-only runs):
    "tdrop": ["--text_dropout", "0.5"],                                  # embed text, blank it for 50% of training samples
    "sim": ["--text_mode", "sim"],                                       # only candidate-to-context text similarity features
    "both": ["--text_mode", "both", "--text_dropout", "0.5"],            # embed + similarity features + text dropout
}
TEXT_PRESETS = {"tdrop", "sim", "both"}
# two-sided 95% t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def ci95(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    _, sd = mean_std(xs)
    return T95.get(n - 1, 1.96) * sd / math.sqrt(n)


def parse_feature(s):
    name, rest = s.split("=", 1)
    path, dim = rest.rsplit("@", 1)
    return name, path, int(dim)


def run_jobs(args):
    train_script = Path(__file__).resolve().parent / "08_train_fusion.py"
    out_root = Path(args.out_root)
    if any(m != "visual" for m in args.modes) and not args.text_dir:
        raise SystemExit("--modes visual_text / text need --text_dir (the 05_generate_text_embeddings.py output)")
    feats = list(map(parse_feature, args.features))
    tag = f"-{args.text_tag}" if args.text_tag else ""
    jobs = []  # (label, visual_dir, visual_dim, extra_flags, is_visual_only)
    for fname, fdir, fdim in feats:
        if "visual" in args.modes:
            jobs.append((fname, fdir, fdim, ["--no_text"], True))
        if "visual_text" in args.modes:
            jobs.append((f"{fname}+ocr{tag}", fdir, fdim, ["--text_dir", args.text_dir], False))
    if "text" in args.modes:   # OCR-only does not depend on the visual encoder: run it once
        fname, fdir, fdim = feats[0]
        jobs.append((f"ocr-only{tag}", fdir, fdim, ["--text_dir", args.text_dir, "--no_visual"], False))
    for label, fdir, fdim, flags, visual_only in jobs:
        for cfg in args.configs:
            if visual_only and cfg in TEXT_PRESETS:
                continue          # a text option makes no sense without text
            for seed in args.seeds:
                run_dir = out_root / f"{label}__{cfg}__s{seed}"
                if (run_dir / "results.json").exists() and not args.force:
                    print(f"[skip] {run_dir.name} (results.json exists)")
                    continue
                cmd = [sys.executable, str(train_script), "--seed", str(seed),
                       "--out", str(run_dir), "--manifest_dir", args.manifest_dir,
                       "--visual_dir", fdir, "--visual_dim", str(fdim),
                       "--holdout", str(args.holdout), "--shuffle_reps", str(args.shuffle_reps),
                       "--num_workers", str(args.num_workers)]
                if args.data_root:
                    cmd += ["--data_root", args.data_root]
                if args.no_plots:
                    cmd += ["--no_plots"]
                cmd += flags + PRESETS[cfg] + args.extra
                print(f"\n[run] {run_dir.name}\n      {' '.join(cmd)}")
                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    print(f"[FAILED] {run_dir.name} (exit code {rc}); continuing")


def aggregate(out_root):
    out_root = Path(out_root)
    groups = {}
    for res in sorted(out_root.glob("*/results.json")):
        parts = res.parent.name.split("__")
        if len(parts) != 3:
            continue
        feat, cfg, seedtag = parts
        r = json.load(open(res))
        groups.setdefault((feat, cfg), []).append({
            "seed": int(seedtag.lstrip("s")),
            "val_acc": r["val"]["acc"], "val_lo": r["val"]["acc_ci95"][0], "val_hi": r["val"]["acc_ci95"][1],
            "hold_acc": r["holdout"]["acc"],
            "last_epoch_val_acc": r.get("last_epoch_val_acc", float("nan")),
            "epoch": r.get("selected_epoch"),
            "agree": r.get("robustness_val", {}).get("agreement_with_identity", float("nan")),
        })
    if not groups:
        print(f"no finished runs under {out_root}")
        return
    rows = []
    fw = max(10, max(len(f) for f, _ in groups) + 2)   # wide enough for names like "siglip+ocr"
    hdr = f"{'features':<{fw}}{'config':<7}{'n':>2}  {'VAL acc mean+/-std':<20}{'t-95% CI':>9}  {'min-max':<13}{'HOLDOUT acc':<17}{'last-ep VAL':<12}{'agree':>6}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for (feat, cfg), runs in sorted(groups.items()):
        v = [x["val_acc"] for x in runs]
        h = [x["hold_acc"] for x in runs]
        l = [x["last_epoch_val_acc"] for x in runs]
        a = [x["agree"] for x in runs]
        vm, vs = mean_std(v); hm, hs = mean_std(h); lm, _ = mean_std(l); am, _ = mean_std(a)
        c = ci95(v)
        print(f"{feat:<{fw}}{cfg:<7}{len(runs):>2}  {vm:.4f} +/- {vs:.4f}   {c:>8.4f}  "
              f"{min(v):.3f}-{max(v):.3f}  {hm:.4f} +/- {hs:.4f}  {lm:.4f}      {am:.3f}")
        rows.append({"features": feat, "config": cfg, "n_seeds": len(runs), "val_acc_mean": vm, "val_acc_std": vs,
                     "val_acc_t95_halfwidth": c, "val_acc_min": min(v), "val_acc_max": max(v),
                     "holdout_acc_mean": hm, "holdout_acc_std": hs, "last_epoch_val_acc_mean": lm,
                     "shuffle_agreement_mean": am, "seeds": [x["seed"] for x in runs],
                     "per_seed_val_acc": v, "selected_epochs": [x["epoch"] for x in runs]})
    print("\nChance = 0.20 for 5 candidates. With 3 seeds the t-based CI is wide; small gaps between configs are noise.")
    json.dump(rows, open(out_root / "summary.json", "w"), indent=1)
    with open(out_root / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_root/'summary.json'} and {out_root/'summary.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default="data/manifests")
    ap.add_argument("--features", nargs="+", default=["dinov2=data/embeddings/visual@768"])
    ap.add_argument("--configs", nargs="+", choices=list(PRESETS), default=["base", "reg"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_root", default="outputs/sweeps")
    ap.add_argument("--text_dir", default=None, help="OCR text embeddings (05_generate_text_embeddings.py output)")
    ap.add_argument("--text_tag", default="", help="suffix for the OCR run names, e.g. c0.8-reading, to compare OCR variants")
    ap.add_argument("--modes", nargs="+", choices=["visual", "visual_text", "text"], default=["visual"],
                    help="visual = images only; visual_text = images + OCR; text = OCR only (run once)")
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--shuffle_reps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run even if results.json exists")
    ap.add_argument("--data_root", default=None, help="forwarded to 08: folder of the panel images -> prediction pictures in every run folder")
    ap.add_argument("--no_plots", action="store_true", help="forwarded to 08: skip curves and prediction pictures")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra args forwarded to 08_train_fusion.py (after --)")
    args = ap.parse_args()
    if not args.aggregate_only:
        run_jobs(args)
    aggregate(args.out_root)


if __name__ == "__main__":
    main()
