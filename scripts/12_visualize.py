"""Step 12: figures for the experiments (matplotlib + numpy + pillow only; no torch needed).

    python scripts/12_visualize.py all --runs_root outputs/sweeps_siglib_clip --manifest_dir D:/new_comic/manifests ^
        --data_root D:/new_comic --ocr_dir D:/new_comic/ocr/ppocrv6_medium

Commands (run one, or `all` for every figure whose inputs exist). Figures go to <out_dir> (default <runs_root>/figures).
  curves       training curves per group (mean over seeds + thin per-seed lines)      needs --runs_root
  compare      validation / holdout accuracy per group, seed dots + per-run bootstrap CI   --runs_root [--manifest_dir]
  by_position  accuracy by missing-panel position                                      --runs_root [--manifest_dir]
  calibration  reliability diagram, selective accuracy, seed-ensemble gain             --runs_root (needs logits.npz, see 13_dump_logits.py)
  dataset      label / missing-position distributions, label x position heatmap, split sizes   --manifest_dir
  samples      context + candidates picture of individual samples (answer / prediction outlined)
               --manifest <split>.jsonl --data_root [--n 6 --seed 0 --ids A B --pred_json preds.json --only_errors]
  ocr          OCR statistics: text coverage, detections, confidence, text length, text area   --ocr_dir
  history      training curves + automatic health check of finished runs (fusion or Qwen)   --run_dir <run>  or  --runs_root <sweep>
               writes <run>/training_curves.png and prints whether the training looks healthy
  predictions  WHAT THE MODEL ACTUALLY PREDICTS: rebuilds the panel sequence with the correct option and with the model's pick,
               plus every option's probability. Needs <run>/logits.npz (08 now writes it; older runs: 13_dump_logits.py;
               Qwen runs: 07/07b write it)   --run_dir <run> --split val|hold|test --manifest_dir --data_root [--n 8 --which mixed]
  shift        LEAKAGE + DISTRIBUTION SHIFT: how many holdout / val / test panels (exact image) and texts also occur in the
               training rows, image sizes, lettering statistics                                   --ocr_dir [--holdout 2000 --holdout_gap 200]
Use --groups to keep only some runs, e.g.  --groups siglip clip   (substring match on "<features>/<config>").
"""
import argparse
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CHANCE = 0.20
N_CAND = 5
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_runs(root, patterns=None):
    """{ "feature/config": [ {seed, res, hist, dir}, ... ] } from <root>/<feature>__<config>__s<seed>/."""
    groups = defaultdict(list)
    for res in sorted(Path(root).glob("*/results.json")):
        parts = res.parent.name.split("__")
        if len(parts) != 3:
            continue
        feat, cfg, seed = parts
        name = f"{feat}/{cfg}"
        if patterns and not any(p in name for p in patterns):
            continue
        hp = res.parent / "history.json"
        groups[name].append({
            "seed": int(seed.lstrip("s")),
            "res": json.load(open(res, encoding="utf-8")),
            "hist": json.load(open(hp, encoding="utf-8")) if hp.exists() else [],
            "dir": res.parent,
        })
    return dict(sorted(groups.items()))


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def palette(names):
    return {n: plt.cm.tab10(i % 10) for i, n in enumerate(names)}


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    return p


def majority_share(manifest_dir):
    p = Path(manifest_dir) / "validation.jsonl"
    if not p.exists():
        return None
    labels = [r["label"] for r in read_jsonl(p) if r["label"] >= 0]
    return max(np.bincount(labels, minlength=N_CAND)) / len(labels) if labels else None


# --------------------------------------------------------------------------- #
# 1. training curves
# --------------------------------------------------------------------------- #
def plot_curves(groups, out_dir):
    col = palette(groups)
    panels = [("train_loss", "Train loss", "loss"), ("hold_loss", "Holdout loss", "loss"),
              ("hold_acc", "Holdout accuracy (used to pick the checkpoint)", "acc"),
              ("val_acc", "Validation accuracy (diagnostic only, never used for selection)", "acc")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title, kind) in zip(axes.ravel(), panels):
        for g, runs in groups.items():
            series = [[h[key] for h in r["hist"]] for r in runs if r["hist"]]
            if not series:
                continue
            for s in series:
                ax.plot(range(1, len(s) + 1), s, color=col[g], alpha=0.22, lw=1)
            L = min(len(s) for s in series)
            ax.plot(range(1, L + 1), np.mean([s[:L] for s in series], axis=0), color=col[g], lw=2.4, label=g)
            if key == "hold_acc":                       # mark the selected epoch of every seed
                for r in runs:
                    ep = r["res"].get("selected_epoch")
                    if ep and r["hist"] and ep <= len(r["hist"]):
                        ax.scatter(ep, r["hist"][ep - 1][key], marker="v", s=46, color=col[g], edgecolor="k", zorder=5)
        if kind == "acc":
            ax.axhline(CHANCE, color="gray", ls="--", lw=1, label="chance (0.20)")
        else:
            ax.axhline(math.log(N_CAND), color="gray", ls="--", lw=1, label="chance (ln 5)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training curves  (thin = individual seeds, thick = mean, triangle = selected epoch)", y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, "curves.png")


# --------------------------------------------------------------------------- #
# 2. model comparison
# --------------------------------------------------------------------------- #
def plot_compare(groups, out_dir, majority=None):
    col = palette(groups)
    names = sorted(groups, key=lambda g: -np.mean([r["res"]["val"]["acc"] for r in groups[g]]))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.2 * len(names) + 4), 5.2), sharey=True)
    for ax, key, title in zip(axes, ["val", "holdout"], ["Validation", "Train holdout"]):
        for i, g in enumerate(names):
            runs = groups[g]
            accs = np.array([r["res"][key]["acc"] for r in runs])
            ax.bar(i, accs.mean(), color=col[g], alpha=0.55,
                   yerr=accs.std(ddof=1) if len(accs) > 1 else None, capsize=5)
            xs = i + np.linspace(-0.2, 0.2, len(accs)) if len(accs) > 1 else np.array([float(i)])
            ax.scatter(xs, accs, color="k", s=20, zorder=4)
            for x, r in zip(xs, runs):                  # per-run bootstrap 95% CI
                lo, hi = r["res"][key]["acc_ci95"]
                ax.plot([x, x], [lo, hi], color="k", lw=0.9, alpha=0.5)
            ax.text(i, accs.mean() + 0.012 + (accs.std(ddof=1) if len(accs) > 1 else 0), f"{accs.mean():.3f}",
                    ha="center", fontsize=9, weight="bold")
        n = groups[names[0]][0]["res"][key]["n"]
        ax.axhline(CHANCE, color="gray", ls="--", lw=1, label="chance (0.20)")
        if majority and key == "val":
            ax.axhline(majority, color="gray", ls=":", lw=1.2, label=f"always-best-slot ({majority:.3f})")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(f"{title} accuracy (n={n})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("accuracy")
    lo = min(CHANCE - 0.05, min(r["res"]["val"]["acc_ci95"][0] for g in groups.values() for r in g) - 0.02)
    axes[0].set_ylim(max(0, lo), None)
    fig.suptitle("Bars = mean over seeds (±std);  dots = seeds;  vertical lines = per-run bootstrap 95% CI", y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, "compare.png")


# --------------------------------------------------------------------------- #
# 3. accuracy by missing position
# --------------------------------------------------------------------------- #
def plot_by_position(groups, out_dir, manifest_dir=None):
    col = palette(groups)
    counts = {"val": None, "holdout": None}
    if manifest_dir:
        mdir = Path(manifest_dir)
        if (mdir / "validation.jsonl").exists():
            counts["val"] = np.bincount([r["missing_position"] for r in read_jsonl(mdir / "validation.jsonl")], minlength=5)
        if (mdir / "train.jsonl").exists():
            H = next(iter(groups.values()))[0]["res"].get("split", {}).get("holdout_rows")
            if H:
                tr = read_jsonl(mdir / "train.jsonl")[-H:]
                counts["holdout"] = np.bincount([r["missing_position"] for r in tr], minlength=5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    P = 5
    for ax, key, title in zip(axes, ["val", "holdout"], ["Validation", "Train holdout"]):
        width = 0.8 / len(groups)
        for gi, (g, runs) in enumerate(groups.items()):
            M = np.array([[(r["res"][key]["acc_per_missing_pos"].get(str(p)) if r["res"][key]["acc_per_missing_pos"].get(str(p)) is not None else np.nan)
                           for p in range(P)] for r in runs], dtype=float)
            ax.bar(np.arange(P) - 0.4 + width * (gi + 0.5), np.nanmean(M, 0), width, color=col[g], label=g,
                   yerr=np.nanstd(M, 0, ddof=1) if len(runs) > 1 else None, capsize=2)
        c = counts[key]
        ax.set_xticks(range(P))
        ax.set_xticklabels([f"missing at {p}\n(n={c[p]})" if c is not None else f"missing at {p}" for p in range(P)])
        ax.axhline(CHANCE, color="gray", ls="--", lw=1)
        ax.set_title(f"{title}: accuracy by position of the missing panel")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.text(0.5, -0.02, "Validation has only ~40-60 samples per position (95% CI of about ±13 points each): "
             "use the holdout column for trends.", ha="center", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "by_position.png")


# --------------------------------------------------------------------------- #
# 4. dataset statistics
# --------------------------------------------------------------------------- #
def plot_dataset(mdir, out_dir):
    mdir = Path(mdir)
    splits = {s: read_jsonl(mdir / f"{s}.jsonl") for s in ["train", "validation", "test"] if (mdir / f"{s}.jsonl").exists()}
    cols = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]                                     # label distribution
    lab_splits = [s for s in ["train", "validation"] if s in splits]
    for k, s in enumerate(lab_splits):
        labs = [r["label"] for r in splits[s] if r["label"] >= 0]
        frac = np.bincount(labs, minlength=N_CAND) / max(1, len(labs))
        ax.bar(np.arange(N_CAND) + (k - 0.5 * (len(lab_splits) - 1)) * 0.38, frac, 0.38, color=cols[s], label=f"{s} (n={len(labs)})")
    ax.axhline(CHANCE, color="gray", ls="--", lw=1)
    ax.set_xticks(range(N_CAND)); ax.set_xlabel("correct candidate slot"); ax.set_ylabel("fraction")
    ax.set_title("Where is the right answer? (should be ~uniform)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    ax = axes[0, 1]                                     # missing position distribution
    for k, (s, rows) in enumerate(splits.items()):
        frac = np.bincount([r["missing_position"] for r in rows], minlength=N_CAND) / len(rows)
        ax.bar(np.arange(N_CAND) + (k - 1) * 0.27, frac, 0.27, color=cols[s], label=s)
    ax.set_xticks(range(N_CAND)); ax.set_xlabel("position of the missing panel in the 5-panel window")
    ax.set_title("Missing-panel position"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    ax = axes[1, 0]                                     # P(label | missing position) on train
    if "train" in splits:
        M = np.zeros((N_CAND, N_CAND))
        for r in splits["train"]:
            if r["label"] >= 0:
                M[r["missing_position"], r["label"]] += 1
        Mn = M / M.sum(1, keepdims=True).clip(1)
        im = ax.imshow(Mn, cmap="viridis", vmin=0, vmax=max(0.4, Mn.max()))
        for i in range(N_CAND):
            for j in range(N_CAND):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set_xlabel("correct slot"); ax.set_ylabel("missing position"); ax.set_xticks(range(N_CAND)); ax.set_yticks(range(N_CAND))
        ax.set_title("Train: P(correct slot | missing position)"); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 1]                                     # split sizes
    names = list(splits)
    kept = [len(splits[s]) for s in names]
    odd = [len(read_jsonl(mdir / f"{s}_nonstandard.jsonl")) if (mdir / f"{s}_nonstandard.jsonl").exists() else 0 for s in names]
    ax.bar(names, kept, color=[cols[s] for s in names])
    ax.bar(names, odd, bottom=kept, color="lightgray", hatch="//", edgecolor="gray", label="set aside (non-standard layout)")
    for i, (k, o) in enumerate(zip(kept, odd)):
        ax.text(i, k + o, f"{k}" + (f" (+{o})" if o else ""), ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(100, 3 * max(k + o for k, o in zip(kept, odd)))
    ax.set_title("Samples per split (log scale)"); ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, out_dir, "dataset.png")


# --------------------------------------------------------------------------- #
# 5. sample viewer
# --------------------------------------------------------------------------- #
def draw_border(ax, color, lw=5, ls="-"):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=color, lw=lw, ls=ls, clip_on=False))


def plot_sample(rec, data_root, out_path, pred=None):
    ctx, cands, m = rec["context_panels"], rec["candidates"], rec["missing_position"]
    ncol = max(len(ctx) + 1, len(cands))
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 7.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    ci = 0
    for slot in range(len(ctx) + 1):
        ax = axes[0, slot]
        if slot == m:
            ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=54, color="gray", transform=ax.transAxes)
            draw_border(ax, "gray", 2, "--")
            ax.set_title(f"slot {slot}: MISSING", fontsize=10)
        else:
            ax.imshow(Image.open(Path(data_root) / ctx[ci]).convert("RGB"))
            ax.set_title(f"slot {slot}  (context {ci})", fontsize=10)
            ci += 1
    label = rec.get("label", -1)
    for j, p in enumerate(cands):
        ax = axes[1, j]
        ax.imshow(Image.open(Path(data_root) / p).convert("RGB"))
        tag = chr(ord("A") + j)
        if label == j and pred == j:
            draw_border(ax, "tab:green"); ax.set_title(f"{tag}: correct & predicted", color="tab:green", fontsize=10, weight="bold")
        elif label == j:
            draw_border(ax, "tab:green"); ax.set_title(f"{tag}: correct answer", color="tab:green", fontsize=10, weight="bold")
        elif pred == j:
            draw_border(ax, "tab:red" if label >= 0 else "tab:blue")
            ax.set_title(f"{tag}: model pick" + (" (wrong)" if label >= 0 else ""), color="tab:red" if label >= 0 else "tab:blue",
                         fontsize=10, weight="bold")
        else:
            ax.set_title(tag, fontsize=10)
    fig.suptitle(f"sample {rec['sample_id']}  |  {rec.get('split', '')}  |  missing slot {m}  |  "
                 f"label {chr(ord('A') + label) if label >= 0 else 'hidden'}"
                 + (f"  |  prediction {chr(ord('A') + pred)}" if pred is not None else ""), y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def cmd_samples(args, out_dir):
    recs = read_jsonl(args.manifest)
    preds = json.load(open(args.pred_json, encoding="utf-8")) if args.pred_json else {}
    if args.ids:
        chosen = [r for r in recs if r["sample_id"] in set(args.ids)]
    else:
        pool = recs
        if args.only_errors:
            pool = [r for r in recs if r["sample_id"] in preds and r["label"] >= 0 and preds[r["sample_id"]] != r["label"]]
        rng = np.random.default_rng(args.seed)
        chosen = [pool[i] for i in rng.permutation(len(pool))[: args.n]]
    out = Path(out_dir) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        plot_sample(r, args.data_root, out / f"sample_{r['sample_id']}.png", preds.get(r["sample_id"]))
    print(f"wrote {len(chosen)} sample figures -> {out}")


# --------------------------------------------------------------------------- #
# 6. OCR statistics
# --------------------------------------------------------------------------- #
def panel_sort_key(pid):
    kind, idx = re.match(r"(context|candidate)_(\d+)", pid).groups()
    return (kind == "candidate", int(idx))


def plot_ocr(ocr_dir, out_dir):
    ocr_dir = Path(ocr_dir)
    cols = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    stats = {}
    for s in ["train", "validation", "test"]:
        p = ocr_dir / f"{s}.jsonl"
        if not p.exists():
            continue
        st = {"n": 0, "with": 0, "failed": 0, "pos": defaultdict(lambda: [0, 0]), "ndet": [], "conf": [], "chars": [], "area": []}
        for r in read_jsonl(p):
            dets = r["ocr"]
            st["n"] += 1; st["failed"] += int(bool(r.get("failed")))
            has = len(dets) > 0
            st["with"] += has
            st["pos"][r["panel_id"]][0] += has; st["pos"][r["panel_id"]][1] += 1
            st["ndet"].append(len(dets))
            st["conf"].extend(d["confidence"] for d in dets)
            if has:
                st["chars"].append(sum(len(" ".join(str(d["text"]).split())) for d in dets))
                wh = max(1, r.get("width", 1) * r.get("height", 1))
                st["area"].append(min(1.0, sum(max(0, d["bbox"][2] - d["bbox"][0]) * max(0, d["bbox"][3] - d["bbox"][1]) for d in dets) / wh))
        stats[s] = st
    if not stats:
        print(f"no OCR files in {ocr_dir}")
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    for k, (s, st) in enumerate(stats.items()):
        v = 100 * st["with"] / st["n"]
        ax.bar(k, v, color=cols[s]); ax.text(k, v + 1, f"{v:.1f}%\n({st['with']}/{st['n']})", ha="center", fontsize=9)
    ax.set_xticks(range(len(stats))); ax.set_xticklabels(list(stats)); ax.set_ylim(0, 112)
    ax.set_title("Panels with at least one text detection"); ax.set_ylabel("% of panels")
    ax = axes[0, 1]
    for s, st in stats.items():
        pids = sorted(st["pos"], key=panel_sort_key)
        ax.plot(range(len(pids)), [100 * st["pos"][p][0] / st["pos"][p][1] for p in pids], "o-", color=cols[s], label=s)
    ax.set_xticks(range(len(pids))); ax.set_xticklabels(pids, rotation=45, ha="right", fontsize=8)
    ax.set_title("Text coverage by panel role"); ax.set_ylabel("% with text"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[0, 2]
    for s, st in stats.items():
        ax.hist(np.clip(st["ndet"], 0, 12), bins=np.arange(-0.5, 13.5), density=True, histtype="step", lw=2, color=cols[s], label=s)
    ax.set_title("Detections per panel (clipped at 12)"); ax.set_xlabel("# detections"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    for s, st in stats.items():
        if st["conf"]:
            ax.hist(st["conf"], bins=40, range=(0, 1), density=True, histtype="step", lw=2, color=cols[s], label=s)
    for t in (0.5, 0.8):
        ax.axvline(t, color="gray", ls=":", lw=1)
    ax.set_title("Recognition confidence (dotted: 0.5 / 0.8 filter candidates)"); ax.set_xlabel("confidence"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for s, st in stats.items():
        if st["chars"]:
            ax.hist(np.clip(st["chars"], 0, 400), bins=40, density=True, histtype="step", lw=2, color=cols[s], label=f"{s} (median {int(np.median(st['chars']))})")
    ax.set_yscale("log"); ax.set_title("Characters per panel with text (clipped at 400)"); ax.set_xlabel("characters"); ax.legend(fontsize=8)
    ax = axes[1, 2]
    for s, st in stats.items():
        if st["area"]:
            ax.hist(st["area"], bins=40, range=(0, 1), density=True, histtype="step", lw=2, color=cols[s], label=s)
    ax.set_title("Fraction of the panel covered by text boxes"); ax.set_xlabel("sum(box area) / panel area"); ax.legend(fontsize=8)
    fig.suptitle("OCR statistics (per panel occurrence, as the model sees them)", y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, "ocr_stats.png")


# --------------------------------------------------------------------------- #
# 6b. leakage and distribution shift (uses the "hash" of each panel image stored by 04_run_ocr.py)
# --------------------------------------------------------------------------- #
def load_samples(path):
    from itertools import groupby
    return [list(g) for _, g in groupby(read_jsonl(path), key=lambda r: r["sample_id"])]


def _ptext(r):
    return " ".join(" ".join(str(d["text"]).split()) for d in r["ocr"]).lower().strip()


def _words(t):
    return re.findall(r"[a-z0-9']+", t)


def plot_shift(ocr_dir, out_dir, holdout=2000, gap=200, min_text_len=8):
    ocr_dir = Path(ocr_dir)
    S = {s: load_samples(ocr_dir / f"{s}.jsonl") for s in ["train", "validation", "test"] if (ocr_dir / f"{s}.jsonl").exists()}
    if "train" not in S:
        print("shift needs train.jsonl in --ocr_dir")
        return None
    n = len(S["train"])
    fit_name = "train (fit part)"
    groups = {fit_name: S["train"][: max(0, n - holdout - gap)]}
    if n > holdout + gap:
        groups["train holdout"] = S["train"][n - holdout:]
    for sp in ["validation", "test"]:
        if sp in S:
            groups[sp] = S[sp]
    cols = {fit_name: "tab:blue", "train holdout": "tab:cyan", "validation": "tab:orange", "test": "tab:green"}
    ref = groups[fit_name]
    ref_hash = {r["hash"] for smp in ref for r in smp}
    ref_texts = [_ptext(r) for smp in ref for r in smp]
    ref_text_set = {t for t in ref_texts if len(t) >= min_text_len}
    ref_vocab = {w for t in ref_texts for w in _words(t)}
    others = [g for g in groups if g != fit_name]

    res = {}
    for g, smps in groups.items():
        recs = [r for smp in smps for r in smp]
        uniq = {r["hash"]: r for r in recs}.values()
        texts = [_ptext(r) for r in recs]
        long_t = [t for t in texts if len(t) >= min_text_len]
        words = [w for t in texts for w in _words(t)]
        det_txt = [str(d["text"]) for r in recs for d in r["ocr"]]
        chars = "".join(det_txt)
        letters = [c for c in chars if c.isalpha()]
        shared_per_sample = [sum(r["hash"] in ref_hash for r in smp) for smp in smps]
        res[g] = {
            "n_samples": len(smps), "n_panels": len(recs),
            "heights": [r.get("height", 0) for r in uniq], "widths": [r.get("width", 0) for r in uniq],
            "img_overlap": float(np.mean([r["hash"] in ref_hash for r in recs])) if recs else float("nan"),
            "shared_per_sample": shared_per_sample,
            "text_overlap": float(np.mean([t in ref_text_set for t in long_t])) if long_t else float("nan"),
            "oov": float(np.mean([w not in ref_vocab for w in words])) if words else float("nan"),
            "upper": sum(c.isupper() for c in letters) / max(1, len(letters)),
            "nonascii": sum(ord(c) > 127 for c in chars) / max(1, len(chars)),
            "chars_per_det": len(chars) / max(1, len(det_txt)),
        }

    print(f"\nreference = {fit_name}: {len(ref)} samples, {len(ref_hash)} unique panel images, {len(ref_text_set)} distinct texts (>= {min_text_len} chars)")
    hdr = f"{'group':<18}{'samples':>8}{'img seen':>10}{'>=4/9 seen':>12}{'text seen':>11}{'word OOV':>10}{'UPPER':>8}{'nonASCII':>10}{'med h x w':>12}"
    print(hdr + "\n" + "-" * len(hdr))
    for g, r in res.items():
        k4 = np.mean([c >= 4 for c in r["shared_per_sample"]]) if r["shared_per_sample"] else float("nan")
        mh, mw = (int(np.median(r["heights"])), int(np.median(r["widths"]))) if r["heights"] else (0, 0)
        tag = "(ref)" if g == fit_name else ""
        print(f"{g:<18}{r['n_samples']:>8}{r['img_overlap']:>10.3f}{k4:>12.3f}{r['text_overlap']:>11.3f}{r['oov']:>10.3f}"
              f"{r['upper']:>8.3f}{r['nonascii']:>10.3f}{f'{mh}x{mw}':>12} {tag}")
    print("img seen = share of panels whose exact image occurs in the fit part; text seen = share of panels (with >= "
          f"{min_text_len} chars of OCR text) whose text occurs there. A holdout far above val/test means it LEAKS.")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, key, title in [(axes[0, 0], "heights", "Panel image height (unique images)"), (axes[0, 1], "widths", "Panel image width (unique images)")]:
        for g, r in res.items():
            if r[key]:
                ax.hist(np.clip(r[key], 0, 1500), bins=40, density=True, histtype="step", lw=2, color=cols[g], label=g)
        ax.set_title(title); ax.set_xlabel("pixels (clipped at 1500)"); ax.legend(fontsize=8)
    ax = axes[0, 2]
    for k, g in enumerate(others):
        v = 100 * res[g]["img_overlap"]
        ax.bar(k, v, color=cols[g]); ax.text(k, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(range(len(others))); ax.set_xticklabels(others); ax.set_ylim(0, 105)
    ax.set_title("Panels whose exact image is also in the fit part"); ax.set_ylabel("% of panels")
    ax = axes[1, 0]
    for g in others:
        sh = np.array(res[g]["shared_per_sample"])
        if len(sh):
            ks = np.arange(1, 10)
            ax.plot(ks, [100 * np.mean(sh >= k) for k in ks], "o-", color=cols[g], label=g)
    ax.set_xlabel("k"); ax.set_ylabel("% of samples"); ax.set_title("Samples with at least k of their 9 panels seen in the fit part")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[1, 1]
    w = 0.38
    for k, g in enumerate(others):
        ax.bar(k - w / 2, 100 * res[g]["text_overlap"], w, color="tab:purple", label="panel text seen in fit part" if k == 0 else None)
        ax.bar(k + w / 2, 100 * res[g]["oov"], w, color="tab:red", alpha=0.7, label="word never seen in fit part (OOV)" if k == 0 else None)
    ax.set_xticks(range(len(others))); ax.set_xticklabels(others); ax.set_ylabel("%"); ax.legend(fontsize=8)
    ax.set_title("OCR text overlap with the fit part")
    ax = axes[1, 2]
    names = list(res)
    for k, key, lab in [(-1, "upper", "uppercase share of letters"), (0, "nonascii", "non-ASCII share of characters")]:
        ax.bar(np.arange(len(names)) + 0.2 * (1 if k == 0 else -1), [100 * res[g][key] for g in names], 0.4, label=lab)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15); ax.set_ylabel("%"); ax.legend(fontsize=8)
    ax.set_title("Lettering / language proxies")
    fig.suptitle("Leakage and distribution shift (reference = rows the model is fitted on)", y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, "shift.png")


# --------------------------------------------------------------------------- #
# 7. calibration + ensemble (needs logits.npz written by 13_dump_logits.py)
# --------------------------------------------------------------------------- #
def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def reliability(conf, correct, bins=10, min_count=20):
    edges = np.linspace(CHANCE, 1.0, bins + 1)
    xs, ys, ns, ece = [], [], [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
            if m.sum() >= min_count:
                xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns, float(ece)


def plot_calibration(groups, out_dir, split="hold"):
    pre = "hold" if split == "hold" else "val"
    col = palette(groups)
    data = {}
    for g, runs in groups.items():
        seeds = []
        for r in sorted(runs, key=lambda r: r["seed"]):
            f = r["dir"] / "logits.npz"
            if f.exists():
                z = np.load(f)
                seeds.append((softmax(z[f"{pre}_logits"]), z[f"{pre}_labels"]))
        if seeds:
            data[g] = seeds
    if not data:
        print("no logits.npz found: run 13_dump_logits.py first (skipping calibration plot)")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.plot([CHANCE, 1], [CHANCE, 1], "k--", lw=1, label="perfect calibration")
    summary = {}
    for g, seeds in data.items():
        conf = np.concatenate([p.max(1) for p, _ in seeds]); corr = np.concatenate([(p.argmax(1) == y) for p, y in seeds]).astype(float)
        x, y, _, e = reliability(conf, corr)
        ax.plot(x, y, "o-", color=col[g], label=f"{g} (ECE {e:.3f})")
        summary[g] = e
    ax.set_xlabel("predicted confidence (max softmax)"); ax.set_ylabel("accuracy in bin"); ax.set_title("Reliability diagram (below the diagonal = overconfident)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[1]
    cover = np.linspace(0.05, 1.0, 20)
    for g, seeds in data.items():
        curves = []
        for p, y in seeds:
            order = np.argsort(-p.max(1)); ok = (p.argmax(1) == y)[order]
            curves.append([ok[: max(1, int(c * len(ok)))].mean() for c in cover])
        ax.plot(100 * cover, np.mean(curves, 0), color=col[g], lw=2, label=g)
    ax.axhline(CHANCE, color="gray", ls="--", lw=1)
    ax.set_xlabel("% of samples kept (most confident first)"); ax.set_ylabel("accuracy on kept samples")
    ax.set_title("Selective accuracy: does confidence tell right from wrong?"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axes[2]
    names = list(data)
    for i, g in enumerate(names):
        seeds = data[g]
        singles = np.array([(p.argmax(1) == y).mean() for p, y in seeds])
        ax.bar(i - 0.2, singles.mean(), 0.38, color=col[g], alpha=0.6, yerr=singles.std(ddof=1) if len(singles) > 1 else None, capsize=4,
               label="single run (mean ±std)" if i == 0 else None)
        ax.scatter(np.full(len(singles), i - 0.2), singles, color="k", s=14, zorder=4)
        if len(seeds) > 1:
            ens = (np.mean([p for p, _ in seeds], axis=0).argmax(1) == seeds[0][1]).mean()
            ax.bar(i + 0.2, ens, 0.38, color=col[g], label="mean-probability ensemble of seeds" if i == 0 else None, hatch="//", edgecolor="k")
            ax.text(i + 0.2, ens + 0.005, f"{ens:.3f}", ha="center", fontsize=8)
    ax.axhline(CHANCE, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title(f"Single seeds vs seed ensemble ({'holdout' if pre == 'hold' else 'validation'})"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save(fig, out_dir, f"calibration_{pre}.png")


# --------------------------------------------------------------------------- #
def cmd_history(args):
    from src.viz import pap_plots as VIZ
    runs = [Path(args.run_dir)] if args.run_dir else sorted(p.parent for p in Path(args.runs_root).glob("*/history.json"))
    if not runs:
        raise SystemExit("no runs found: pass --run_dir <run folder> or --runs_root <sweep folder>")
    for run in runs:
        res = json.load(open(run / "results.json", encoding="utf-8")) if (run / "results.json").exists() else {}
        a = res.get("args", {})
        out = run / "training_curves.png"
        if (run / "train_log.jsonl").exists():                                   # Qwen LoRA run
            steps = [json.loads(l) for l in open(run / "train_log.jsonl", encoding="utf-8") if l.strip()]
            evals = json.load(open(run / "history.json", encoding="utf-8")) if (run / "history.json").exists() else []
            checks = VIZ.plot_qwen_history(steps, evals, out, from_adapter=bool(a.get("init_adapter")), title=f"{run.name}: Qwen LoRA training")
        elif (run / "history.json").exists() and "epoch" not in json.load(open(run / "history.json", encoding="utf-8"))[0]:
            # Qwen run made before per-step logging existed: only the evaluation points were saved
            ev = json.load(open(run / "history.json", encoding="utf-8"))
            n_default = a.get("eval_samples", 0) or 40
            steps = [{"step": h["step"], "loss": h["loss_ema"], "ema": h["loss_ema"], "lr": float("nan"), "grad_norm": None} for h in ev]
            ev = [{**h, "n": h.get("n", n_default)} for h in ev]
            print(f"[note] {run.name}: older Qwen run, no per-step log: loss curve shows the moving average at evaluation points only")
            checks = VIZ.plot_qwen_history(steps, ev, out, from_adapter=bool(a.get("init_adapter")), title=f"{run.name}: Qwen LoRA training (evaluation points only)", loss_trend=False)
        elif (run / "history.json").exists():                                    # fusion run
            hist = json.load(open(run / "history.json", encoding="utf-8"))
            checks = VIZ.plot_fusion_history(hist, out, selected_epoch=res.get("selected_epoch"), n_eval=res.get("holdout", {}).get("n", 300),
                                             epochs_cap=a.get("epochs"), title=f"{run.name}: training curves")
        else:
            print(f"[skip] {run}: no history.json / train_log.jsonl")
            continue
        print(f"\n{run.name}\n{VIZ.format_health(checks)}\nwrote {out}")


def cmd_predictions(args):
    from src.viz import pap_plots as VIZ
    if not args.run_dir:
        raise SystemExit("predictions needs --run_dir <run folder with logits.npz>")
    if not args.manifest_dir or not args.data_root or args.data_root == "data":
        print("[note] pass --manifest_dir and --data_root (e.g. D:/new_comic) so the images can be found")
    run = Path(args.run_dir)
    recs, logits = VIZ.load_run_predictions(run, args.split, args.manifest_dir, args.holdout, args.holdout_gap)
    folder = {"val": "pred_viz_val", "hold": "pred_viz_holdout", "test": "pred_viz_test"}[args.split]     # same names the trainers use
    VIZ.plot_predictions(recs, logits, args.data_root, run / folder, args.n, args.which, args.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "curves", "compare", "by_position", "calibration", "dataset", "samples", "ocr", "shift", "history", "predictions"])
    ap.add_argument("--runs_root", default="outputs/sweeps")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--manifest_dir", default=None)
    ap.add_argument("--ocr_dir", default=None)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--manifest", default=None, help="samples: one manifest .jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--pred_json", default=None, help="samples: {sample_id: predicted candidate index}")
    ap.add_argument("--only_errors", action="store_true")
    ap.add_argument("--split", choices=["hold", "val", "test"], default="hold", help="calibration / predictions: which logits to use")
    ap.add_argument("--run_dir", default=None, help="history / predictions: one run folder")
    ap.add_argument("--which", default="mixed", choices=["mixed", "wrong", "correct", "confident_wrong", "uncertain", "random"],
                    help="predictions: which samples to draw")
    ap.add_argument("--holdout", type=int, default=2000, help="shift: holdout rows at the end of train")
    ap.add_argument("--holdout_gap", type=int, default=200, help="shift: rows dropped before the holdout")
    args = ap.parse_args()
    out_dir = args.out_dir or str(Path(args.runs_root) / "figures")

    need_runs = args.cmd in ("all", "curves", "compare", "by_position", "calibration")
    groups = load_runs(args.runs_root, args.groups) if need_runs else {}
    if need_runs and not groups and args.cmd != "all":
        raise SystemExit(f"no finished runs under {args.runs_root}")
    if args.cmd in ("all", "curves") and groups:
        plot_curves(groups, out_dir)
    if args.cmd in ("all", "compare") and groups:
        plot_compare(groups, out_dir, majority_share(args.manifest_dir) if args.manifest_dir else None)
    if args.cmd in ("all", "by_position") and groups:
        plot_by_position(groups, out_dir, args.manifest_dir)
    if args.cmd in ("all", "calibration") and groups:
        plot_calibration(groups, out_dir, args.split)
    if args.cmd in ("all", "dataset") and args.manifest_dir:
        plot_dataset(args.manifest_dir, out_dir)
    if args.cmd in ("all", "ocr") and args.ocr_dir and Path(args.ocr_dir).exists():
        plot_ocr(args.ocr_dir, out_dir)
    if args.cmd in ("all", "shift") and args.ocr_dir and Path(args.ocr_dir).exists():
        try:
            plot_shift(args.ocr_dir, out_dir, args.holdout, args.holdout_gap)
        except KeyError as e:                      # OCR files written without the image hash field
            if args.cmd == "shift":
                raise
            print(f"[skip] shift figure: OCR records lack field {e}")
    if args.cmd == "history":
        cmd_history(args)
    if args.cmd == "predictions":
        cmd_predictions(args)
    if args.cmd == "samples":
        if not args.manifest:
            raise SystemExit("samples needs --manifest <split>.jsonl and --data_root")
        cmd_samples(args, out_dir)
    if args.cmd == "all" and args.manifest and args.data_root:
        cmd_samples(args, out_dir)


if __name__ == "__main__":
    main()
