"""Shared plots for the training / evaluation scripts.  Place at  src/viz/pap_plots.py  (numpy + matplotlib + pillow only).

1. reconstruct_figure / plot_predictions : how a trained model behaves on real samples. For one sample it draws
       row 1  the full panel sequence rebuilt with the CORRECT option inserted at the missing slot   (blue border)
       row 2  the same sequence rebuilt with the MODEL'S PICK                                       (green = right, red = wrong)
       row 3  all options with the model's probability for each
   (the same idea as the reconstruction check on the dataset: the correct panel should make the story read naturally).
   Our manifests keep the context WITHOUT a placeholder, so the option is inserted at `missing_position`.
2. plot_fusion_history / plot_qwen_history : training curves plus an automatic checklist (fusion_health / qwen_health) that
   answers "is the training working?" (starts at chance loss, loss falls, accuracy above chance, overfitting, noise level).
3. load_run_predictions : read a run's logits.npz and align it with the manifest records.

Notebook use:   from src.viz.pap_plots import reconstruct_figure ; reconstruct_figure(rec, "D:/new_comic", probs, pred, show=True)
Scripts select a non-interactive backend themselves (matplotlib.use("Agg")) before importing this module.
"""
import json
import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

CHECK_COLORS = {"ok": "tab:green", "warn": "darkorange", "fail": "tab:red", "info": "dimgray"}
CHECK_TAGS = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "info": "INFO"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _img(root, rel):
    with Image.open(Path(root) / rel) as im:
        return im.convert("RGB")


# --------------------------------------------------------------------------- #
# 1. prediction reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_figure(rec, root, probs=None, pred=None, out_path=None, title=None, show=False):
    """rec: manifest record (context_panels, candidates, missing_position, label, sample_id).
    probs: per-option probabilities (optional); pred: index of the model's pick (optional)."""
    from matplotlib.patches import Rectangle
    plt = _plt()
    ctx = [_img(root, p) for p in rec["context_panels"]]
    opts = [_img(root, p) for p in rec["candidates"]]
    m, label, K = rec["missing_position"], rec.get("label", -1), len(opts)
    has_truth = label >= 0
    rows = int(has_truth) + int(pred is not None) + 1
    ncols = max(len(ctx) + 1, K)
    fig, axes = plt.subplots(rows, ncols, figsize=(2.6 * ncols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    def border(ax, color, lw=5):
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=color, lw=lw, clip_on=False))

    def row_label(r, text):
        axes[r, 0].text(-0.06, 0.5, text, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=10, weight="bold")

    def sequence_row(r, opt_idx, color, tag):
        seq = ctx[:m] + [opts[opt_idx]] + ctx[m:]
        for i, img in enumerate(seq):
            ax = axes[r, i]
            ax.imshow(img)
            if i == m:
                border(ax, color)
                ax.set_title(tag, color=color, fontsize=9, weight="bold")
            else:
                ax.set_title(f"panel {i}", fontsize=9)

    r = 0
    if has_truth:
        sequence_row(r, label, "tab:blue", "CORRECT option")
        row_label(r, "ground truth")
        r += 1
    if pred is not None:
        color = "tab:orange" if not has_truth else ("tab:green" if pred == label else "tab:red")
        tag = "MODEL PICK" + ("" if not has_truth else (" (right)" if pred == label else " (wrong)"))
        sequence_row(r, pred, color, tag)
        row_label(r, "model pick")
        r += 1
    for j, img in enumerate(opts):
        ax = axes[r, j]
        ax.imshow(img)
        t = f"option {j}" + (f"\np = {probs[j]:.2f}" if probs is not None else "")
        if j == label and j == pred:
            border(ax, "tab:green"); t += "\ncorrect + picked"
        elif j == label:
            border(ax, "tab:blue"); t += "\ncorrect"
        elif j == pred:
            border(ax, "tab:red" if has_truth else "tab:orange"); t += "\nmodel pick"
        ax.set_title(t, fontsize=9)
    row_label(r, "options")
    verdict = ""
    if pred is not None and has_truth:
        verdict = "  |  model is " + ("CORRECT" if pred == label else "WRONG")
    conf = f" (confidence {probs[pred]:.0%})" if (probs is not None and pred is not None) else ""
    fig.suptitle(f"{title or 'sample ' + str(rec['sample_id'])}  |  missing slot {m}{verdict}{conf}", fontsize=11)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def select_samples(labels, preds, conf, which="mixed", n=8, seed=0):
    """Indices to show. which: mixed (half right, half wrong) | wrong | correct | confident_wrong | uncertain | random."""
    rng = np.random.default_rng(seed)
    labels, preds, conf = np.asarray(labels), np.asarray(preds), np.asarray(conf)
    N = len(labels)
    idx = np.arange(N)
    if not (labels >= 0).all() or which == "random":
        return [int(i) for i in rng.permutation(N)[:n]]
    ok = preds == labels
    right, wrong = idx[ok], idx[~ok]
    if which == "correct":
        return [int(i) for i in rng.permutation(right)[:n]]
    if which == "wrong":
        return [int(i) for i in rng.permutation(wrong)[:n]]
    if which == "confident_wrong":
        return [int(i) for i in wrong[np.argsort(-conf[wrong])][:n]]
    if which == "uncertain":
        return [int(i) for i in idx[np.argsort(conf)][:n]]
    nr = n // 2
    picks = [int(i) for i in rng.permutation(right)[:nr]] + [int(i) for i in rng.permutation(wrong)[: n - nr]]
    if len(picks) < n:
        taken = set(picks)
        picks += [int(i) for i in rng.permutation(N) if int(i) not in taken][: n - len(picks)]
    return picks


def plot_predictions(records, logits, root, out_dir, n=8, which="mixed", seed=0, prefix="pred"):
    """records and logits must be aligned row by row. Writes one PNG per selected sample."""
    logits = np.asarray(logits, dtype=float)
    probs = softmax(logits)
    preds = logits.argmax(1)
    labels = np.array([r.get("label", -1) for r in records])
    idx = select_samples(labels, preds, probs.max(1), which, n, seed)
    out_dir = Path(out_dir)
    saved = []
    for i in idx:
        ok = "" if labels[i] < 0 else ("_right" if preds[i] == labels[i] else "_wrong")
        path = out_dir / f"{prefix}_{records[i]['sample_id']}{ok}.png"
        reconstruct_figure(records[i], root, probs[i], int(preds[i]), path)
        saved.append(path)
    acc = f"accuracy over all {len(records)} samples: {(preds == labels).mean():.3f}" if (labels >= 0).all() else "labels hidden"
    print(f"[viz] wrote {len(saved)} prediction figures ({which}) -> {out_dir}  ({acc})")
    return saved


_SPLIT_PRE = {"val": "val", "validation": "val", "hold": "hold", "holdout": "hold", "test": "test"}


def load_run_predictions(run_dir, split, manifest_dir, holdout=2000, gap=200):
    """(records, logits) of a finished run from <run_dir>/logits.npz, aligned with the manifest."""
    pre = _SPLIT_PRE[split]
    z = np.load(Path(run_dir) / "logits.npz", allow_pickle=True)
    if f"{pre}_logits" not in z.files:
        raise SystemExit(f"{run_dir}/logits.npz has no '{pre}_logits' (it has: {z.files}); run 13_dump_logits.py or re-evaluate")
    logits = z[f"{pre}_logits"]
    mdir = Path(manifest_dir)
    src = {"val": "validation", "hold": "train", "test": "test"}[pre]
    recs = read_jsonl(mdir / f"{src}.jsonl")
    if f"{pre}_sample_ids" in z.files:
        by_id = {r["sample_id"]: r for r in recs}
        recs = [by_id[str(s)] for s in z[f"{pre}_sample_ids"]]
    elif pre == "hold":
        recs = recs[len(recs) - len(logits):]
    if len(recs) != len(logits):
        raise SystemExit(f"{len(recs)} manifest rows but {len(logits)} logits rows: wrong --manifest_dir or --split?")
    return recs, logits


# --------------------------------------------------------------------------- #
# 2. training health checks
# --------------------------------------------------------------------------- #
def _finite(xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def fusion_health(history, n_eval=300, n_options=5, selected_epoch=None, epochs_cap=None):
    """[(level, message)] for a fusion-model run. history entries: epoch, train_loss, hold_loss, hold_acc, [val_acc]."""
    if not history:
        return [("fail", "no training history was recorded")]
    chance, ln_c = 1.0 / n_options, math.log(n_options)
    tl, hl, ha = [h["train_loss"] for h in history], [h["hold_loss"] for h in history], [h["hold_acc"] for h in history]
    out = []
    if not _finite(tl + hl + ha):
        return [("fail", "NaN or infinite values in the losses/accuracies: training diverged (lower the learning rate, check the features)")]
    if abs(tl[0] - ln_c) <= 0.15:
        out.append(("ok", f"epoch-1 train loss {tl[0]:.3f} is about ln({n_options}) = {ln_c:.3f}: the model starts at chance, as it should"))
    else:
        out.append(("info", f"epoch-1 train loss {tl[0]:.3f} differs from ln({n_options}) = {ln_c:.3f} (it is an epoch average, so fast early learning lowers it)"))
    drop = tl[0] - min(tl)
    if drop >= 0.05:
        out.append(("ok", f"train loss falls from {tl[0]:.3f} to {tl[-1]:.3f}: the model is learning the training set"))
    else:
        out.append(("fail", f"train loss barely moves ({tl[0]:.3f} -> {tl[-1]:.3f}): the model is not learning (check learning rate, features, labels)"))
    se = math.sqrt(chance * (1 - chance) / max(1, n_eval))
    best = max(ha)
    if best >= chance + 2 * se:
        out.append(("ok", f"best holdout accuracy {best:.3f} is clearly above chance ({chance:.2f} + 2 SE = {chance + 2 * se:.3f}, n={n_eval})"))
    else:
        out.append(("warn", f"best holdout accuracy {best:.3f} is not clearly above chance ({chance:.2f}; 2 SE = {2 * se:.3f}, n={n_eval})"))
    k = int(np.argmin(hl))
    rise = hl[-1] - hl[k]
    if rise > 0.15:
        out.append(("warn", f"holdout loss rose {rise:.2f} above its minimum (epoch {history[k]['epoch']}): overfitting. Harmless if holdout accuracy kept "
                            "improving (checkpoints are chosen by accuracy), but regularise if accuracy stalls"))
    else:
        out.append(("ok", "no strong overfitting: holdout loss stays close to its minimum"))
    gap = hl[-1] - tl[-1]
    out.append(("info", f"final generalisation gap (holdout loss - train loss) = {gap:+.2f}"))
    if selected_epoch is not None and epochs_cap and selected_epoch >= epochs_cap:
        out.append(("warn", f"the selected epoch ({selected_epoch}) is the last allowed one: still improving when --epochs ran out, raise it"))
    vals = [h.get("val_acc") for h in history if h.get("val_acc") is not None]
    if vals and abs(float(np.mean(vals[-3:])) - float(np.mean(ha[-3:]))) > 0.08:
        out.append(("info", "validation and holdout accuracy differ by more than 8 points (validation is small, n=259: partly noise)"))
    return out


def qwen_health(steps, evals, from_adapter=False, n_options=5, loss_trend=True):
    """[(level, message)] for a Qwen LoRA run. steps: dicts with loss, grad_norm, lr; evals: dicts with holdout_acc, n.
    loss_trend=False skips the loss-trend verdict (older runs only have moving averages at the evaluation points)."""
    if not steps:
        return [("fail", "no training steps were logged")]
    chance = 1.0 / n_options
    loss = [s["loss"] for s in steps]
    out = []
    if not _finite(loss):
        return [("fail", "NaN or infinite loss: training diverged (lower --lr; check images/labels)")]
    k = max(2, len(loss) // 5)
    if not loss_trend:
        out.append(("info", "loss trend not judged: only moving averages at the evaluation points are available for this run"))
    elif len(loss) < 10:
        out.append(("info", f"only {len(loss)} steps logged: too few to judge the loss trend"))
    else:
        a, b = np.array(loss[:k]), np.array(loss[-k:])
        first, last = float(a.mean()), float(b.mean())
        se = math.sqrt(a.var(ddof=1) / k + b.var(ddof=1) / k)          # noise of a per-step loss is large: compare with it
        fell, rose = (first - last) > 2 * se, (last - first) > 2 * se
        if from_adapter:
            if fell:
                out.append(("ok", f"loss falls from {first:.3f} to {last:.3f} (first vs last {k} steps, beyond the step-to-step noise)"))
            elif rose and last > 1.15 * first:
                out.append(("warn", f"loss went UP from {first:.3f} to {last:.3f}, beyond the noise: lr may be too high, or the data/prompt differs from what the adapter was trained on"))
            else:
                out.append(("info", f"loss is flat within the step-to-step noise ({first:.3f} -> {last:.3f}, noise +-{2 * se:.3f}): expected when continuing "
                                    "from an already trained adapter with a small budget"))
        else:
            if fell and last <= 0.9 * first:
                out.append(("ok", f"loss falls from {first:.3f} to {last:.3f}: the adapter is learning"))
            else:
                out.append(("warn", f"loss is not clearly falling ({first:.3f} -> {last:.3f}, noise +-{2 * se:.3f}): train longer, raise --lr, or check the data"))
    gn = [s.get("grad_norm") for s in steps if s.get("grad_norm") is not None]
    if gn:
        med = float(np.median(gn))
        if not _finite(gn):
            out.append(("fail", "NaN/inf gradient norm"))
        elif med < 1e-8:
            out.append(("fail", "gradients are ~0: the LoRA weights are not receiving a training signal"))
        elif max(gn) > 50 * max(med, 1e-8):
            out.append(("warn", f"gradient-norm spike ({max(gn):.2f} vs median {med:.2f}): unstable steps, consider a lower --lr"))
        else:
            out.append(("ok", f"gradients are healthy (median norm {med:.3f}, max {max(gn):.3f})"))
    if evals:
        accs = [e["holdout_acc"] for e in evals]
        n = evals[0].get("n", 0) or 1
        p = float(np.mean(accs))
        half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
        se = math.sqrt(chance * (1 - chance) / n)
        if max(accs) >= chance + 2 * se:
            out.append(("ok", f"holdout accuracy {min(accs):.3f}-{max(accs):.3f} is above chance ({chance:.2f})"))
        else:
            out.append(("warn", f"holdout accuracy {max(accs):.3f} is not clearly above chance ({chance:.2f}, n={n})"))
        if len(accs) > 1 and (max(accs) - min(accs)) < 2 * half:
            out.append(("info", f"all {len(accs)} holdout evaluations lie within the sampling noise (+-{half:.3f} at n={n}): they cannot tell checkpoints apart; "
                                "use --eval_samples 150 or more"))
    return out


def format_health(checks):
    return "\n".join(f"[{CHECK_TAGS[l]}] {m}" for l, m in checks)


def _draw_checks(ax, checks):
    ax.axis("off")
    y = 1.0
    for level, msg in checks:
        lines = textwrap.wrap(f"[{CHECK_TAGS[level]}] {msg}", 165, subsequent_indent="        ")
        for ln in lines:
            ax.text(0.0, y, ln, va="top", ha="left", fontsize=9.5, family="monospace", color=CHECK_COLORS[level], transform=ax.transAxes)
            y -= 0.115
        y -= 0.02


def _save(fig, out_png):
    plt = _plt()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 3. training curves
# --------------------------------------------------------------------------- #
def plot_fusion_history(history, out_png, selected_epoch=None, n_eval=300, n_options=5, epochs_cap=None, title=None):
    from matplotlib.gridspec import GridSpec
    plt = _plt()
    checks = fusion_health(history, n_eval, n_options, selected_epoch, epochs_cap)
    ep = np.array([h["epoch"] for h in history])
    tl, hl = np.array([h["train_loss"] for h in history]), np.array([h["hold_loss"] for h in history])
    ha = np.array([h["hold_acc"] for h in history])
    va = np.array([h["val_acc"] if h.get("val_acc") is not None else np.nan for h in history])
    vl = np.array([h["val_loss"] if h.get("val_loss") is not None else np.nan for h in history])
    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 3, height_ratios=[1, 0.55], figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, tl, "o-", color="tab:blue", label="train loss (epoch mean)")
    ax.plot(ep, hl, "o-", color="tab:orange", label="holdout loss")
    if np.isfinite(vl).any():
        ax.plot(ep, vl, "--", color="tab:red", alpha=0.5, label="validation loss (diagnostic)")
    ax.axhline(math.log(n_options), color="gray", ls=":", label=f"chance (ln {n_options})")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[0, 1])
    half = 1.96 * np.sqrt(np.clip(ha * (1 - ha), 1e-9, None) / max(1, n_eval))
    ax.fill_between(ep, ha - half, ha + half, color="tab:orange", alpha=0.15, label=f"95% band at n={n_eval}")
    ax.plot(ep, ha, "o-", color="tab:orange", label="holdout accuracy (selects the checkpoint)")
    if np.isfinite(va).any():
        ax.plot(ep, va, "--", color="tab:red", alpha=0.6, label="validation accuracy (diagnostic only)")
    ax.axhline(1.0 / n_options, color="gray", ls=":", label="chance")
    if selected_epoch:
        ax.axvline(selected_epoch, color="tab:green", ls="--", label=f"selected epoch {selected_epoch}")
    ax.set_title("Accuracy"); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ep, hl - tl, "o-", color="tab:purple")
    ax.axhline(0, color="gray", lw=1)
    ax.set_title("Generalisation gap: holdout loss - train loss (rising = overfitting)"); ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
    _draw_checks(fig.add_subplot(gs[1, :]), checks)
    fig.suptitle(title or "Training curves and health check", y=0.995, fontsize=12)
    _save(fig, out_png)
    return checks


def plot_qwen_history(steps, evals, out_png, from_adapter=False, n_options=5, title=None, loss_trend=True):
    from matplotlib.gridspec import GridSpec
    plt = _plt()
    checks = qwen_health(steps, evals, from_adapter, n_options, loss_trend)
    st = np.array([s["step"] for s in steps])
    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 3, height_ratios=[1, 0.55], figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(st, [s["loss"] for s in steps], color="tab:blue", alpha=0.35, label="loss per optimizer step")
    if "ema" in steps[0]:
        ax.plot(st, [s["ema"] for s in steps], color="tab:blue", lw=2.5, label="moving average")
    ax.set_title("Training loss (mean over the answer tokens)"); ax.set_xlabel("optimizer step"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(st, [s["lr"] for s in steps], color="tab:green")
    ax.set_title("Learning rate (left) and gradient norm (right)"); ax.set_xlabel("optimizer step"); ax.grid(alpha=0.25)
    gn = [s.get("grad_norm") for s in steps]
    if all(g is not None for g in gn):
        ax2 = ax.twinx()
        ax2.plot(st, gn, color="tab:red", alpha=0.6)
        ax2.set_yscale("log")
    ax = fig.add_subplot(gs[0, 2])
    if evals:
        es = np.array([e["step"] for e in evals])
        acc = np.array([e["holdout_acc"] for e in evals])
        n = np.array([e.get("n", 1) or 1 for e in evals])
        half = 1.96 * np.sqrt(np.clip(acc * (1 - acc), 1e-9, None) / n)
        ax.errorbar(es, acc, yerr=half, fmt="o-", color="tab:orange", capsize=4, label="holdout accuracy +- 95% (binomial)")
    ax.axhline(1.0 / n_options, color="gray", ls=":", label="chance")
    ax.set_ylim(0, 1); ax.set_title("Holdout accuracy at each evaluation"); ax.set_xlabel("optimizer step"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    _draw_checks(fig.add_subplot(gs[1, :]), checks)
    fig.suptitle(title or "Qwen LoRA training curves and health check", y=0.995, fontsize=12)
    _save(fig, out_png)
    return checks
