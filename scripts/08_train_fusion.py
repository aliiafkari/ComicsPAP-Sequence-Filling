"""Step 8: train the candidate-conditioned fusion model from cached features.

Visual-only baseline:
    python scripts/08_train_fusion.py --no_text --seed 0 --out outputs/fusion_visual_s0 \
        --manifest_dir D:/new_comic/manifests --visual_dir D:/new_comic/embeddings/visual --num_workers 0

Evaluation protocol (this is what makes the numbers trustworthy):
  * A contiguous block of the END of the train manifest is held out (--holdout rows) with a
    buffer (--holdout_gap rows) removed between it and the training rows, to reduce leakage from
    neighbouring samples that share panels. The holdout is used for checkpoint selection and
    early stopping. The official validation split is NEVER used for selection.
  * The reported validation accuracy is the accuracy of the selected checkpoint, with a
    bootstrap 95% CI. The last-epoch validation accuracy is also stored for comparison.
  * Candidate-shuffle robustness: predictions are recomputed under random candidate
    permutations and compared with the identity order.
  * The test split has hidden labels (-1): only predictions are written.

Note: CandidateConditionedFusion scores every candidate in its own sequence with a shared head and
has no candidate-index embedding, so it is permutation-equivariant by construction. The robustness
test should therefore report ~100% agreement; it is a sanity check here and becomes informative for
models that see all candidates together (e.g. the Qwen track).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import CachedFeatureDataset, shuffle_candidates  # noqa: E402
from src.models.fusion import CandidateConditionedFusion  # noqa: E402

try:                                    # plotting is optional: it must never block training
    import matplotlib
    matplotlib.use("Agg")
    from src.viz import pap_plots as VIZ  # noqa: E402
except Exception as _e:                 # pragma: no cover
    VIZ = None
    print(f"[viz] plotting disabled ({_e!r})")

N_CTX = 4


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def bootstrap_ci(correct, n_boot=2000, seed=0):
    """95% percentile bootstrap CI of the mean of a 0/1 correctness vector."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = correct[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.no_grad()
def collect_logits(model, dl, dev):
    model.eval()
    L, Y, P = [], [], []
    for b in dl:
        b = to_dev(b, dev)
        logits = model(b["visual"], b["text"], b["has_text"], b["missing_pos"])
        L.append(logits.float().cpu()); Y.append(b["label"].cpu()); P.append(b["missing_pos"].cpu())
    return torch.cat(L), torch.cat(Y), torch.cat(P)


def metrics_from_logits(logits, labels, pos):
    assert (labels >= 0).all(), "labels must be >= 0 (do not use this on the hidden-label test split)"
    pred = logits.argmax(1)
    correct = (pred == labels)
    top2 = (logits.topk(2, dim=1).indices == labels[:, None]).any(1)
    try:
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(labels.numpy(), pred.numpy(), average="macro"))
    except Exception:
        macro_f1 = float("nan")
    acc_pos = {}
    for p in range(N_CTX + 1):
        m = pos == p
        acc_pos[p] = float(correct[m].float().mean()) if m.any() else None
    lo, hi = bootstrap_ci(correct.numpy())
    return {
        "n": int(len(labels)),
        "acc": float(correct.float().mean()),
        "acc_ci95": [lo, hi],
        "top2": float(top2.float().mean()),
        "loss": float(F.cross_entropy(logits, labels)),
        "macro_f1": macro_f1,
        "acc_per_missing_pos": acc_pos,
    }, pred


@torch.no_grad()
def shuffled_predictions(model, dl, dev, reps, seed):
    """Predicted ORIGINAL candidate index under `reps` random candidate permutations -> [reps, N]."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    per_rep = [[] for _ in range(reps)]
    for b in dl:
        b = to_dev(b, dev)
        B, N = b["visual"].shape[:2]
        K = N - N_CTX
        for r in range(reps):
            perm = torch.stack([torch.randperm(K, generator=gen) for _ in range(B)]).to(dev)  # new slot -> old idx
            idx = torch.cat([torch.arange(N_CTX, device=dev).expand(B, N_CTX), perm + N_CTX], dim=1)

            def g(x):
                return x.gather(1, idx.view(B, N, *([1] * (x.dim() - 2))).expand_as(x))

            logits = model(g(b["visual"]), g(b["text"]), g(b["has_text"]), b["missing_pos"])
            pred_new = logits.argmax(1)
            per_rep[r].append(perm.gather(1, pred_new[:, None]).squeeze(1).cpu())
    return torch.stack([torch.cat(p) for p in per_rep])


def robustness_report(model, dl, dev, logits, labels, reps, seed):
    pred_id = logits.argmax(1)
    shuf = shuffled_predictions(model, dl, dev, reps, seed)            # [R, N], original indices
    accs = (shuf == labels[None]).float().mean(1)
    agree = (shuf == pred_id[None]).float().mean()
    hist = torch.bincount(pred_id, minlength=logits.shape[1]).float()
    hist = (hist / hist.sum()).tolist()
    return {
        "reps": reps,
        "acc_identity": float((pred_id == labels).float().mean()),
        "acc_shuffled_mean": float(accs.mean()),
        "acc_shuffled_std": float(accs.std(unbiased=False)),
        "acc_shuffled_min": float(accs.min()),
        "acc_shuffled_max": float(accs.max()),
        "agreement_with_identity": float(agree),
        "pred_slot_share_identity_order": hist,
        "max_slot_share": float(max(hist)),
    }


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    preds = []
    for b in dl:
        b = to_dev(b, dev)
        preds += model(b["visual"], b["text"], b["has_text"], b["missing_pos"]).argmax(1).tolist()
    return preds


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default="data/manifests")
    ap.add_argument("--visual_dir", default="data/embeddings/visual")
    ap.add_argument("--visual_dim", type=int, default=768, help="768 DINOv2 CLS; 1536 DINOv2 cls_mean; 512 CLIP-B; 768 SigLIP-B")
    ap.add_argument("--text_dir", default=None)
    ap.add_argument("--no_text", action="store_true")
    ap.add_argument("--no_visual", action="store_true")
    ap.add_argument("--out", default="outputs/fusion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--text_mode", choices=["embed", "sim", "both"], default="embed",
                    help="how OCR text enters the model: embed = per-panel text embedding (original); sim = only "
                         "candidate-to-context text-similarity features; both")
    ap.add_argument("--text_dropout", type=float, default=0.0,
                    help="probability (per training sample) of blanking ALL text inputs, so the model cannot rely on text")
    ap.add_argument("--no_shuffle_candidates", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=2000, help="rows from the end of train kept for model selection")
    ap.add_argument("--holdout_gap", type=int, default=200, help="rows dropped between train and holdout")
    ap.add_argument("--select_metric", choices=["acc", "loss"], default="acc")
    ap.add_argument("--shuffle_reps", type=int, default=10, help="candidate permutations for the robustness test (0 = skip)")
    ap.add_argument("--data_root", default=None, help="folder the manifest panel paths are relative to; enables the prediction pictures")
    ap.add_argument("--viz_samples", type=int, default=8, help="how many validation / holdout samples to draw (right and wrong picks)")
    ap.add_argument("--no_plots", action="store_true", help="skip training curves and prediction pictures")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.manifest_dir)
    text_dir = None if args.no_text else args.text_dir
    use_text = text_dir is not None

    def ds(split):
        return CachedFeatureDataset(mdir / f"{split}.jsonl", args.visual_dir, text_dir)

    full_train, val_ds = ds("train"), ds("validation")
    N = len(full_train)
    H, gap = args.holdout, args.holdout_gap
    if N < H + gap + 1000:
        raise SystemExit(f"train has {N} rows; too few for holdout={H} gap={gap}")
    train_idx = list(range(0, N - H - gap))
    hold_idx = list(range(N - H, N))
    split_info = {
        "train_rows": len(train_idx), "gap_rows": gap, "holdout_rows": len(hold_idx), "val_rows": len(val_ds),
        "holdout_first_sample_id": full_train.records[hold_idx[0]]["sample_id"],
        "holdout_last_sample_id": full_train.records[hold_idx[-1]]["sample_id"],
    }
    print("split:", split_info)

    train_dl = DataLoader(Subset(full_train, train_idx), batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    hold_dl = DataLoader(Subset(full_train, hold_idx), batch_size=64, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=args.num_workers)

    model = CandidateConditionedFusion(visual_dim=args.visual_dim, hidden=args.hidden, layers=args.layers,
                                       heads=args.heads, ff=args.ff, dropout=args.dropout,
                                       use_text=use_text, use_visual=not args.no_visual,
                                       text_mode=args.text_mode).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | use_text={use_text} | text_mode={args.text_mode if use_text else '-'}"
          f" | text_dropout={args.text_dropout if use_text else 0} | device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def sel_score(m):
        return m["acc"] if args.select_metric == "acc" else -m["loss"]

    best, bad, best_ep, hist = -1e9, 0, 0, []

    def write_curves():
        """history.json + training_curves.png (refreshed every epoch, so you can open it while training runs)."""
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        if VIZ is None or args.no_plots:
            return None
        try:
            return VIZ.plot_fusion_history(hist, out / "training_curves.png", selected_epoch=best_ep or None, n_eval=len(hold_idx),
                                           epochs_cap=args.epochs, title=f"{out.name}: training curves")
        except Exception as e:
            print(f"[viz] training curves skipped: {e!r}")
            return None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = nb = 0
        for b in train_dl:
            b = to_dev(b, dev)
            if use_text and args.text_dropout > 0:          # modality dropout: blank all text for some samples
                keep = (torch.rand(b["text"].shape[0], device=dev) >= args.text_dropout).float()
                b["text"] = b["text"] * keep[:, None, None]
                b["has_text"] = b["has_text"] * keep[:, None]
            if not args.no_shuffle_candidates:
                b["visual"], b["text"], b["has_text"], b["label"] = shuffle_candidates(
                    b["visual"], b["text"], b["has_text"], b["label"])
            logits = model(b["visual"], b["text"], b["has_text"], b["missing_pos"])
            loss = F.cross_entropy(logits, b["label"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        mh, _ = metrics_from_logits(*collect_logits(model, hold_dl, dev))
        mv, _ = metrics_from_logits(*collect_logits(model, val_dl, dev))   # diagnostics only, never used for selection
        hist.append({"epoch": ep, "train_loss": tot / nb, "hold_acc": mh["acc"], "hold_loss": mh["loss"],
                     "val_acc": mv["acc"], "val_loss": mv["loss"]})
        print(f"ep {ep:02d} train_loss {tot/nb:.4f} | holdout acc {mh['acc']:.4f} loss {mh['loss']:.4f} "
              f"| val acc {mv['acc']:.4f} (diagnostic)")
        stop = False
        if sel_score(mh) > best:
            best, bad, best_ep = sel_score(mh), 0, ep
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            stop = bad >= args.patience
        write_curves()
        if stop:
            print("early stopping"); break

    checks = write_curves()
    if checks:
        print("\ntraining health check:\n" + VIZ.format_health(checks))
        json.dump([{"level": l, "message": m} for l, m in checks], open(out / "training_health.json", "w"), indent=1)
    last_epoch_val_acc = hist[-1]["val_acc"]

    # ---- final evaluation of the checkpoint chosen on the holdout ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    h_logits, h_labels, h_pos = collect_logits(model, hold_dl, dev)
    v_logits, v_labels, v_pos = collect_logits(model, val_dl, dev)
    result = {"args": vars(args), "split": split_info, "selected_epoch": best_ep,
              "selection_metric": args.select_metric, "last_epoch_val_acc": last_epoch_val_acc}
    result["holdout"], _ = metrics_from_logits(h_logits, h_labels, h_pos)
    result["val"], _ = metrics_from_logits(v_logits, v_labels, v_pos)
    if args.shuffle_reps > 0:
        result["robustness_val"] = robustness_report(model, val_dl, dev, v_logits, v_labels, args.shuffle_reps, args.seed)
        result["robustness_holdout"] = robustness_report(model, hold_dl, dev, h_logits, h_labels, args.shuffle_reps, args.seed)

    mv, mh = result["val"], result["holdout"]
    print(f"\nselected epoch {best_ep} ({args.select_metric} on holdout)")
    print(f"HOLDOUT  acc {mh['acc']:.4f}  95% CI [{mh['acc_ci95'][0]:.4f}, {mh['acc_ci95'][1]:.4f}]  n={mh['n']}")
    print(f"VAL      acc {mv['acc']:.4f}  95% CI [{mv['acc_ci95'][0]:.4f}, {mv['acc_ci95'][1]:.4f}]  n={mv['n']}  "
          f"(last-epoch val acc {last_epoch_val_acc:.4f})")
    if "robustness_val" in result:
        r = result["robustness_val"]
        print(f"SHUFFLE  val acc {r['acc_shuffled_mean']:.4f} +/- {r['acc_shuffled_std']:.4f} over {r['reps']} perms | "
              f"agreement with identity order {r['agreement_with_identity']:.4f} | max slot share {r['max_slot_share']:.3f}")
    json.dump(result, open(out / "results.json", "w"), indent=1)

    # ---- test: hidden labels -> predictions only ----
    test_manifest = mdir / "test.jsonl"
    arrays = dict(hold_logits=h_logits.numpy(), hold_labels=h_labels.numpy(), hold_pos=h_pos.numpy(),
                  hold_sample_ids=np.array([full_train.records[i]["sample_id"] for i in hold_idx]),
                  val_logits=v_logits.numpy(), val_labels=v_labels.numpy(), val_pos=v_pos.numpy(),
                  val_sample_ids=np.array([r["sample_id"] for r in val_ds.records]))
    test_ds = None
    if test_manifest.exists():
        test_ds = ds("test")
        test_dl = DataLoader(test_ds, batch_size=64, num_workers=args.num_workers)
        t_logits, _, _ = collect_logits(model, test_dl, dev)
        preds = t_logits.argmax(1).tolist()
        ids = [r["sample_id"] for r in test_ds.records]
        assert len(ids) == len(preds), (len(ids), len(preds))
        json.dump(dict(zip(ids, preds)), open(out / "test_predictions.json", "w"), indent=1)
        arrays.update(test_logits=t_logits.numpy(), test_sample_ids=np.array(ids))
        print(f"wrote {len(preds)} test predictions -> {out / 'test_predictions.json'}")
    np.savez_compressed(out / "logits.npz", **arrays)      # for 12_visualize.py predictions / calibration, ensembles

    # ---- pictures of what the model actually predicts ----
    if VIZ is not None and not args.no_plots:
        if not args.data_root:
            print("[viz] pass --data_root (the folder the manifest panel paths are relative to) to also get the prediction pictures")
        else:
            try:
                VIZ.plot_predictions(val_ds.records, v_logits.numpy(), args.data_root, out / "pred_viz_val", args.viz_samples, "mixed", args.seed)
                VIZ.plot_predictions([full_train.records[i] for i in hold_idx], h_logits.numpy(), args.data_root, out / "pred_viz_holdout",
                                     args.viz_samples, "mixed", args.seed)
            except Exception as e:
                print(f"[viz] prediction pictures skipped: {e!r}")


if __name__ == "__main__":
    main()
