"""Step 13: save the model's scores (logits) for every finished run, for calibration / ensemble / error-analysis plots.

Works on runs trained with the current 08_train_fusion.py (needs results.json + best.pt in each run folder). It rebuilds
the model from the arguments stored in results.json, reloads best.pt (the checkpoint chosen on the train holdout) and writes
<run_dir>/logits.npz with:
    hold_logits [2000,5], hold_labels, hold_pos     (the same train-holdout rows used for checkpoint selection)
    val_logits  [259,5],  val_labels,  val_pos
    test_logits [929,5],  test_sample_ids           (labels are hidden)

    python scripts/13_dump_logits.py --runs_root outputs/sweeps_siglib_clip
    python scripts/12_visualize.py calibration --runs_root outputs/sweeps_siglib_clip

Existing logits.npz files are kept unless --force is given.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import CachedFeatureDataset  # noqa: E402
from src.models.fusion import CandidateConditionedFusion  # noqa: E402


@torch.no_grad()
def logits_of(model, dl, dev):
    model.eval()
    L, Y, P = [], [], []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        L.append(model(b["visual"], b["text"], b["has_text"], b["missing_pos"]).float().cpu())
        Y.append(b["label"].cpu()); P.append(b["missing_pos"].cpu())
    return torch.cat(L).numpy(), torch.cat(Y).numpy(), torch.cat(P).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for res_path in sorted(Path(args.runs_root).glob("*/results.json")):
        run = res_path.parent
        out = run / "logits.npz"
        if out.exists() and not args.force:
            print(f"[skip] {run.name} (logits.npz exists)")
            continue
        a = json.load(open(res_path, encoding="utf-8")).get("args", {})
        if "holdout" not in a or not (run / "best.pt").exists():
            print(f"[skip] {run.name}: not a current-format run (no holdout args or best.pt)")
            continue
        text_dir = None if a.get("no_text") else a.get("text_dir")
        mk = lambda split: CachedFeatureDataset(Path(a["manifest_dir"]) / f"{split}.jsonl", a["visual_dir"], text_dir)
        model = CandidateConditionedFusion(visual_dim=a.get("visual_dim", 768), hidden=a["hidden"], layers=a["layers"],
                                           heads=a["heads"], ff=a["ff"], dropout=a["dropout"], use_text=text_dir is not None,
                                           use_visual=not a.get("no_visual", False),
                                           text_mode=a.get("text_mode", "embed")).to(dev)
        model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
        train = mk("train")
        N, H = len(train), a["holdout"]
        loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, num_workers=0)
        hl, hy, hp = logits_of(model, loader(Subset(train, list(range(N - H, N)))), dev)
        vds = mk("validation")
        vl, vy, vp = logits_of(model, loader(vds), dev)
        arrays = dict(hold_logits=hl, hold_labels=hy, hold_pos=hp, val_logits=vl, val_labels=vy, val_pos=vp,
                      hold_sample_ids=np.array([train.records[i]["sample_id"] for i in range(N - H, N)]),
                      val_sample_ids=np.array([r["sample_id"] for r in vds.records]))
        tm = Path(a["manifest_dir"]) / "test.jsonl"
        if tm.exists():
            tl, _, _ = logits_of(model, loader(mk("test")), dev)
            arrays["test_logits"] = tl
            arrays["test_sample_ids"] = np.array([json.loads(l)["sample_id"] for l in open(tm, encoding="utf-8")])
        np.savez_compressed(out, **arrays)
        acc = (vl.argmax(1) == vy).mean()
        print(f"[ok] {run.name}: val acc from saved logits {acc:.4f} (results.json says {json.load(open(res_path))['val']['acc']:.4f})")


if __name__ == "__main__":
    main()
