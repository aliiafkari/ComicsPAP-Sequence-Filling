"""Dataset over cached embeddings (visual always; text optional until OCR is run)."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    def __init__(self, manifest, visual_dir, text_dir=None, text_dim=384):
        self.records = [json.loads(l) for l in open(manifest)]
        self.visual_dir = Path(visual_dir)
        self.text_dir = Path(text_dir) if text_dir else None
        self.text_dim = text_dim
        missing = [r["sample_id"] for r in self.records if not (self.visual_dir / f"{r['sample_id']}.pt").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} samples lack visual features, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        sid = r["sample_id"]
        vis = torch.load(self.visual_dir / f"{sid}.pt")["visual"].float()   # [10,768]
        n = vis.shape[0]
        text, has_text = torch.zeros(n, self.text_dim), torch.zeros(n)
        if self.text_dir is not None and (self.text_dir / f"{sid}.pt").exists():
            t = torch.load(self.text_dir / f"{sid}.pt")
            text, has_text = t["text"].float(), t["has_text"].float()
        return {
            "visual": vis, "text": text, "has_text": has_text,
            "missing_pos": torch.tensor(r["missing_position"]),
            "label": torch.tensor(r["label"]),
        }


def shuffle_candidates(batch_visual, batch_text, batch_has_text, label, n_ctx=4):
    """Randomly permute the 6 candidates per sample and remap the label (anti-order-bias)."""
    B, N = batch_visual.shape[:2]
    K = N - n_ctx
    perm = torch.stack([torch.randperm(K) for _ in range(B)]).to(label.device)  # new slot -> old idx
    idx = torch.cat([torch.arange(n_ctx, device=label.device).expand(B, n_ctx), perm + n_ctx], dim=1)

    def g(x):
        return x.gather(1, idx.view(B, N, *([1] * (x.dim() - 2))).expand_as(x))

    new_label = (perm == label[:, None]).float().argmax(1)
    return g(batch_visual), g(batch_text), g(batch_has_text), new_label
