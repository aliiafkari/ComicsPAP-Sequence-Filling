"""Step 6: frozen visual embeddings for the panels of each sample (4 context + 5 candidates = 9).

Usage (one output folder per encoder so they never mix):
    # DINOv2 CLS (768-d)              -> the baseline you already have
    python scripts/06_extract_visual_features.py --manifest D:/new_comic/manifests/train.jsonl \
        --data_root D:/new_comic --out_dir D:/new_comic/embeddings/visual --fp16

    # DINOv2 CLS + mean of patch tokens (1536-d)
    python scripts/06_extract_visual_features.py --encoder dinov2 --pool cls_mean ^
        --manifest ... --data_root D:/new_comic --out_dir D:/new_comic/embeddings/visual_dinov2_clsmean --fp16

    # CLIP ViT-B/16 (512-d)  or  SigLIP ViT-B/16 (768-d)
    python scripts/06_extract_visual_features.py --encoder clip   --out_dir D:/new_comic/embeddings/visual_clip   ...
    python scripts/06_extract_visual_features.py --encoder siglip --out_dir D:/new_comic/embeddings/visual_siglip ...

Run it for train, validation AND test with the same --out_dir. Samples that already have a .pt are
skipped (resume). Cache layout: <out_dir>/<sample_id>.pt = {"visual": Tensor[9, D]}
Panel order: 0-3 = context (original order), 4-8 = candidates 0..4.
Train 08_train_fusion.py with the matching --visual_dim (printed at the end of each run).

Notes: the HF image processors resize and (for DINOv2/CLIP) centre-crop to 224x224, so the edges of
non-square panels are cropped; SigLIP squashes to 224x224 instead. "cls_mean" is pooled patch
information, not full patch-level features (those would need a different model input).
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODELS = {
    "dinov2": "facebook/dinov2-base",
    "clip": "openai/clip-vit-base-patch16",
    "siglip": "google/siglip-base-patch16-224",
}


class PanelSet(Dataset):
    def __init__(self, records, data_root, processor):
        self.records, self.root, self.proc = records, Path(data_root), processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        paths = r["context_panels"] + r["candidates"]
        imgs = []
        for p in paths:
            with Image.open(self.root / p) as im:
                imgs.append(im.convert("RGB"))
        pix = self.proc(images=imgs, return_tensors="pt")["pixel_values"]  # [P,3,H,W]
        return r["sample_id"], pix


def collate(b):
    return [x[0] for x in b], torch.stack([x[1] for x in b])


def encode(model, x, encoder, pool):
    if encoder == "dinov2":
        h = model(pixel_values=x).last_hidden_state                       # [n, 1+patches, D]
        cls = h[:, 0]
        return torch.cat([cls, h[:, 1:].mean(1)], dim=-1) if pool == "cls_mean" else cls
    out = model.get_image_features(pixel_values=x)                        # CLIP / SigLIP projected embedding
    if not torch.is_tensor(out):                                          # some transformers versions return a ModelOutput
        out = getattr(out, "pooler_output", None) if getattr(out, "pooler_output", None) is not None else out[0]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="data/embeddings/visual")
    ap.add_argument("--encoder", choices=list(DEFAULT_MODELS), default="dinov2")
    ap.add_argument("--model", default=None, help="override the HF model id")
    ap.add_argument("--pool", choices=["cls", "cls_mean"], default="cls", help="dinov2 only")
    ap.add_argument("--batch_size", type=int, default=8, help="samples per batch (x9 panels each)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    if args.pool == "cls_mean" and args.encoder != "dinov2":
        raise SystemExit("--pool cls_mean is only supported for --encoder dinov2")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh]
    records = [r for r in records if not (out / f"{r['sample_id']}.pt").exists()]  # resume
    print(f"{len(records)} samples to process -> {out}")
    if not records:
        return

    model_id = args.model or DEFAULT_MODELS[args.encoder]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    half = args.fp16 and dev == "cuda"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    if half:
        model = model.half()

    dl = DataLoader(PanelSet(records, args.data_root, proc), batch_size=args.batch_size,
                    num_workers=args.num_workers, collate_fn=collate)
    dim = None
    for ids, pix in tqdm(dl):
        B, P = pix.shape[:2]
        x = pix.flatten(0, 1).to(dev)
        if half:
            x = x.half()
        feats = encode(model, x, args.encoder, args.pool).float().cpu().view(B, P, -1)
        dim = feats.shape[-1]
        for sid, f in zip(ids, feats):
            torch.save({"visual": f.clone()}, out / f"{sid}.pt")
    print(f"done. embedding dim = {dim}  ->  use --visual_dim {dim} in 08_train_fusion.py / 11_run_sweep.py")


if __name__ == "__main__":
    main()
