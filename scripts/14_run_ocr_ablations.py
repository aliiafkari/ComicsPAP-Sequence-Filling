"""Step 14: OCR-variant ablations (confidence filtering x reading order) with the visual+OCR model.

For every variant it (1) re-runs 05_generate_text_embeddings.py into its own folder, then (2) runs 11_run_sweep.py
with --modes visual_text and a --text_tag naming the variant, so all runs land in ONE table next to your baselines.
Run it in the MAIN (torch) environment:

    python scripts/14_run_ocr_ablations.py ^
        --ocr_dir D:/new_comic/ocr/ppocrv6_medium ^
        --manifest_dir D:/new_comic/manifests ^
        --text_root D:/new_comic/embeddings/text_ppocrv6_medium ^
        --features siglip=D:/new_comic/embeddings/visual_siglip@768 ^
        --out_root outputs/sweeps_siglib_clip_text --configs base --seeds 0 1 2

Variants (default): c0.5-raw, c0.8-raw, c0-reading, c0.8-reading. The unfiltered/raw baseline is the existing
"siglip+ocr" row (no tag), produced by your earlier sweep, so it is not re-run. Use --variants to choose, e.g.
    --variants 0.8:reading 0.5:raw
(format <min_conf>:<raw|reading>). Embedding folders are named <text_root>_c<conf>-<order> and are reused if they exist
(add --force_embeddings to rebuild). Add --configs base sim to also try the similarity-feature model on every variant.

Read the result with care: each variant is a different comparison against the same baseline, with 3 seeds each and
validation n=259. Differences of 1-2 points are noise; look at the holdout column too (n=2000) and prefer a change only if
it moves both columns the same way.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_variant(s):
    conf, order = s.split(":")
    if order not in ("raw", "reading"):
        raise SystemExit(f"bad variant {s!r}: order must be raw or reading")
    return float(conf), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_dir", required=True)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--text_root", required=True, help="prefix of the embedding folders, e.g. D:/new_comic/embeddings/text_ppocrv6_medium")
    ap.add_argument("--features", nargs="+", required=True, help="name=dir@dim, as in 11_run_sweep.py")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--variants", nargs="+", default=["0.5:raw", "0.8:raw", "0.0:reading", "0.8:reading"])
    ap.add_argument("--configs", nargs="+", default=["base"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--force_embeddings", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print the commands only")
    ap.add_argument("--data_root", default=None, help="forwarded to the sweep: enables prediction pictures in every run folder")
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    manifests = [str(mdir / f"{s}.jsonl") for s in ["validation", "test", "train"] if (mdir / f"{s}.jsonl").exists()]
    sweep_cmds = []
    for v in args.variants:
        conf, order = parse_variant(v)
        tag = f"c{conf:g}-{order}"
        text_dir = f"{args.text_root}_{tag}"
        if Path(text_dir, "text_config.json").exists() and not args.force_embeddings:
            print(f"[reuse] embeddings for {tag}: {text_dir}")
        else:
            cmd = [sys.executable, str(HERE / "05_generate_text_embeddings.py"), "--ocr_dir", args.ocr_dir,
                   "--out_dir", text_dir, "--min_conf", str(conf), "--order", order, "--manifest", *manifests]
            print("\n[embeddings]", " ".join(cmd))
            if not args.dry_run and subprocess.run(cmd).returncode != 0:
                print(f"[FAILED] embeddings for {tag}; skipping this variant")
                continue
        sweep_cmds.append([sys.executable, str(HERE / "11_run_sweep.py"), "--manifest_dir", args.manifest_dir,
                           "--features", *args.features, "--text_dir", text_dir, "--text_tag", tag,
                           "--modes", "visual_text", "--configs", *args.configs, "--seeds", *args.seeds,
                           "--out_root", args.out_root] + (["--data_root", args.data_root] if args.data_root else []))
    for cmd in sweep_cmds:
        print("\n[sweep]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd)
    final = [sys.executable, str(HERE / "11_run_sweep.py"), "--aggregate_only", "--out_root", args.out_root]
    print("\n[summary]", " ".join(final))
    if not args.dry_run:
        subprocess.run(final)


if __name__ == "__main__":
    main()
