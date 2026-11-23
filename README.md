# ComicsPAP: Sequence Filling with Frozen Encoders, OCR and Qwen2.5-VL

Research code for **Task 1.1 (sequence filling)** of the **ComicsPAP** (*Pick-a-Panel*) benchmark: given four consecutive comic panels with one panel missing, choose the correct panel out of five candidates. Chance accuracy is **0.20**.

This repository contains a full, reproducible pipeline and the experiments built on it:

1. **Data preparation**: parquet to panel JPEGs and JSONL manifests.
2. **Frozen visual encoders** (DINOv2, CLIP, SigLIP) feeding a small, permutation-equivariant, candidate-conditioned transformer ("fusion" model).
3. **OCR** with PP-OCRv6 (PaddleOCR 3.x), OCR-quality spot-checking, and text features (MiniLM) fused with the visual features.
4. **Qwen2.5-VL-3B-Instruct** LoRA / QLoRA: zero-shot, the released ComicsPAP adapter, continued fine-tuning with and without OCR text, all runnable on an **8 GB GPU**.
5. **Diagnostics**: bootstrap confidence intervals, candidate-shuffle robustness, calibration, per-position accuracy, train/validation/test leakage and shift checks, prediction galleries.

> **Main finding so far:** OCR text did not help in any setup I tried, and frozen-feature fusion models stay far below a fine-tuned vision-language model. Validation has only 259 samples, so differences of a few points are within noise (see [Results](#results) and [Caveats](#caveats)).

---

## Table of contents

- [Task and data](#task-and-data)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data setup](#data-setup)
- [Running the pipeline](#running-the-pipeline)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Caveats](#caveats)
- [Hardware notes](#hardware-notes)
- [Troubleshooting](#troubleshooting)
---

## Task and data

Each sample has **4 context panels** (a comic strip with one panel removed) and **5 candidate panels**. The label is which candidate belongs in the gap, and `missing_position` (0 to 4) says where in the sequence the gap is. Predictions are scored by accuracy.

Dataset and baselines come from the ComicsPAP authors (VLR-CVC, Computer Vision Center). Download the sequence-filling parquet files from Hugging Face (check the dataset page for the exact repository id and its license before redistributing anything derived from it). The released baseline adapter is [`VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP`](https://huggingface.co/VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP).

Sizes after preparation (standard 4 + 5 layout only):

| split | samples | set aside (non-standard layout) |
|---|---|---|
| train | 23,503 | 101 |
| validation | 259 | 3 |
| test | 929 (labels hidden, `label = -1`) | 3 |

The dataset itself is **not** included in this repository.

---

## Repository structure

```
.
├── configs/
│   └── dataset.yaml            # parquet column names / label base for 02_prepare_manifest.py
├── scripts/
│   ├── 01_inspect_dataset.py          # print parquet schema
│   ├── 02_prepare_manifest.py         # parquet -> panel JPEGs + manifests/<split>.jsonl
│   ├── 04_run_ocr.py                  # PP-OCRv6 over all unique panels (md5-cached)   [ocr env]
│   ├── 04b_ocr_spotcheck.py           # HTML labeler + CER/WER + detection P/R         [ocr env]
│   ├── 05_generate_text_embeddings.py # MiniLM-L6-v2 embeddings of OCR text
│   ├── 06_extract_visual_features.py  # DINOv2 / CLIP / SigLIP frozen features
│   ├── 07_train_qwen.py               # Qwen2.5-VL LoRA/QLoRA training                 [qwen env]
│   ├── 07_train_qwen_eval.py          # score a saved adapter on validation            [qwen env]
│   ├── 07b_eval_qwen.py               # eval zero-shot / released / any adapter        [qwen env]
│   ├── 08_train_fusion.py             # train one fusion model
│   ├── 11_run_sweep.py                # seeds x configs x encoders, then aggregate
│   ├── 12_visualize.py                # curves, calibration, galleries, shift/leakage
│   ├── 13_dump_logits.py              # save logits of finished fusion runs
│   ├── 14_run_ocr_ablations.py        # OCR confidence / reading-order ablations
│   ├── extract_parquet_ground_truth.py
│   └── text_utils.py                  # OCR text assembly shared by 05 / 07 / 14
├── src/
│   ├── data/      (features.py, pap_render.py)
│   ├── models/    (fusion.py, pap_render.py, qwen_pap.py)
│   └── viz/       (pap_plots.py)
├── tests/test_fusion.py
├── notebook.ipynb              # step-by-step runner for everything above
├── requirements.txt            # main env
├── requirements-ocr.txt        # OCR env
└── requirements-qwen.txt       # Qwen env
```

Scripts use a numeric prefix for pipeline order (numbers 03, 09 and 10 are unused). Everything is written to `outputs/` (git-ignored) and to a data work directory outside the repo.

---

## Installation

Three separate Python environments are used, because PaddlePaddle and the Qwen stack (bitsandbytes, a specific transformers) conflict with each other. Python 3.10+ is recommended.

```bash
# 1. main env: data prep, visual/text features, fusion, plots (also the notebook kernel)
python -m venv .venv-main && source .venv-main/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install -r requirements.txt

# 2. OCR env (PaddleOCR 3.x, PP-OCRv6)
python -m venv .venv-ocr && source .venv-ocr/bin/activate
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/   # see Paddle docs
pip install -r requirements-ocr.txt

# 3. Qwen env
python -m venv .venv-qwen && source .venv-qwen/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-qwen.txt
```

On Windows use `.venv-main\Scripts\activate` and note the interpreter path (`...\Scripts\python.exe`) for the notebook configuration.

Run the unit tests (main env): `pytest tests/`.

---

## Data setup

1. Download the sequence-filling parquet files (`train`, `validation`, `test`) into a folder, e.g. `/data/ComicsPAP/sequence_filling`.
2. Inspect the schema:
   ```bash
   python scripts/01_inspect_dataset.py --data_root /data/ComicsPAP/sequence_filling
   ```
3. Edit `configs/dataset.yaml` from that output. The keys read by `02_prepare_manifest.py` are:
   - `columns`: mapping for `sample_id`, `context`, `candidates`, `label`, `missing_position`, `book_id`, `task_type`
   - `num_context` (4), `num_candidates` (5)
   - `label_base`, `missing_position_base` (0 or 1; the manifests are always zero-based)
   - `task_filter`: keep only sequence-filling rows if the parquet mixes tasks
4. Extract panels and build manifests:
   ```bash
   python scripts/02_prepare_manifest.py --data_root /data/ComicsPAP/sequence_filling \
       --out_root /data/work --config configs/dataset.yaml --on_mismatch keep
   ```
   Outputs: `/data/work/extracted_panels/...` and `/data/work/manifests/{train,validation,test}.jsonl`.
5. A few samples do not have the 4 + 5 layout. The notebook (section 1) moves them to `<split>_nonstandard.jsonl` and keeps a `.bak` of the original manifest.

---

## Running the pipeline

Everything below is also in **`notebook.ipynb`**, which exposes all paths in a single configuration cell (data root, interpreters for the OCR and Qwen envs, font). Start Jupyter from the repository root. The commands here use `$DATA`, `$MAN` and so on as shorthand.

```bash
DATA=/data/work; MAN=$DATA/manifests; EMB=$DATA/embeddings
```

### 1. Frozen visual features (main env)

```bash
for S in validation test train; do
  python scripts/06_extract_visual_features.py --encoder dinov2 --manifest $MAN/$S.jsonl \
      --data_root $DATA --out_dir $EMB/visual --batch_size 8 --fp16
done
# repeat with --encoder clip (out_dir $EMB/visual_clip, dim 512) and --encoder siglip (out_dir $EMB/visual_siglip, dim 768)
```

### 2. Visual-only fusion baselines

```bash
python scripts/11_run_sweep.py --manifest_dir $MAN --seeds 0 1 2 --configs base \
    --features dinov2=$EMB/visual@768 clip=$EMB/visual_clip@512 siglip=$EMB/visual_siglip@768 \
    --out_root outputs/sweep_visual --data_root $DATA
python scripts/11_run_sweep.py --aggregate_only --out_root outputs/sweep_visual
```

Presets: `base`, `reg` (dropout 0.3, weight decay 0.1, lr 5e-5, 2 layers), and, for text runs, `tdrop` (50% text dropout), `sim` (candidate-to-context text similarity only), `both`. To train a single model directly use `scripts/08_train_fusion.py` (`--no_text`, `--no_visual`, `--text_dir`, `--text_mode embed|sim|both`, `--text_dropout`, `--holdout`, `--holdout_gap`).

### 3. OCR and text features

```bash
# OCR env
python scripts/04_run_ocr.py --data_root $DATA --out_dir $DATA/ocr/ppocrv6_medium \
    --manifest $MAN/validation.jsonl $MAN/test.jsonl $MAN/train.jsonl
# main env
python scripts/05_generate_text_embeddings.py --ocr_dir $DATA/ocr/ppocrv6_medium \
    --manifest $MAN/validation.jsonl $MAN/test.jsonl $MAN/train.jsonl \
    --out_dir $EMB/text_ppocrv6_medium
# visual vs visual+OCR vs OCR-only
python scripts/11_run_sweep.py --manifest_dir $MAN --features siglip=$EMB/visual_siglip@768 \
    --text_dir $EMB/text_ppocrv6_medium --modes visual visual_text text --configs base \
    --seeds 0 1 2 --out_root outputs/sweep_text --data_root $DATA
```

Optional: `04b_ocr_spotcheck.py sample` builds an HTML page to hand-label ~60 to 150 panels, `score` then reports CER / WER and detection precision / recall at several confidence thresholds. `14_run_ocr_ablations.py` sweeps confidence filter (0, 0.5, 0.8) and reading order (`raw`, `reading`).

### 4. Diagnostics

```bash
python scripts/13_dump_logits.py --runs_root outputs/sweep_visual
python scripts/12_visualize.py all --runs_root outputs/sweep_visual --manifest_dir $MAN \
    --data_root $DATA --ocr_dir $DATA/ocr/ppocrv6_medium
python scripts/12_visualize.py shift --ocr_dir $DATA/ocr/ppocrv6_medium --out_dir outputs/figures
```

### 5. Qwen2.5-VL-3B-Instruct (qwen env)

Panels are rendered into a two-row composite image (context row, candidate row) with number labels, and the model answers with a digit 0 to 4. Two decoding modes: `score` (forced choice, compares the logits of the digits 0 to 4, never unparsable) and `generate` (free generation, parsed). All commands below use 4-bit weights and `--max_pixels 401408` to fit 8 GB.

```bash
COMMON="--load_4bit --max_pixels 401408 --manifest_dir $MAN --data_root $DATA --font /path/to/font.ttf"

# E1: released ComicsPAP adapter          E0: zero-shot base model
python scripts/07b_eval_qwen.py --adapter VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP --decode score --out outputs/qwen_e1 $COMMON
python scripts/07b_eval_qwen.py --adapter none --decode both --out outputs/qwen_e0 $COMMON

# E1 + OCR text in the prompt at inference only
python scripts/07b_eval_qwen.py --adapter VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP --decode score \
    --ocr_dir $DATA/ocr/ppocrv6_medium --out outputs/qwen_e1_ocr_infer $COMMON

# continue fine-tuning the released adapter (images only); E8 adds --ocr_dir
python scripts/07_train_qwen.py --init_adapter VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP \
    --lr 2e-5 --train_samples 1000 --epochs 2 --grad_accum 8 --eval_every 50 --eval_samples 150 \
    --seed 0 --out outputs/qwen_cont_images $COMMON

# score a saved adapter on official validation
python scripts/07_train_qwen_eval.py --adapter outputs/qwen_cont_images/best_adapter \
    --out outputs/qwen_cont_images/val_eval --final_decode score $COMMON
```

Checkpoints are selected on a **train holdout**, never on validation. Use `--smoke` in `07b_eval_qwen.py` for a 3-sample test that also saves the rendered composite image. On a 3070 expect several minutes per training step, so budget hours for a 1,000-sample run.

---

## Evaluation protocol

- **Selection set:** the last 2,000 training rows are held out (with a 200-row gap between the training part and the holdout to reduce near-duplicate leakage within a book/series). The checkpoint or epoch is chosen on this holdout.
- **Reporting set:** the official validation split (n = 259), scored once with the selected checkpoint. Validation is **never** used for selection or early stopping.
- **Uncertainty:** bootstrap 95% CIs on accuracy for single runs; mean ± sample std over seeds and a t-based 95% CI for sweeps.
- **Robustness:** a candidate-shuffle test checks that predictions follow the content, not the position (fusion models are permutation-equivariant by construction).
- **Leakage/shift:** `12_visualize.py shift` compares exact-image and OCR-text overlap between train, holdout, validation and test.

---

## Results

Validation accuracy, n = 259 unless noted; chance = 0.20. Fusion rows are mean ± std over 3 seeds.

### Frozen-feature fusion (visual only)

| encoder | validation | train holdout |
|---|---|---|
| DINOv2-base (CLS, 768-d) | 0.312 ± 0.012 | 0.353 |
| DINOv2-base, `reg` preset | 0.242 ± 0.008 | 0.319 |
| CLIP ViT-B/16 (512-d) | 0.350 ± 0.029 | 0.380 |
| SigLIP-B/16 (768-d) | 0.350 ± 0.034 | 0.389 |

### Adding OCR text (SigLIP features)

| setup | validation | train holdout |
|---|---|---|
| OCR only (MiniLM embeddings) | 0.175 ± 0.010 | 0.230 |
| SigLIP + OCR | 0.283 ± 0.033 | 0.374 |
| SigLIP + OCR, text dropout 0.5 | 0.273 | 0.370 |
| SigLIP + OCR, similarity features only | 0.260 | 0.408 |
| SigLIP + OCR, embeddings + similarity + dropout | 0.251 | 0.409 |

OCR-variant ablation (confidence filter × reading order): validation 0.256 to 0.273, holdout 0.366 to 0.376, i.e. no consistent effect.

### Qwen2.5-VL-3B-Instruct

| experiment | validation | notes |
|---|---|---|
| E0: zero-shot | 0.209 (score) / 0.216 (generate) | about chance |
| E1: released adapter (4-bit, low resolution) | 0.579 (95% CI 0.517 to 0.637) | paper reports 62.2% on 262 samples with the full-precision setup |
| E1 + OCR in prompt at inference | 0.305 | adapter never saw OCR text |
| Continued fine-tune, images only (best holdout adapter) | 0.610 (95% CI 0.548 to 0.668) | |
| Fine-tune with OCR text (E8) | 0.440 | best holdout 0.460 |

**Reading the numbers:** the fine-tuned VLM is much better than any frozen-feature model; OCR as an extra input never helped here (neither as embeddings, similarity features, nor prompt text); the gain from continued fine-tuning (0.579 → 0.610) is inside the bootstrap CIs and should not be over-interpreted.

---

## Caveats

- **Small validation set.** With n = 259 the standard error is about 3 points; CIs of competing methods overlap widely. Seeds add further variance (std 1 to 3 points for the fusion models).
- **Test set** labels are hidden; no test numbers are reported here.
- **Low-resolution 4-bit setup.** Qwen runs use `--max_pixels 401408` and 4-bit weights for an 8 GB GPU, which is below the released baseline's configuration; absolute numbers are not directly comparable with the paper.
- **First-stage Qwen fine-tune flags** in the notebook are reconstructed from the later stages, adjust them to your run.
- **OCR quality** on stylised comic lettering is imperfect; use the spot-check script to measure it on your own sample before drawing conclusions about text.
- `02_prepare_manifest.py`'s docstring mentions composite-image handling that the code does not implement.

---

## Hardware notes

Developed on a single **RTX 3070 (8 GB)** under Windows. Tips:

- Qwen training uses 4-bit quantisation, bf16 compute, gradient checkpointing, `--grad_accum` for the effective batch size, and `--max_pixels` to bound visual tokens. Lower `--max_pixels` first if you run out of memory.
- Feature extraction and fusion training run comfortably on 8 GB; the fusion models also train on CPU, only slower.
- With a 24 GB+ GPU drop `--load_4bit` and raise `--max_pixels`.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `SyntaxError` in `02_prepare_manifest.py` | make sure you use the version in this repo (a stray line was removed) |
| `CUDA error: an illegal memory access` in bitsandbytes `dequantize_4bit` during Qwen training, very slow steps | reduce `--max_pixels`, close other GPU processes, update bitsandbytes/torch, resume from `last_adapter` (`--init_adapter outputs/<run>/last_adapter`) |
| `UnicodeEncodeError` on Windows consoles | `set PYTHONUTF8=1` (the notebook sets it) |
| PaddleOCR / torch import conflicts | keep the three environments separate; run OCR only with the OCR interpreter |
| `panels have no OCR record` in step 05 | run `04_run_ocr.py` for every manifest you pass to `05` |
| Hugging Face downloads blocked | set `HF_ENDPOINT` / `HF_HOME` (see the notebook config cell) |
| Windows paths with backslashes in commands | use forward slashes or quote the path |
| notebook says `scripts/ not found` | start Jupyter from the repository root |

---
