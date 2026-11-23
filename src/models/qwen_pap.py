"""Qwen2.5-VL helpers for ComicsPAP sequence filling (E0 zero-shot, E1 released LoRA, E2 sequence-only LoRA, E8 +OCR).
Place at  src/models/qwen_pap.py.  Heavy libraries (torch / transformers / peft) are imported lazily, so the pure helpers
(prompt, parser, key remapping, OCR block, metrics) can be imported and tested without them.

Conventions taken from the official ComicsPAP code (llabres/ComicsPAP, model_builders.py + the dataset card):
  * labels (`solution_index`) and the missing slot (`index`) are 0-based; option numbers drawn on the image are 0-based;
  * the model is prompted with PROMPT + "context: " + one composite image and answers "answer: <number>";
  * the same prompt is used for training and evaluation (the 'You can reason...' variant is commented out in their eval).
Differences you should know about: (1) their training loss also covered the prompt tokens, ours covers only the answer;
(2) we use scaled-dot-product attention instead of flash-attention 2; (3) generation settings of their evaluate.py were not
inspected, so E0/E1 numbers can differ slightly from the published ones.
"""
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROMPT = ("Pick A Panel Task: In the image you have two rows of comic panels. The top row is the context and the bottom row is "
          "the options. The context row has a missing panel marked with a question mark. Choose the option that best fits the "
          "missing panel. You must return your final answer as a number with 'answer: <your answer here>'\n\n")
ANSWER_PREFIX = "answer: "


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
_ANS = re.compile(r"answer\s*:\s*(\d+)", re.I)


def parse_answer(text: str, n_opts: int):
    """-> (prediction or None, used_fallback). Strict 'answer: N' first; otherwise the last integer in the text."""
    m = _ANS.findall(text)
    if m:
        v = int(m[-1])
        return (v if 0 <= v < n_opts else None), False
    nums = re.findall(r"\b\d+\b", text)
    if nums:
        v = int(nums[-1])
        return (v if 0 <= v < n_opts else None), True
    return None, False


def bootstrap_ci(correct, n_boot=2000, seed=0):
    correct = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), size=(n_boot, len(correct)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def digit_token_ids(tokenizer, n: int, prefix: str = ANSWER_PREFIX):
    """Token id of each digit when it follows `prefix`; None if the tokenizer does not split them as one extra token."""
    base = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = []
    for d in range(n):
        full = tokenizer(prefix + str(d), add_special_tokens=False)["input_ids"]
        if full[:len(base)] != base or len(full) != len(base) + 1:
            return None
        ids.append(full[-1])
    return ids


def build_messages(ocr_block: Optional[str] = None):
    content = [{"type": "text", "text": PROMPT}, {"type": "text", "text": "context: "}, {"type": "image"}]
    if ocr_block:
        content.append({"type": "text", "text": "\n" + ocr_block})
    return [{"role": "user", "content": content}]


def prompt_text(processor, ocr_block: Optional[str] = None) -> str:
    return processor.apply_chat_template(build_messages(ocr_block), tokenize=False, add_generation_prompt=True)


def load_ocr_index(ocr_dir, split, sample_ids=None):
    """{(sample_id, panel_id): [detections]} from the 04_run_ocr.py output."""
    idx = {}
    with open(Path(ocr_dir) / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if sample_ids is None or r["sample_id"] in sample_ids:
                idx[(r["sample_id"], r["panel_id"])] = r["ocr"]
    return idx


def make_ocr_block(rec, ocr_index, option_order=None, order="raw", min_conf=0.0, max_chars=200) -> str:
    """Structured OCR text aligned with the picture: top-row panel numbers = slots (the missing one is marked), option
    numbers = the digits drawn on the options (after any shuffling)."""
    from scripts.text_utils import panel_text
    option_order = list(range(len(rec["candidates"]))) if option_order is None else list(option_order)

    def txt(rel):
        t = panel_text(ocr_index.get((rec["sample_id"], Path(rel).stem), []), order, min_conf)
        if len(t) > max_chars:
            t = t[:max_chars].rstrip() + "..."
        return t if t else "(no text)"

    lines = ["Text read from the panels by OCR (it may contain errors):", "Top row:"]
    ci = 0
    for slot in range(len(rec["context_panels"]) + 1):
        if slot == rec["missing_position"]:
            lines.append(f"  panel {slot}: [missing]")
        else:
            lines.append(f"  panel {slot}: {txt(rec['context_panels'][ci])}")
            ci += 1
    lines.append("Options:")
    for k, j in enumerate(option_order):
        lines.append(f"  option {k}: {txt(rec['candidates'][j])}")
    return "\n".join(lines)


# ---- manifests / splits (same protocol as the fusion models) ----
def read_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_split(manifest_dir, split, holdout=2000, gap=200):
    """validation | test | train | fit (train minus holdout and gap) | holdout (last `holdout` train rows)."""
    mdir = Path(manifest_dir)
    if split in ("validation", "test"):
        return read_manifest(mdir / f"{split}.jsonl")
    tr = read_manifest(mdir / "train.jsonl")
    n = len(tr)
    if split == "train":
        return tr
    if split == "fit":
        return tr[: n - holdout - gap]
    if split == "holdout":
        return tr[n - holdout:]
    raise ValueError(split)


def ocr_split_name(split):
    return "train" if split in ("train", "fit", "holdout") else split


def subsample(records, n):
    """n evenly spaced records (deterministic)."""
    if not n or n >= len(records):
        return records
    idx = np.linspace(0, len(records) - 1, n).round().astype(int)
    return [records[i] for i in idx]


# --------------------------------------------------------------------------- #
# adapter key handling (old PEFT adapters vs. newer transformers module names)
# --------------------------------------------------------------------------- #
_RULES = [(".model.layers.", ".model.language_model.layers."),
          ("base_model.model.visual.", "base_model.model.model.visual.")]


def remap_adapter_keys(src_keys: Sequence[str], target_keys: Sequence[str]) -> Dict[str, str]:
    """Map keys of a saved adapter onto the key names of the current PEFT model (transformers >= 4.52 moved the decoder to
    `.model.language_model` and the vision tower to `.model.visual`). Only keys that exist in the target are mapped."""
    tset, mapping = set(target_keys), {}
    for k in src_keys:
        if k in tset:
            mapping[k] = k
            continue
        for old, new in _RULES:
            if old in k and k.replace(old, new, 1) in tset:
                mapping[k] = k.replace(old, new, 1)
                break
    return mapping


def lora_stats(model):
    n = nz = 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            n += 1
            nz += int(p.detach().abs().sum().item() > 0)
    return n, nz


def remap_and_load(peft_model, adapter_path):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils import load_peft_weights
    sd = load_peft_weights(adapter_path, device="cpu")
    target = list(get_peft_model_state_dict(peft_model).keys())
    mp = remap_adapter_keys(list(sd.keys()), target)
    print(f"[adapter] key remap: {len(mp)}/{len(sd)} saved tensors matched the current model "
          f"({sum(k != v for k, v in mp.items())} renamed)")
    set_peft_model_state_dict(peft_model, {mp[k]: v for k, v in sd.items() if k in mp})
    return peft_model


def attach_adapter(model, adapter: str, trainable: bool = False):
    """Load a PEFT/LoRA adapter and VERIFY that its weights actually landed in the model (a mismatch of module names
    would otherwise silently leave an untrained, zero-initialised adapter, i.e. the zero-shot model)."""
    from peft import PeftModel
    pm = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    n, nz = lora_stats(pm)
    if n == 0:
        raise RuntimeError(f"{adapter}: no LoRA tensors found in the model after loading")
    if nz < n:
        print(f"[adapter] only {nz}/{n} LoRA tensors are non-zero after loading; trying to remap saved key names ...")
        pm = remap_and_load(pm, adapter)
        n, nz = lora_stats(pm)
    print(f"[adapter] {adapter}: {n} LoRA B tensors, {nz} non-zero")
    if nz == 0:
        raise RuntimeError(f"{adapter}: the adapter weights were NOT loaded (all LoRA B tensors are zero). Results would be "
                           "zero-shot results. Check the installed peft/transformers versions.")
    if nz < n:
        print(f"[WARN] {n - nz} of {n} LoRA tensors are still zero: the adapter is only partially applied")
    return pm


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def gpu_info():
    import torch
    if not torch.cuda.is_available():
        return "no CUDA device"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 2**30:.1f} GiB, bf16 {'yes' if torch.cuda.is_bf16_supported() else 'NO'}"


def set_pixel_limits(processor, min_pixels=None, max_pixels=None):
    ip = processor.image_processor
    if min_pixels:
        ip.min_pixels = int(min_pixels)
    if max_pixels:
        ip.max_pixels = int(max_pixels)
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        size = dict(size)
        if min_pixels:
            size["shortest_edge"] = int(min_pixels)
        if max_pixels:
            size["longest_edge"] = int(max_pixels)
        ip.size = size


def load_qwen(model_id="Qwen/Qwen2.5-VL-3B-Instruct", adapter=None, attn="sdpa", min_pixels=None, max_pixels=None,
              load_4bit=False, trainable_adapter=False, device="cuda", load_8bit=False, keep_vision_bf16=False):
    """load_4bit / load_8bit need bitsandbytes (the 3B model is ~7.5 GB in bf16, too big for 8 GB cards).
    keep_vision_bf16 leaves the vision tower un-quantised (+~1.3 GB, better image fidelity)."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kw = dict(attn_implementation=attn)
    quant = load_4bit or load_8bit
    if quant:
        from transformers import BitsAndBytesConfig
        skip = ["visual"] if keep_vision_bf16 else None
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt,
                                                           bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        kw["device_map"] = {"": 0}
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dt, **kw)
    except TypeError:                                   # older transformers
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dt, **kw)
    if not quant:
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    set_pixel_limits(processor, min_pixels, max_pixels)
    processor.tokenizer.padding_side = "left"
    if adapter:
        model = attach_adapter(model, adapter, trainable=trainable_adapter)
    if torch.cuda.is_available():
        print(f"[load] GPU memory in use after loading: {torch.cuda.memory_allocated() / 2**30:.2f} GiB of "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    return model, processor


# --------------------------------------------------------------------------- #
# inference: forced-choice scoring and free generation
# --------------------------------------------------------------------------- #
def _dev(model):
    return next(model.parameters()).device


def encode(processor, texts, images, device):
    return processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)


def score_options(model, processor, images, blocks, digit_ids):
    """Logits of the digit tokens right after 'answer: ' -> [B, K] (forced choice among the K options)."""
    import torch
    texts = [prompt_text(processor, b) + ANSWER_PREFIX for b in blocks]
    enc = encode(processor, texts, images, _dev(model))
    with torch.inference_mode():
        try:
            out = model(**enc, logits_to_keep=1)          # avoids materialising logits for every image token
        except TypeError:
            out = model(**enc)
    return out.logits[:, -1, :][:, digit_ids].float().cpu()


def generate_answers(model, processor, images, blocks, max_new_tokens=64):
    import torch
    texts = [prompt_text(processor, b) for b in blocks]
    enc = encode(processor, texts, images, _dev(model))
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def prepare(rec, root, renderer, ocr_index=None, ocr_cfg=None, option_order=None):
    from src.models.pap_render import render_record
    image, label = render_record(renderer, rec, root, option_order)
    block = None
    if ocr_index is not None:
        block = make_ocr_block(rec, ocr_index, option_order, **(ocr_cfg or {}))
    return image, block, label


def evaluate(model, processor, records, root, renderer, ocr_index=None, ocr_cfg=None, decode="both",
             batch_size=1, max_new_tokens=64, verbose=True):
    from tqdm import tqdm
    n_opts = len(records[0]["candidates"])
    digit_ids = None
    if decode in ("score", "both"):
        digit_ids = digit_token_ids(processor.tokenizer, n_opts)
        if digit_ids is None:
            raise RuntimeError("could not map the digits 0..K-1 to single tokens after 'answer: '; use --decode generate")
    logits, texts = [], []
    for i in tqdm(range(0, len(records), batch_size), desc="eval", disable=not verbose):
        chunk = records[i:i + batch_size]
        prep = [prepare(r, root, renderer, ocr_index, ocr_cfg) for r in chunk]
        images, blocks = [p[0] for p in prep], [p[1] for p in prep]
        if decode in ("score", "both"):
            logits.append(score_options(model, processor, images, blocks, digit_ids))
        if decode in ("generate", "both"):
            texts.extend(generate_answers(model, processor, images, blocks, max_new_tokens))
    import torch
    res = {"sample_ids": [r["sample_id"] for r in records],
           "labels": np.array([r.get("label", -1) for r in records]),
           "pos": np.array([r["missing_position"] for r in records]),
           "n_opts": n_opts, "logits": torch.cat(logits).numpy() if logits else None, "texts": texts or None}
    return res


def summarize(res) -> dict:
    labels, pos, K = res["labels"], res["pos"], res["n_opts"]
    known = bool((labels >= 0).all())
    out = {"n": int(len(labels)), "labels_known": known}

    def block(pred):
        d = {"pred_hist": np.bincount(pred[pred >= 0], minlength=K).tolist()}
        if known:
            ok = (pred == labels)
            d["acc"] = float(ok.mean())
            d["acc_ci95"] = list(bootstrap_ci(ok))
            d["acc_per_missing_pos"] = {int(p): float(ok[pos == p].mean()) for p in np.unique(pos)}
        return d

    if res["logits"] is not None:
        out["score"] = block(res["logits"].argmax(1))
        if known:
            top2 = np.argsort(-res["logits"], axis=1)[:, :2]
            out["score"]["top2"] = float((top2 == labels[:, None]).any(1).mean())
    if res["texts"] is not None:
        parsed = [parse_answer(t, K) for t in res["texts"]]
        pred = np.array([-1 if p is None else p for p, _ in parsed])
        out["generate"] = block(pred)
        out["generate"]["unparsed_rate"] = float((pred < 0).mean())
        out["generate"]["fallback_rate"] = float(np.mean([f for _, f in parsed]))
        out["_pred_generate"] = pred.tolist()
    return out


# --------------------------------------------------------------------------- #
# training example (loss only on the answer tokens)
# --------------------------------------------------------------------------- #
def answer_loss(model, enc, short=True):
    """Cross-entropy on the answer tokens (the unmasked labels, which are the last tokens of the sequence).
    short=True asks the model for the logits of only those positions (logits_to_keep): a full 3k-token x 152k-vocabulary logits
    tensor costs ~1 GB or more, which matters on small GPUs. Numerically identical to the model's own loss (tested)."""
    import torch
    import torch.nn.functional as F
    labels = enc["labels"]
    n_ans = int((labels[0] != -100).sum())
    inputs = {k: v for k, v in enc.items() if k != "labels"}
    if short and labels.shape[0] == 1:
        try:
            out = model(**inputs, logits_to_keep=n_ans + 1)
        except TypeError:
            out = None
        if out is not None:
            lg = out.logits[:, :-1, :].float()                       # the positions that predict the answer tokens
            tgt = inputs["input_ids"][:, -n_ans:]
            return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1))
    return model(**enc).loss


def build_train_example(processor, image, block, label, device):
    prompt = prompt_text(processor, block)
    p = processor(text=[prompt], images=[image], return_tensors="pt")
    f = processor(text=[prompt + f"answer: {label}<|im_end|>"], images=[image], return_tensors="pt")
    labels = f["input_ids"].clone()
    labels[:, : p["input_ids"].shape[1]] = -100
    f["labels"] = labels
    return f.to(device)
