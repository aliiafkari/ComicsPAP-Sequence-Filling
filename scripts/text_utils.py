"""Helpers shared by the OCR scripts (04b spot-check, 05 text embeddings).

Pure Python: no paddle / torch imports, so it works in both the PaddleOCR and the training environment.
Place this file at  src/ocr/text_utils.py  in the repository.

A detection is a dict: {"text": str, "confidence": float, "bbox": [x_min, y_min, x_max, y_max]}.
"""
from typing import Dict, List


def clean_text(s) -> str:
    return " ".join(str(s).split())


def _yc(d: Dict) -> float:
    b = d["bbox"]
    return (b[1] + b[3]) / 2.0


def _h(d: Dict) -> float:
    b = d["bbox"]
    return max(1.0, float(b[3] - b[1]))


def reading_order(dets: List[Dict], ltr: bool = True, y_tol: float = 0.6) -> List[Dict]:
    """Sort detections top-to-bottom, then left-to-right within a text line.

    Detections whose vertical centres are within `y_tol` x median box height are treated as the
    same line. This is plain line ordering, not speech-bubble ordering: in panels with several
    bubbles side by side it can interleave their lines. Comic-specific tools (e.g. Magi) model
    bubble order explicitly.
    """
    dets = list(dets)
    if len(dets) <= 1:
        return dets
    hs = sorted(_h(d) for d in dets)
    med = hs[len(hs) // 2]
    lines = []  # [mean_yc, [dets]]
    for d in sorted(dets, key=_yc):
        yc = _yc(d)
        if lines and abs(yc - lines[-1][0]) <= y_tol * med:
            lines[-1][1].append(d)
            lines[-1][0] = sum(_yc(x) for x in lines[-1][1]) / len(lines[-1][1])
        else:
            lines.append([yc, [d]])
    out: List[Dict] = []
    for _, items in lines:
        out.extend(sorted(items, key=lambda d: d["bbox"][0], reverse=not ltr))
    return out


def select_dets(dets: List[Dict], min_conf: float = 0.0, min_chars: int = 1) -> List[Dict]:
    return [d for d in dets
            if float(d.get("confidence", 0.0)) >= min_conf and len(clean_text(d.get("text", ""))) >= min_chars]


def panel_text(dets: List[Dict], order: str = "raw", min_conf: float = 0.0, min_chars: int = 1) -> str:
    """Concatenate the kept detections of one panel into a single string."""
    kept = select_dets(dets, min_conf, min_chars)
    if order == "reading":
        kept = reading_order(kept)
    return " ".join(clean_text(d["text"]) for d in kept).strip()


def panel_confidence(dets: List[Dict], min_conf: float = 0.0, min_chars: int = 1) -> float:
    kept = select_dets(dets, min_conf, min_chars)
    return sum(float(d["confidence"]) for d in kept) / len(kept) if kept else 0.0
