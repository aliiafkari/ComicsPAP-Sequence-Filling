"""Render one ComicsPAP sequence-filling sample as a SINGLE composite image (the input format of the released
Qwen2.5-VL baselines).  Place at  src/data/pap_render.py.

Layout (top to bottom): the label "Context", a row with the context panels and a grey "?" panel inserted at the missing
position, the label "Options", a row with the option panels, each with its 0-BASED index drawn in the bottom-right corner
(the model's answer is that number). Every panel is first limited to a longest edge of `max_size` px and then resized to the
median panel size. This is an independent implementation written from the format described on the dataset card; it was
checked to give pixel-identical output to the official `SingleImagePickAPanel` for the same font.

The official baselines used Arial; pass --font C:/Windows/Fonts/arial.ttf (the default search finds it on Windows).
"""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                    "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]


def find_font(font_path: Optional[str] = None) -> str:
    """Arial if available (what the released baselines used), else DejaVu Sans shipped with matplotlib."""
    for c in ([font_path] if font_path else []) + _FONT_CANDIDATES:
        if c and Path(c).exists():
            return str(c)
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if p.exists():
            print(f"[warn] Arial not found; using {p}. The digits drawn on the options will look slightly different "
                  "from the training images of the released adapter (pass --font to fix).")
            return str(p)
    except ImportError:
        pass
    raise FileNotFoundError("no usable font found: pass --font path/to/Arial.ttf")


class SingleImageRenderer:
    def __init__(self, font_path: Optional[str] = None, max_size: int = 500, margin: int = 10, label_space: int = 20,
                 label_font_size: int = 20, number_font_size: int = 24):
        self.font_path = find_font(font_path)
        self.max_size, self.margin, self.label_space = max_size, margin, label_space
        self.label_font = ImageFont.truetype(self.font_path, label_font_size)
        self.number_font = ImageFont.truetype(self.font_path, number_font_size)

    # -- pieces -------------------------------------------------------------------------------------------
    def _limit(self, img: Image.Image) -> Image.Image:
        if max(img.size) > self.max_size:
            r = self.max_size / max(img.size)
            return img.resize(tuple(int(d * r) for d in img.size), Image.Resampling.LANCZOS)
        return img

    def _mask_panel(self, w: int, h: int) -> Image.Image:
        panel = Image.new("RGB", (w, h), (200, 200, 200))
        draw = ImageDraw.Draw(panel)
        font = ImageFont.truetype(self.font_path, int(h * 0.8))
        box = draw.textbbox((0, 0), "?", font=font)
        draw.text(((w - (box[2] - box[0])) // 2, (h - (box[3] - box[1])) // 2), "?", fill="black", font=font)
        return panel

    def _numbered(self, panel: Image.Image, number: int) -> Image.Image:
        p = panel.copy().convert("RGBA")
        draw = ImageDraw.Draw(p)
        box = draw.textbbox((0, 0), str(number), font=self.number_font)
        tw, th, pad = box[2] - box[0], box[3] - box[1], 2
        x, y = p.size[0] - tw - pad, p.size[1] - th - pad
        draw.rectangle([(x - pad, y - pad), (x + tw + pad, y + th + pad)], fill=(255, 255, 255, 180))
        draw.text((x, y), str(number), fill="black", font=self.number_font)
        return p

    # -- main ---------------------------------------------------------------------------------------------
    def render(self, context: Sequence[Image.Image], options: Sequence[Image.Image], mask_idx: int) -> Image.Image:
        ctx = [self._limit(i) for i in context]
        opt = [self._limit(i) for i in options]
        allp = ctx + opt
        pw = int(np.median([i.size[0] for i in allp]))
        ph = int(np.median([i.size[1] for i in allp]))
        ctx = [i.resize((pw, ph)) for i in ctx]
        opt = [i.resize((pw, ph)) for i in opt]
        ctx.insert(mask_idx, self._mask_panel(pw, ph))

        m, ls = self.margin, self.label_space
        row_c, row_o = pw * len(ctx) + m * (len(ctx) - 1), pw * len(opt) + m * (len(opt) - 1)
        width, height = max(row_c, row_o), ph * 2 + ls * 2
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)

        y = 0
        box = draw.textbbox((0, 0), "Context", font=self.label_font)
        draw.text(((width - (box[2] - box[0])) // 2, y), "Context", fill="black", font=self.label_font)
        y += ls
        x = (width - row_c) // 2
        for p in ctx:
            canvas.paste(p, (x, y))
            x += pw + m
        y += ph
        box = draw.textbbox((0, 0), "Options", font=self.label_font)
        draw.text(((width - (box[2] - box[0])) // 2, y), "Options", fill="black", font=self.label_font)
        y += ls
        x = (width - row_o) // 2
        for i, p in enumerate(opt):
            q = self._numbered(p, i)
            canvas.paste(q, (x, y), q)
            x += pw + m
        return canvas


def load_panels(root, rels: Sequence[str]) -> List[Image.Image]:
    out = []
    for r in rels:
        with Image.open(Path(root) / r) as im:
            out.append(im.convert("RGB"))
    return out


def render_record(renderer: SingleImageRenderer, rec: dict, root, option_order: Optional[Sequence[int]] = None
                  ) -> Tuple[Image.Image, int]:
    """Render a manifest record. `option_order[k]` = original index of the option shown at position k (augmentation /
    robustness). Returns (image, label_in_displayed_order); label is -1 when the answer is hidden."""
    ctx = load_panels(root, rec["context_panels"])
    opts = load_panels(root, rec["candidates"])
    order = list(range(len(opts))) if option_order is None else list(option_order)
    shown = [opts[j] for j in order]
    label = rec.get("label", -1)
    new_label = order.index(label) if label >= 0 else -1
    return renderer.render(ctx, shown, rec["missing_position"]), new_label
