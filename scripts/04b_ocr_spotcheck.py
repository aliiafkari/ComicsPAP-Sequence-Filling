"""Step 4b: check OCR quality on a random panel subset with a point-and-click labeling page, then score it.

Needs only pillow (run in either environment). Uses src/ocr/text_utils.py.

1) Build the labeling page
    python scripts/04b_ocr_spotcheck.py sample --ocr_dir D:/new_comic/ocr/ppocrv6_medium --split validation ^
        --data_root D:/new_comic --out_dir D:/new_comic/ocr/spotcheck --n 60 --seed 0

2) Open <out_dir>/index.html in a browser and label in TWO passes (progress is auto-saved in the browser):
   PASS 1 - blind transcription. For every panel type ALL legible dialogue/caption text in reading order, or tick
            "panel has no text" / "skip". The OCR output is hidden so it cannot bias you. Decide once whether sound
            effects and background signs count, and keep that rule for all panels.
   PASS 2 - press "Reveal OCR". Boxes appear on the image; for every detection pick "text" (a real text region) or
            "not text / garbage", and enter how many real text regions have NO box ("missed"). This gives detection
            precision / recall.
   Press "Download labels.json" (any time; "Load labels.json" resumes from a file). Save it next to index.html.

3) Score
    python scripts/04b_ocr_spotcheck.py score --labels D:/new_comic/ocr/spotcheck/labels.json

Reported (for several minimum-confidence thresholds, raw vs reading-order text):
   CER / WER                 corpus level, text lower-cased and whitespace-collapsed
   text-presence P/R/F1      does the OCR fire on panels that have text, and stay silent on those that do not
   detection precision/recall  region level, from your pass-2 judgments (a confidence threshold removes detections:
                             valid ones removed count as misses). Also the share of garbage detections per confidence bin.

Panels are sampled uniformly from unique images, so only a few have no text; with n=60 treat differences of a few
points as noise (use --n 150 for stable numbers). Annotator effort is the limit, not the code.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import panel_confidence, panel_text # noqa: E402

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f4f6;color:#222}
 #bar{position:sticky;top:0;z-index:5;background:#1f2937;color:#fff;padding:8px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #bar button,#bar label.btn{background:#3b82f6;color:#fff;border:0;border-radius:5px;padding:6px 12px;cursor:pointer;font-size:14px}
 #bar button.on{background:#f59e0b}
 #cards{padding:12px}
 .card{background:#fff;border-radius:8px;margin:0 0 14px;padding:10px 14px;box-shadow:0 1px 3px #0003}
 .card.done{border-left:6px solid #22c55e}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .imgs img{max-width:380px;max-height:420px;border:1px solid #ccc;margin-right:8px}
 .ann{display:none}.pass2{display:none}
 body.reveal .ann{display:inline}body.reveal .pass2{display:block}
 textarea{width:430px;height:90px;font-size:14px}
 table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 7px}
 .hint{color:#555;font-size:13px}
</style></head><body>
<div id="bar"><b>OCR spot-check</b><span id="prog"></span>
 <button id="rev">Reveal OCR (pass 2)</button>
 <button id="dl">Download labels.json</button>
 <label class="btn">Load labels.json<input id="ld" type="file" accept=".json" style="display:none"></label></div>
<div id="cards"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const ITEMS = JSON.parse(document.getElementById('data').textContent);
const KEY = 'ocr_spotcheck::' + document.title;
let S = {};
try { S = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { S = {}; }
function st(i) {
  if (!S[i]) S[i] = {gt: '', none: false, skip: false, judg: ITEMS[i].dets.map(() => null), missed: ''};
  return S[i];
}
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function transcribed(s) { return s.skip || s.none || s.gt.trim() !== ''; }
function judged(s) { return s.judg.every(v => v !== null) && s.missed !== '' && !isNaN(parseInt(s.missed)); }
function progress() {
  let a = 0, b = 0;
  ITEMS.forEach((it, i) => { const s = st(i); if (transcribed(s)) a++; if (judged(s)) b++;
    const c = document.getElementById('card-' + i); if (c) c.classList.toggle('done', transcribed(s)); });
  document.getElementById('prog').textContent = 'transcribed ' + a + '/' + ITEMS.length + '  |  detections judged ' + b + '/' + ITEMS.length;
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} progress(); }
function build() {
  const root = document.getElementById('cards'); root.innerHTML = '';
  ITEMS.forEach((it, i) => {
    const s = st(i);
    const rows = it.dets.map((d, k) =>
      '<tr><td>' + k + '</td><td>' + d.confidence.toFixed(2) + '</td><td>' + esc(d.text) + '</td><td>' +
      '<label><input type="radio" name="j-' + i + '-' + k + '" value="valid" data-i="' + i + '" data-k="' + k + '"' + (s.judg[k] === 'valid' ? ' checked' : '') + '> text</label> ' +
      '<label><input type="radio" name="j-' + i + '-' + k + '" value="fp" data-i="' + i + '" data-k="' + k + '"' + (s.judg[k] === 'fp' ? ' checked' : '') + '> not text / garbage</label></td></tr>'
    ).join('') || '<tr><td colspan="4"><i>no detections</i></td></tr>';
    const div = document.createElement('div'); div.className = 'card'; div.id = 'card-' + i;
    div.innerHTML =
      '<h4>#' + it.id + '</h4><div class="row"><div class="imgs"><img src="' + it.original + '"><img class="ann" src="' + it.annotated + '"></div>' +
      '<div><textarea class="gt" data-i="' + i + '" placeholder="Pass 1: type all legible text in reading order">' + esc(s.gt) + '</textarea><br>' +
      '<label><input type="checkbox" class="none" data-i="' + i + '"' + (s.none ? ' checked' : '') + '> panel has no text</label> &nbsp; ' +
      '<label><input type="checkbox" class="skip" data-i="' + i + '"' + (s.skip ? ' checked' : '') + '> skip this panel</label>' +
      '<div class="pass2"><p class="hint">Pass 2: judge every detection; boxes are numbered on the right-hand image.</p>' +
      '<table><tr><th>box</th><th>conf</th><th>OCR text</th><th>judgment</th></tr>' + rows + '</table>' +
      '<p>Real text regions with NO box: <input type="number" min="0" class="missed" data-i="' + i + '" value="' + esc(s.missed) + '" style="width:60px"></p></div></div></div>';
    root.appendChild(div);
  });
  progress();
}
document.getElementById('cards').addEventListener('input', ev => {
  const t = ev.target, i = parseInt(t.dataset.i);
  if (isNaN(i)) return;
  const s = st(i);
  if (t.classList.contains('gt')) s.gt = t.value;
  else if (t.classList.contains('none')) s.none = t.checked;
  else if (t.classList.contains('skip')) s.skip = t.checked;
  else if (t.classList.contains('missed')) s.missed = t.value;
  else if (t.type === 'radio') s.judg[parseInt(t.dataset.k)] = t.value;
  save();
});
document.getElementById('rev').addEventListener('click', () => {
  document.body.classList.toggle('reveal');
  document.getElementById('rev').classList.toggle('on');
});
function exportItems() {
  return ITEMS.map((it, i) => {
    const s = st(i);
    let gt = null;
    if (s.skip) gt = 'SKIP'; else if (s.none) gt = ''; else if (s.gt.trim() !== '') gt = s.gt.trim();
    const mr = (s.missed === '' || isNaN(parseInt(s.missed))) ? null : parseInt(s.missed);
    return Object.assign({}, it, {gt_text: gt, det_judgments: s.judg.slice(), missed_regions: mr});
  });
}
document.getElementById('dl').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(exportItems(), null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels.json';
  document.body.appendChild(a); a.click(); a.remove();
});
document.getElementById('ld').addEventListener('change', ev => {
  const f = ev.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    JSON.parse(r.result).forEach(x => {
      const i = ITEMS.findIndex(it => it.id === x.id); if (i < 0) return;
      const s = st(i);
      s.skip = x.gt_text === 'SKIP'; s.none = x.gt_text === ''; s.gt = (x.gt_text && x.gt_text !== 'SKIP') ? x.gt_text : '';
      if (Array.isArray(x.det_judgments) && x.det_judgments.length === s.judg.length) s.judg = x.det_judgments.slice();
      s.missed = (x.missed_regions === null || x.missed_regions === undefined) ? '' : String(x.missed_regions);
    });
    save(); build();
  };
  r.readAsText(f);
});
build();
</script></body></html>
"""


# --------------------------------------------------------------------------- #
def cmd_sample(args):
    ocr_path = Path(args.ocr_dir) / f"{args.split}.jsonl"
    with open(ocr_path, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh]
    seen, uniq = set(), []
    for r in recs:                      # unique images only
        if r["hash"] not in seen and not r.get("failed"):
            seen.add(r["hash"])
            uniq.append(r)
    picked = random.Random(args.seed).sample(uniq, min(args.n, len(uniq)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    items = []
    for i, r in enumerate(picked):
        src = root / r["image_path"]
        orig, ann = f"orig_{i:03d}.jpg", f"ann_{i:03d}.jpg"
        shutil.copyfile(src, out / orig)
        with Image.open(src) as im:
            img = im.convert("RGB")
        dr = ImageDraw.Draw(img)
        for k, d in enumerate(r["ocr"]):
            x0, y0, x1, y1 = d["bbox"]
            col = (0, 200, 0) if d["confidence"] >= args.show_thr else (230, 0, 0)
            dr.rectangle([x0, y0, x1, y1], outline=col, width=2)
            dr.text((x0 + 2, max(0, y0 - 11)), str(k), fill=col)
        img.save(out / ann, quality=90)
        items.append({"id": i, "image": r["image_path"], "original": orig, "annotated": ann, "hash": r["hash"],
                      "dets": r["ocr"], "gt_text": None, "det_judgments": [None] * len(r["ocr"]), "missed_regions": None})
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    (out / "index.html").write_text(PAGE.replace("__TITLE__", f"OCR spot-check {args.split} seed{args.seed}")
                                    .replace("__DATA__", payload), encoding="utf-8")
    json.dump(items, open(out / "labels.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {len(items)} panels -> {out}/index.html")
    print("Open index.html in a browser, label (pass 1 blind, then 'Reveal OCR' for pass 2), press 'Download labels.json',\n"
          f"save it as {out}/labels.json (replace the empty template), then run the `score` command.")


# --------------------------------------------------------------------------- #
def edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalize(s, strip_punct=False):
    import re
    s = s.lower()
    if strip_punct:
        s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def score_text(items, order, min_conf, strip_punct):
    ce = cl = we = wl = tp = fp = fn = 0
    for it in items:
        gt = normalize(it["gt_text"], strip_punct)
        pr = normalize(panel_text(it["dets"], order=order, min_conf=min_conf), strip_punct)
        ce += edit_distance(pr, gt); cl += len(gt)
        we += edit_distance(pr.split(), gt.split()); wl += len(gt.split())
        if pr and gt: tp += 1
        elif pr and not gt: fp += 1
        elif gt and not pr: fn += 1
    return {"cer": ce / cl if cl else float("nan"), "wer": we / wl if wl else float("nan"), **prf(tp, fp, fn, "presence_")}


def prf(tp, fp, fn, prefix=""):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return {prefix + "precision": p, prefix + "recall": r, prefix + "f1": f}


def score_detection(items, min_conf):
    """Region-level counts from the human judgments; a confidence filter turns dropped valid regions into misses."""
    tp = fp = fn = 0
    for it in items:
        for d, j in zip(it["dets"], it["det_judgments"]):
            if d["confidence"] >= min_conf:
                tp += (j == "valid"); fp += (j == "fp")
            else:
                fn += (j == "valid")
        fn += it["missed_regions"]
    return {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}


def cmd_score(args):
    items = json.load(open(args.labels, encoding="utf-8"))
    done = [it for it in items if it["gt_text"] is not None and str(it["gt_text"]).strip().upper() != "SKIP"]
    n_text = sum(1 for it in done if it["gt_text"].strip())
    print(f"transcribed: {len(done)} panels ({n_text} with text, {len(done) - n_text} without); "
          f"{len(items) - len(done)} unlabelled/skipped")
    if not done:
        raise SystemExit("nothing to score: label the panels first")
    out = {"n_transcribed": len(done), "text": [], "detection": []}

    hdr = f"{'order':<8}{'min_conf':>9}{'CER':>8}{'WER':>8}{'presence P':>12}{'presence R':>12}{'F1':>7}"
    print("\nTEXT QUALITY\n" + hdr + "\n" + "-" * len(hdr))
    for order in ["raw", "reading"]:
        for thr in args.thresholds:
            m = score_text(done, order, thr, args.strip_punct)
            out["text"].append({"order": order, "min_conf": thr, **m})
            print(f"{order:<8}{thr:>9.2f}{m['cer']:>8.3f}{m['wer']:>8.3f}{m['presence_precision']:>12.3f}"
                  f"{m['presence_recall']:>12.3f}{m['presence_f1']:>7.3f}")

    judged = [it for it in done if it.get("det_judgments") is not None and it.get("missed_regions") is not None
              and all(j in ("valid", "fp") for j in it["det_judgments"])]
    print(f"\nDETECTION QUALITY (pass-2 judgments on {len(judged)} panels)")
    if judged:
        hdr = f"{'min_conf':>9}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>11}{'recall':>9}{'F1':>7}"
        print(hdr + "\n" + "-" * len(hdr))
        for thr in args.thresholds:
            m = score_detection(judged, thr)
            out["detection"].append({"min_conf": thr, **m})
            print(f"{thr:>9.2f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}{m['precision']:>11.3f}{m['recall']:>9.3f}{m['f1']:>7.3f}")
        bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]
        print("\nShare of detections judged 'not text / garbage' by confidence bin (helps pick --min_conf):")
        for lo, hi in bins:
            js = [j for it in judged for d, j in zip(it["dets"], it["det_judgments"]) if lo <= d["confidence"] < hi]
            print(f"  conf [{lo:.1f}, {min(hi, 1.0):.1f}]: {len(js):>4} detections, garbage {100 * js.count('fp') / len(js):.0f}%" if js
                  else f"  conf [{lo:.1f}, {min(hi, 1.0):.1f}]:    0 detections")
    else:
        print("  none yet: do pass 2 (Reveal OCR) in index.html, download labels.json, and score again.")
    print("\nCER/WER are corpus-level (total edits / total reference length); lower is better. Few panels -> differences of a "
          "few points are noise. Box-level IoU matching is not used: detection P/R rests on your region judgments.")
    outp = Path(args.labels).with_name("spotcheck_scores.json")
    json.dump(out, open(outp, "w"), indent=1)
    print(f"wrote {outp}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--ocr_dir", required=True)
    s.add_argument("--split", default="validation")
    s.add_argument("--data_root", default="data")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--show_thr", type=float, default=0.8, help="boxes below this confidence are drawn red")
    s.set_defaults(fn=cmd_sample)
    c = sub.add_parser("score")
    c.add_argument("--labels", required=True)
    c.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.8, 0.9])
    c.add_argument("--strip_punct", action="store_true", help="also remove punctuation before comparing")
    c.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
