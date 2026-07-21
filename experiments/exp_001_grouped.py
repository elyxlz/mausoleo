from __future__ import annotations

import io
import json
import pathlib as pl
import subprocess
import sys
import typing as tp

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(pl.Path(__file__).parent))
sys.path.insert(0, "src")

from grouper_features import DATES, features, load_regions, load_gt, align_regions_to_gt, start_labels

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
PADDLE_PYTHON = pl.Path.home() / "paddle_env" / "bin" / "python"
MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"

LAYOUT_SCRIPT = r"""
import json, sys
from paddleocr import LayoutDetection
model = LayoutDetection(model_name="PP-DocLayoutV3")
pages = json.loads(sys.stdin.read())
results = []
for path in pages:
    output = model.predict(path, batch_size=1)
    boxes = []
    for res in output:
        for box in res.json["res"]["boxes"]:
            boxes.append({"label": box["label"], "score": float(box["score"]), "coordinate": [float(c) for c in box["coordinate"]]})
    results.append(boxes)
print(json.dumps(results))
"""

TITLE_LABELS = {"doc_title", "paragraph_title", "title"}
TEXT_LABELS = {"text", "paragraph_title", "doc_title", "title", "abstract", "content", "figure_title", "table_title", "chart_title", "vision_footnote"}
_ENGINE: dict[str, tp.Any] = {}


def _pages(date: str) -> list[pl.Path]:
    return sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))


def detect_layout(page_paths: list[str]) -> list[list[dict[str, tp.Any]]]:
    proc = subprocess.run([str(PADDLE_PYTHON), "-c", LAYOUT_SCRIPT], input=json.dumps(page_paths),
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"layout failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def build_regions(page_boxes: list[dict[str, tp.Any]], page_num: int) -> list[dict[str, tp.Any]]:
    regions = []
    for box in page_boxes:
        if box["label"] not in TEXT_LABELS:
            continue
        x1, y1, x2, y2 = box["coordinate"]
        if (x2 - x1) * (y2 - y1) < 1500:
            continue
        cls = "title" if box["label"] in TITLE_LABELS else "text"
        regions.append({"page": page_num, "bbox": [int(x1), int(y1), int(x2), int(y2)], "class": cls})
    return regions


def crop_regions(date: str, regions_per_page: list[list[dict[str, tp.Any]]]) -> tuple[list[Image.Image], list[dict[str, tp.Any]]]:
    crops: list[Image.Image] = []
    flat: list[dict[str, tp.Any]] = []
    for page_path, regions in zip(_pages(date), regions_per_page):
        img = Image.open(page_path).convert("RGB")
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            pad = 15
            crops.append(img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad))))
            flat.append(r)
    return crops, flat


def _engine() -> tuple[tp.Any, tp.Any]:
    if "llm" not in _ENGINE:
        from transformers import AutoProcessor
        from vllm import LLM

        _ENGINE["llm"] = LLM(model=MODEL, trust_remote_code=True, gpu_memory_utilization=0.90,
                             max_model_len=16384, limit_mm_per_prompt={"image": 1}, dtype="bfloat16",
                             enable_prefix_caching=False, seed=0)
        _ENGINE["proc"] = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    return _ENGINE["llm"], _ENGINE["proc"]


def ocr_crops(crops: list[Image.Image]) -> list[str]:
    from vllm import SamplingParams

    llm, proc = _engine()
    prompts = []
    for img in crops:
        w, h = img.size
        if max(w, h) > 1600:
            s = 1600 / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                     {"type": "text", "text": "OCR:"}]}]
        prompts.append({"prompt": proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                        "multi_modal_data": {"image": img}})
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2048))
    return [o.outputs[0].text.strip() for o in outputs]


def _train(exclude: str) -> GradientBoostingClassifier:
    X: list[list[float]] = []
    y: list[int] = []
    for d in DATES:
        if d == exclude:
            continue
        rg = load_regions(d)
        X.extend(features(rg))
        y.extend(start_labels(rg, align_regions_to_gt(rg, load_gt(d))))
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05).fit(np.array(X), np.array(y))


def _group(regions: list[dict], texts: list[str], starts: list[int]) -> list[dict]:
    articles: list[dict] = []
    cur: list[tuple[dict, str]] = []

    def flush() -> None:
        if not cur:
            return
        title = next((t.strip() for r, t in cur if r["class"] == "title" and t.strip()), None)
        body = [t.strip() for r, t in cur if not (r["class"] == "title" and t.strip() == title) and t.strip()]
        pages = sorted({r["page"] for r, _ in cur})
        articles.append({"unit_type": "article", "headline": title,
                         "paragraphs": [{"text": "\n".join(body or [t for _, t in cur])}],
                         "page_span": [pages[0], pages[-1]] if len(pages) > 1 else [pages[0]]})

    for (r, t), s in zip([(r, texts[i]) for i, r in enumerate(regions)], starts):
        if s and cur:
            flush()
            cur = []
        cur.append((r, t))
    flush()
    return articles


def run_date(date: str) -> dict:
    pages = [str(p) for p in _pages(date)]
    layout = detect_layout(pages)
    regions_per_page = [build_regions(boxes, i + 1) for i, boxes in enumerate(layout)]
    crops, flat = crop_regions(date, regions_per_page)
    texts = ocr_crops(crops)
    for r, t in zip(flat, texts):
        r["text"] = t
    clf = _train(exclude=date)
    starts = clf.predict(np.array(features(flat))).tolist()
    if flat:
        starts[0] = 1
    return {"date": date, "source": "exp_001_grouped", "articles": _group(flat, texts, starts)}


def main() -> None:
    for date in (sys.argv[1:] or list(DATES)):
        pred = run_date(date)
        (PRED_DIR / f"exp_001_grouped_{date}.json").write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles", flush=True)


if __name__ == "__main__":
    main()
