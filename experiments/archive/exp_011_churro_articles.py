from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import gc
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
PADDLE_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
CHURRO_MODEL = "stanford-oval/churro-3B"
_ENGINE: dict[str, tp.Any] = {}

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
TEXT_LABELS = {
    "text",
    "paragraph_title",
    "doc_title",
    "title",
    "abstract",
    "content",
    "figure_title",
    "table_title",
    "chart_title",
    "vision_footnote",
}


def _pages(date: str) -> list[pl.Path]:
    return sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))


def detect_layout(page_paths: list[str]) -> list[list[dict[str, tp.Any]]]:
    proc = subprocess.run([str(PADDLE_PYTHON), "-c", LAYOUT_SCRIPT], input=json.dumps(page_paths), capture_output=True, text=True, timeout=1800)
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


def _load_paddle() -> None:
    from transformers import AutoProcessor
    from vllm import LLM

    _ENGINE["llm"] = LLM(
        model=PADDLE_MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        enable_prefix_caching=False,
        seed=0,
    )
    _ENGINE["proc"] = AutoProcessor.from_pretrained(PADDLE_MODEL, trust_remote_code=True)
    _ENGINE["prompt"] = "OCR:"
    _ENGINE["cap"] = 1600


def _load_churro() -> None:
    from transformers import AutoProcessor
    from vllm import LLM

    _ENGINE["llm"] = LLM(
        model=CHURRO_MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        mm_processor_kwargs={"max_pixels": 2300000},
        enable_prefix_caching=False,
        max_num_seqs=8,
        seed=0,
    )
    _ENGINE["proc"] = AutoProcessor.from_pretrained(CHURRO_MODEL, trust_remote_code=True)
    _ENGINE["prompt"] = (
        "Transcribe all the text in reading order exactly as printed, preserving the original spelling. Output only the transcription."
    )
    _ENGINE["cap"] = 2000
    _ENGINE["sys"] = "You are an expert in diplomatic transcription of historical documents from various languages."


def _free_engine() -> None:
    import torch

    for k in ("llm", "proc", "prompt", "cap", "sys"):
        _ENGINE.pop(k, None)
    gc.collect()
    torch.cuda.empty_cache()


def ocr_crops(crops: list[Image.Image]) -> list[str]:
    from vllm import SamplingParams

    llm, proc, prompt, cap = _ENGINE["llm"], _ENGINE["proc"], _ENGINE["prompt"], _ENGINE["cap"]
    sys_msg = _ENGINE.get("sys")
    prompts = []
    for img in crops:
        w, h = img.size
        if max(w, h) > cap:
            s = cap / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        content = [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
        messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": content}]
        prompts.append(
            {"prompt": proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), "multi_modal_data": {"image": img}}
        )
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


def _group_indices(starts: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, s in enumerate(starts):
        if s and cur:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _article_crops(date: str, flat: list[dict], groups: list[list[int]]) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    page_img: dict[int, Image.Image] = {int(p.stem): Image.open(p).convert("RGB") for p in _pages(date)}
    crops: list[Image.Image] = []
    cmap: list[tuple[int, int]] = []
    for gi, grp in enumerate(groups):
        by_page: dict[int, list[list[int]]] = {}
        for idx in grp:
            by_page.setdefault(flat[idx]["page"], []).append(flat[idx]["bbox"])
        for page, boxes in sorted(by_page.items()):
            img = page_img[page]
            pad = 12
            x1 = max(0, min(b[0] for b in boxes) - pad)
            y1 = max(0, min(b[1] for b in boxes) - pad)
            x2 = min(img.width, max(b[2] for b in boxes) + pad)
            y2 = min(img.height, max(b[3] for b in boxes) + pad)
            crops.append(img.crop((x1, y1, x2, y2)))
            cmap.append((gi, page))
    return crops, cmap


def _assemble(flat, groups, region_texts, group_text, group_pages) -> list[dict]:
    articles: list[dict] = []
    for gi, grp in enumerate(groups):
        title = next((region_texts[idx].strip() for idx in grp if flat[idx]["class"] == "title" and region_texts[idx].strip()), None)
        art_text = "\n".join(x for x in group_text.get(gi, []) if x)
        region_concat = "\n".join(region_texts[idx].strip() for idx in grp if region_texts[idx].strip() and region_texts[idx].strip() != title)
        gb: dict[int, list] = {}
        for idx in grp:
            gb.setdefault(flat[idx]["page"], []).append(flat[idx]["bbox"])
        fills = []
        for boxes in gb.values():
            ux1 = min(b[0] for b in boxes)
            uy1 = min(b[1] for b in boxes)
            ux2 = max(b[2] for b in boxes)
            uy2 = max(b[3] for b in boxes)
            uarea = max(1, (ux2 - ux1) * (uy2 - uy1))
            rarea = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
            fills.append(rarea / uarea)
        fill = min(fills) if fills else 1.0
        text = art_text if (fill >= 0.5 and art_text.strip()) else region_concat
        if not text.strip():
            text = art_text or region_concat
        if not text.strip():
            continue
        pgs = sorted(group_pages.get(gi, {flat[grp[0]]["page"]}))
        articles.append(
            {
                "unit_type": "article",
                "headline": title,
                "paragraphs": [{"text": text}],
                "page_span": [pgs[0], pgs[-1]] if len(pgs) > 1 else [pgs[0]],
            }
        )
    return articles


def main() -> None:
    dates = sys.argv[1:] or list(DATES)
    state: dict[str, tp.Any] = {}
    _load_paddle()
    for date in dates:
        layout = detect_layout([str(p) for p in _pages(date)])
        regions_per_page = [build_regions(boxes, i + 1) for i, boxes in enumerate(layout)]
        crops, flat = crop_regions(date, regions_per_page)
        region_texts = ocr_crops(crops)
        for r, t in zip(flat, region_texts):
            r["text"] = t
        clf = _train(exclude=date)
        starts = clf.predict(np.array(features(flat))).tolist()
        if flat:
            starts[0] = 1
        groups = _group_indices(starts)
        art_crops, cmap = _article_crops(date, flat, groups)
        state[date] = {"flat": flat, "region_texts": region_texts, "groups": groups, "art_crops": art_crops, "cmap": cmap}
        print(f"{date}: {len(groups)} articles (region phase)", flush=True)

    _free_engine()
    _load_churro()
    for date in dates:
        st = state[date]
        art_texts = ocr_crops(st["art_crops"])
        group_text: dict[int, list[str]] = {}
        group_pages: dict[int, set[int]] = {}
        for (gi, page), t in zip(st["cmap"], art_texts):
            group_text.setdefault(gi, []).append(t.strip())
            group_pages.setdefault(gi, set()).add(page)
        articles = _assemble(st["flat"], st["groups"], st["region_texts"], group_text, group_pages)
        (PRED_DIR / f"exp_011_churro_articles_{date}.json").write_text(
            json.dumps({"date": date, "source": "exp_011_churro_articles", "articles": articles}, ensure_ascii=False)
        )
        print(f"{date}: {len(articles)} articles (churro phase)", flush=True)


if __name__ == "__main__":
    main()
