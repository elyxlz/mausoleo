from __future__ import annotations

import json
import pathlib as pl
import subprocess
import sys
import time
import typing as tp

sys.path.insert(0, "src")

EXP_NAME = pl.Path(__file__).stem
GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
PREDICTIONS_DIR = pl.Path("eval/predictions")
PADDLE_PYTHON = pl.Path.home() / "paddle_env" / "bin" / "python"

LAYOUT_SCRIPT = r"""
import json
import sys

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


def detect_layout(page_paths: list[str]) -> list[list[dict[str, tp.Any]]]:
    proc = subprocess.run(
        [str(PADDLE_PYTHON), "-c", LAYOUT_SCRIPT],
        input=json.dumps(page_paths),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"layout subprocess failed:\n{proc.stderr[-3000:]}")
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


def crop_regions(date: str, regions_per_page: list[list[dict[str, tp.Any]]]) -> tuple[list[bytes], list[dict[str, tp.Any]]]:
    import io

    from PIL import Image

    crops: list[bytes] = []
    flat_regions: list[dict[str, tp.Any]] = []
    pages = sorted(GROUND_TRUTH_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    for page_path, regions in zip(pages, regions_per_page):
        img = Image.open(page_path)
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            pad = 15
            box = (max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad))
            crop = img.crop(box)
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=95)
            crops.append(buf.getvalue())
            flat_regions.append(region)
    return crops, flat_regions


_ENGINE: dict[str, tp.Any] = {}


def _get_engine() -> tuple[tp.Any, tp.Any]:
    if "llm" not in _ENGINE:
        from transformers import AutoProcessor
        from vllm import LLM

        _ENGINE["llm"] = LLM(
            model="PaddlePaddle/PaddleOCR-VL-1.6",
            trust_remote_code=True,
            gpu_memory_utilization=0.92,
            max_model_len=16384,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=False,
            dtype="bfloat16",
            enable_prefix_caching=False,
            seed=0,
        )
        _ENGINE["processor"] = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL-1.6", trust_remote_code=True)
    return _ENGINE["llm"], _ENGINE["processor"]


def ocr_crops(crops: list[bytes]) -> list[str]:
    import io

    from PIL import Image
    from vllm import SamplingParams

    llm, processor = _get_engine()
    prompts = []
    for crop in crops:
        img = Image.open(io.BytesIO(crop))
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR:"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append({"prompt": text, "multi_modal_data": {"image": img}})
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4096))
    return [out.outputs[0].text for out in outputs]


def _continues_flow(prev_bbox: list[int], new_bbox: list[int], page_h: int, same_page: bool) -> bool:
    if not same_page:
        return False
    px1, py1, px2, py2 = prev_bbox
    nx1, ny1, nx2, ny2 = new_bbox
    overlap_x = min(px2, nx2) - max(px1, nx1)
    min_w = max(1, min(px2 - px1, nx2 - nx1))
    if overlap_x / min_w >= 0.5 and 0 <= ny1 - py2 <= 0.03 * page_h:
        return True
    if nx1 > px2 - 0.2 * min_w and py2 >= 0.75 * page_h and ny1 <= 0.25 * page_h:
        return True
    return False


def assemble(date: str, page_count: int, texts: list[str], regions: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
    from mausoleo.ocr.operators.merge import squeeze_char_runs

    page_heights: dict[int, int] = {}
    for region in regions:
        page_heights[region["page"]] = max(page_heights.get(region["page"], 0), region["bbox"][3])

    articles: list[dict[str, tp.Any]] = []
    current: dict[str, tp.Any] | None = None
    prev_region: dict[str, tp.Any] | None = None
    for text, region in zip(texts, regions):
        text = squeeze_char_runs(text).strip()
        if not text:
            continue
        if region["class"] == "title":
            headline = " ".join(text.split())[:200]
            if current is not None and current["headline"] is not None and not current["paragraphs"]:
                current["headline"] = f"{current['headline']}\n{headline}"[:300]
                if region["page"] not in current["page_span"]:
                    current["page_span"].append(region["page"])
                prev_region = region
                continue
            if current is not None:
                articles.append(current)
            current = {"unit_type": "article", "headline": headline, "paragraphs": [], "page_span": [region["page"]]}
        else:
            attach = current is not None and (
                current["headline"] is not None
                or (
                    prev_region is not None
                    and _continues_flow(
                        prev_region["bbox"], region["bbox"], page_heights.get(region["page"], 4000), prev_region["page"] == region["page"]
                    )
                )
            )
            if not attach:
                if current is not None:
                    articles.append(current)
                current = {"unit_type": "article", "headline": None, "paragraphs": [], "page_span": [region["page"]]}
            current["paragraphs"].append({"text": text})
            if region["page"] not in current["page_span"]:
                current["page_span"].append(region["page"])
        prev_region = region
    if current is not None:
        articles.append(current)

    articles = [a for a in articles if a["paragraphs"] or a["headline"]]
    for a in articles:
        if not a["paragraphs"]:
            a["paragraphs"] = [{"text": a["headline"]}]
    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        art["page_span"] = sorted(art["page_span"])
        for p_idx, para in enumerate(art["paragraphs"]):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    return {"date": date, "source": "il_messaggero", "page_count": page_count, "articles": articles}


def run_date(date: str) -> tuple[float, int]:
    pages = sorted(GROUND_TRUTH_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    if not pages:
        raise SystemExit(f"no images at {GROUND_TRUTH_DIR / date}")
    t0 = time.time()
    layout = detect_layout([str(p) for p in pages])
    regions_per_page = [build_regions(boxes, i + 1) for i, boxes in enumerate(layout)]
    layout_s = time.time() - t0
    crops, flat_regions = crop_regions(date, regions_per_page)
    t1 = time.time()
    texts = ocr_crops(crops)
    ocr_s = time.time() - t1
    issue = assemble(date, len(pages), texts, flat_regions)
    out = PREDICTIONS_DIR / f"{EXP_NAME}_{date}.json"
    out.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    total = time.time() - t0
    print(
        f"{date}: {len(issue['articles'])} articles from {len(crops)} regions | "
        f"layout {layout_s:.1f}s + ocr {ocr_s:.1f}s = {total:.1f}s | {total / len(pages):.2f} s/page -> {out}"
    )
    return total, len(pages)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {EXP_NAME}.py <date> [<date> ...]")
    timings = [run_date(date) for date in sys.argv[1:]]
    if len(timings) > 1:
        steady = timings[1:]
        total_s = sum(t for t, _ in steady)
        total_pp = sum(n for _, n in steady)
        gpu_s_page = total_s / total_pp
        days = 175_000 * gpu_s_page / 2 / 86400
        print(f"steady-state (excl. first issue): {gpu_s_page:.2f} GPU-s/page over {total_pp}pp")
        print(f"corpus extrapolation: 175000 pages on 2 GPUs = {days:.1f} days (budget: 7-14)")


if __name__ == "__main__":
    main()
