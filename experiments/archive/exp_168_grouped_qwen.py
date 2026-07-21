from __future__ import annotations

import io
import json
import pathlib as pl
import sys
import typing as tp

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(pl.Path(__file__).parent))

from grouper_features import DATES, features, load_regions, load_gt, align_regions_to_gt, start_labels

PRED_DIR = pl.Path("eval/predictions")
IMG_DIR = pl.Path("eval/ground_truth")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_ENGINE: dict[str, tp.Any] = {}


def _engine() -> tuple[tp.Any, tp.Any]:
    if "llm" not in _ENGINE:
        from transformers import AutoProcessor
        from vllm import LLM

        _ENGINE["llm"] = LLM(model=MODEL, trust_remote_code=True, gpu_memory_utilization=0.90,
                             max_model_len=8192, limit_mm_per_prompt={"image": 1}, dtype="bfloat16", seed=0)
        _ENGINE["proc"] = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    return _ENGINE["llm"], _ENGINE["proc"]


def _crops(date: str, regions: list[dict[str, tp.Any]]) -> list[Image.Image]:
    pages: dict[int, Image.Image] = {}
    out: list[Image.Image] = []
    for r in regions:
        pg = int(r["page"])
        if pg not in pages:
            pages[pg] = Image.open(IMG_DIR / date / f"{pg}.jpeg").convert("RGB")
        box = [int(v) for v in r["bbox"]] if isinstance(r["bbox"], list) else json.loads(r["bbox"])
        out.append(pages[pg].crop(box))
    return out


def ocr_regions(date: str, regions: list[dict[str, tp.Any]]) -> list[str]:
    from vllm import SamplingParams

    llm, proc = _engine()
    prompts = []
    for img in _crops(date, regions):
        w, h = img.size
        if max(w, h) > 1600:
            s = 1600 / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                     {"type": "text", "text": "Transcribe all the text in this image exactly. Output only the transcription."}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append({"prompt": text, "multi_modal_data": {"image": img}})
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


def _group(regions: list[dict[str, tp.Any]], texts: list[str], starts: list[int]) -> list[dict[str, tp.Any]]:
    articles: list[dict[str, tp.Any]] = []
    cur: list[tuple[dict[str, tp.Any], str]] = []

    def flush() -> None:
        if not cur:
            return
        title = next((t.strip() for r, t in cur if r["class"] == "title" and t.strip()), None)
        body = [t.strip() for r, t in cur if not (r["class"] == "title" and t.strip() == title) and t.strip()]
        pages = sorted({int(r["page"]) for r, _ in cur})
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


def predict_issue(date: str) -> dict[str, tp.Any]:
    regions = load_regions(date)
    texts = ocr_regions(date, regions)
    clf = _train(exclude=date)
    starts = clf.predict(np.array(features(regions))).tolist()
    if regions:
        starts[0] = 1
    return {"date": date, "source": "exp_168_grouped_qwen", "articles": _group(regions, texts, starts)}


def main() -> None:
    for date in (sys.argv[1:] or list(DATES)):
        pred = predict_issue(date)
        out = PRED_DIR / f"exp_168_grouped_qwen_{date}.json"
        out.write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles -> {out}", flush=True)


if __name__ == "__main__":
    main()
