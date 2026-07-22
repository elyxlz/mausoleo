from __future__ import annotations

import io
import json
import pathlib as pl
import re
import sys
import typing as tp

from PIL import Image

sys.path.insert(0, "src")

from mausoleo.ocr.prompts import VLM_OCR_STRUCTURED_V2

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]
NUM_COLUMNS = 3
OVERLAP_PCT = 0.03
HEADER_CROP_PCT = 0.03
FOOTER_CROP_PCT = 0.02
_ENGINE: dict[str, tp.Any] = {}


def _pages(date: str) -> list[pl.Path]:
    return sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))


def column_crops(date: str) -> list[tuple[int, Image.Image]]:
    out: list[tuple[int, Image.Image]] = []
    for page_path in _pages(date):
        img = Image.open(page_path).convert("RGB")
        w, h = img.size
        top = int(h * HEADER_CROP_PCT)
        bottom = int(h * (1 - FOOTER_CROP_PCT))
        col_width = w / NUM_COLUMNS
        overlap = int(col_width * OVERLAP_PCT)
        page_num = int(page_path.stem)
        for col in range(NUM_COLUMNS):
            x1 = max(0, int(col * col_width) - overlap)
            x2 = min(w, int((col + 1) * col_width) + overlap)
            out.append((page_num, img.crop((x1, top, x2, bottom))))
    return out


def _engine() -> tuple[tp.Any, tp.Any]:
    if "llm" not in _ENGINE:
        from transformers import AutoProcessor
        from vllm import LLM

        _ENGINE["llm"] = LLM(model=MODEL, trust_remote_code=True, gpu_memory_utilization=0.90,
                             max_model_len=12288, limit_mm_per_prompt={"image": 1}, dtype="bfloat16",
                             enable_prefix_caching=False, max_num_seqs=16, seed=0)
        _ENGINE["proc"] = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    return _ENGINE["llm"], _ENGINE["proc"]


def ocr_columns(crops: list[Image.Image]) -> list[str]:
    from vllm import SamplingParams

    llm, proc = _engine()
    prompts = []
    for img in crops:
        w, h = img.size
        if max(w, h) > 2200:
            s = 2200 / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                     {"type": "text", "text": VLM_OCR_STRUCTURED_V2}]}]
        prompts.append({"prompt": proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                        "multi_modal_data": {"image": img}})
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8192, repetition_penalty=1.05))
    return [o.outputs[0].text.strip() for o in outputs]


def _parse(text: str) -> list[dict[str, tp.Any]]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    for cand in (t, t[t.find("{"):] if "{" in t else t):
        try:
            obj = json.loads(cand)
            arts = obj.get("articles", obj) if isinstance(obj, dict) else obj
            if isinstance(arts, list):
                return arts
        except (json.JSONDecodeError, AttributeError):
            continue
    return [{"unit_type": "article", "headline": None, "text": text}]


def run_date(date: str) -> dict[str, tp.Any]:
    crops = column_crops(date)
    texts = ocr_columns([c for _, c in crops])
    articles: list[dict[str, tp.Any]] = []
    for (page, _), raw in zip(crops, texts):
        for art in _parse(raw):
            if not isinstance(art, dict):
                art = {"text": str(art)}
            paras = art.get("paragraphs") or ([{"text": art["text"]}] if art.get("text") else [])
            paras = [{"text": p.get("text", str(p)) if isinstance(p, dict) else str(p)} for p in paras]
            if not any(p["text"].strip() for p in paras):
                continue
            articles.append({"unit_type": art.get("unit_type", "article"),
                             "headline": art.get("headline"), "paragraphs": paras, "page_span": [page]})
    return {"date": date, "source": "exp_014_column8b_full", "articles": articles}


def main() -> None:
    for date in (sys.argv[1:] or DATES):
        pred = run_date(date)
        (PRED_DIR / f"exp_014_column8b_full_{date}.json").write_text(json.dumps(pred, ensure_ascii=False))
        print(f"{date}: {len(pred['articles'])} articles", flush=True)


if __name__ == "__main__":
    main()
