from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import pathlib as pl
import sys
import time

from mausoleo.ocr import prompts
from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict
from mausoleo.ocr.operators.parse import _build_issue_json
from mausoleo.ocr.operators.merge import _strip_markdown

GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
PREDICTIONS_DIR = pl.Path("eval/predictions")
ISSUE_DATES = ["1885-06-15", "1910-06-15", "1940-04-01"]


def load_issue_images(issue_dir: pl.Path) -> list[bytes]:
    return [f.read_bytes() for f in sorted(issue_dir.glob("*.jpeg"), key=lambda p: int(p.stem))]


def preprocess_images(images: list[bytes], grayscale: bool = True, max_dim: int = 2000) -> list[bytes]:
    from PIL import Image, ImageEnhance

    processed = []
    for img_bytes in images:
        img = Image.open(io.BytesIO(img_bytes))
        if grayscale:
            img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        processed.append(buf.getvalue())
    return processed


def yolo_crop_columns(images: list[bytes], conf: float = 0.2) -> list[tuple[bytes, int, str]]:
    from doclayout_yolo import YOLOv10
    from PIL import Image

    model = YOLOv10("juliozhao/DocLayout-YOLO-DocStructBench-DS")
    text_classes = {"plain text", "title", "abandon", "figure caption", "table caption", "table", "isolate_formula"}

    crops: list[tuple[bytes, int, str]] = []
    for page_num, img_bytes in enumerate(images):
        pil_img = Image.open(io.BytesIO(img_bytes))
        results = model(pil_img, conf=conf, verbose=False)
        boxes = results[0].boxes

        regions = []
        for box in boxes:
            cls_name = results[0].names[int(box.cls)]
            if cls_name not in text_classes:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            area = (x2 - x1) * (y2 - y1)
            if area < 3000:
                continue
            regions.append((x1, y1, x2, y2, cls_name, float(box.conf)))

        regions.sort(key=lambda r: (r[1], r[0]))

        if not regions:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            crops.append((buf.getvalue(), page_num + 1, "full_page"))
        else:
            for x1, y1, x2, y2, cls_name, _ in regions:
                crop = pil_img.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=95)
                crops.append((buf.getvalue(), page_num + 1, cls_name))

    del model
    return crops


def run_vllm(model_name: str, images: list[bytes], prompt: str, max_tokens: int = 4096, max_model_len: int = 16384) -> list[str]:
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print(f"  loading vLLM: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=max_model_len, limit_mm_per_prompt={"image": 1})
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    vllm_prompts = []
    for img_bytes in images:
        pil_img = Image.open(io.BytesIO(img_bytes))
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
        formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vllm_prompts.append({"prompt": formatted, "multi_modal_data": {"image": pil_img}})

    t0 = time.time()
    outputs = llm.generate(vllm_prompts, sampling_params)
    print(f"  vLLM: {len(outputs)} items in {time.time() - t0:.1f}s", flush=True)

    page_texts = [out.outputs[0].text for out in outputs]
    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return page_texts


def run_vllm_text(model_name: str, text: str, prompt_template: str, max_tokens: int = 8192, max_model_len: int = 32768, **fmt_kwargs: str) -> str:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"  loading vLLM text: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=max_model_len)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    user_content = prompt_template.format(text=text, **fmt_kwargs)
    messages = [{"role": "user", "content": user_content}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    outputs = llm.generate([formatted], sampling_params)
    print(f"  vLLM text: {time.time() - t0:.1f}s", flush=True)

    result = outputs[0].outputs[0].text.strip()
    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return result


def run_hf(model_name: str, images: list[bytes], prompt: str, max_tokens: int = 4096) -> list[str]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

    print(f"  loading HF: {model_name}", flush=True)
    load_kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": torch.bfloat16}

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = None
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    except Exception:
        pass
    if model is None:
        for cls in [AutoModelForVision2Seq, AutoModel, AutoModelForCausalLM]:
            try:
                model = cls.from_pretrained(model_name, **load_kwargs)
                break
            except (ValueError, ImportError, KeyError, OSError):
                continue
    if model is None:
        raise RuntimeError(f"could not load {model_name}")

    page_texts: list[str] = []
    for i, img_bytes in enumerate(images):
        pil_img = Image.open(io.BytesIO(img_bytes))
        print(f"    page {i + 1}/{len(images)}...", end="", flush=True)
        t0 = time.time()

        if hasattr(model, "chat"):
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            transform = T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            pixel_values = transform(pil_img).unsqueeze(0).to(torch.bfloat16).cuda()
            response = model.chat(processor, pixel_values, f"<image>\n{prompt}", generation_config={"max_new_tokens": max_tokens})
            page_texts.append(response)
        elif "got_ocr2" in model_name.lower() or "GOT-OCR" in model_name:
            inputs = processor(pil_img, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            page_texts.append(processor.batch_decode(generated, skip_special_tokens=True)[0])
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(model.device)
            except Exception:
                text = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
                inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            page_texts.append(processor.batch_decode(generated, skip_special_tokens=True)[0])

        print(f" {time.time() - t0:.1f}s", flush=True)

    del model
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return page_texts


def merge_structured(page_texts: list[str]) -> str:
    all_articles = []
    for page_num, page_text in enumerate(page_texts):
        try:
            page_data = json.loads(_strip_markdown(page_text))
            articles = page_data.get("articles", []) if isinstance(page_data, dict) else []
        except json.JSONDecodeError:
            articles = [{"unit_type": "article", "headline": None, "text": page_text}]
        for art in articles:
            if "page_span" not in art:
                art["page_span"] = [page_num + 1]
            all_articles.append(art)
    return json.dumps({"articles": all_articles})


def merge_raw(page_texts: list[str]) -> str:
    combined = "\n\n".join(f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(page_texts))
    return combined


def save_result(config_name: str, issue_date: str, result_json: str, page_count: int, elapsed: float) -> None:
    issue_json = _build_issue_json(result_json, date=issue_date, source="il_messaggero", page_count=page_count)
    issue = issue_from_dict(json.loads(issue_json))

    n_articles = len(issue.articles)
    total_chars = len(extract_full_text(issue))
    print(f"-> {n_articles} articles | {total_chars} chars | {elapsed:.1f}s")

    for i, art in enumerate(issue.articles):
        headline = art.headline or "(no headline)"
        chars = sum(len(p.text) for p in art.paragraphs)
        print(f"   [{i}] {art.unit_type}: {headline} ({chars}c, pages={art.page_span})")

    print(f"preview: {extract_full_text(issue)[:400]}...")

    output_path = PREDICTIONS_DIR / f"{config_name}_{issue_date}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dc.asdict(issue), indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


PIPELINES: dict[str, dict] = {
    "qwen7b_structured": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_STRUCTURED, "mode": "structured", "backend": "vllm", "max_tokens": 4096},
    "qwen3b_structured": {"vlm": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt": prompts.VLM_OCR_STRUCTURED, "mode": "structured", "backend": "vllm", "max_tokens": 4096},
    "qwen7b_raw": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_RAW, "mode": "raw", "backend": "vllm", "max_tokens": 4096},
    "qwen3b_raw": {"vlm": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt": prompts.VLM_OCR_RAW, "mode": "raw", "backend": "vllm", "max_tokens": 4096},
    "internvl4b_structured": {"vlm": "OpenGVLab/InternVL2_5-4B", "prompt": prompts.VLM_OCR_STRUCTURED, "mode": "structured", "backend": "hf", "max_tokens": 4096},
    "got_ocr2_raw": {"vlm": "stepfun-ai/GOT-OCR-2.0-hf", "prompt": prompts.VLM_OCR_RAW, "mode": "raw", "backend": "hf", "max_tokens": 4096},

    "qwen7b_v2_structured": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_STRUCTURED_V2, "mode": "structured", "backend": "vllm", "max_tokens": 8192},
    "qwen7b_v2_raw": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_RAW_V2, "mode": "raw", "backend": "vllm", "max_tokens": 8192},
    "qwen3b_v2_structured": {"vlm": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt": prompts.VLM_OCR_STRUCTURED_V2, "mode": "structured", "backend": "vllm", "max_tokens": 8192},
    "qwen7b_v2_raw_cleanup": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_RAW_V2, "mode": "raw_cleanup", "backend": "vllm", "max_tokens": 8192, "cleanup_model": "Qwen/Qwen2.5-7B-Instruct", "cleanup_prompt": prompts.LLM_CLEANUP_V2},
    "qwen7b_v2_raw_postcorrect": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_RAW_V2, "mode": "raw_postcorrect", "backend": "vllm", "max_tokens": 8192, "correction_model": "Qwen/Qwen2.5-7B-Instruct"},

    "preproc_qwen7b_v2_structured": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_STRUCTURED_V2, "mode": "structured", "backend": "vllm", "max_tokens": 8192, "preprocess": True},
    "preproc_qwen7b_v2_raw": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_RAW_V2, "mode": "raw", "backend": "vllm", "max_tokens": 8192, "preprocess": True},

    "yolo_qwen7b_column": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_COLUMN, "mode": "yolo_raw", "backend": "vllm", "max_tokens": 4096},
    "yolo_qwen7b_structured": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_STRUCTURED_V2, "mode": "yolo_structured", "backend": "vllm", "max_tokens": 4096},
    "yolo_qwen3b_column": {"vlm": "Qwen/Qwen2.5-VL-3B-Instruct", "prompt": prompts.VLM_OCR_COLUMN, "mode": "yolo_raw", "backend": "vllm", "max_tokens": 4096},
    "yolo_qwen7b_column_cleanup": {"vlm": "Qwen/Qwen2.5-VL-7B-Instruct", "prompt": prompts.VLM_OCR_COLUMN, "mode": "yolo_raw_cleanup", "backend": "vllm", "max_tokens": 4096, "cleanup_model": "Qwen/Qwen2.5-7B-Instruct", "cleanup_prompt": prompts.LLM_CLEANUP_V2},

    "qwen72b_awq_structured": {"vlm": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "prompt": prompts.VLM_OCR_STRUCTURED_V2, "mode": "structured", "backend": "vllm", "max_tokens": 8192, "max_model_len": 8192},
}


def run_pipeline(config_name: str, issue_date: str) -> None:
    if config_name not in PIPELINES:
        print(f"unknown: {config_name}")
        print(f"available: {', '.join(sorted(PIPELINES.keys()))}")
        return

    cfg = PIPELINES[config_name]
    issue_dir = GROUND_TRUTH_DIR / issue_date
    if not issue_dir.exists():
        print(f"no images at {issue_dir}")
        return

    output_path = PREDICTIONS_DIR / f"{config_name}_{issue_date}.json"
    if output_path.exists():
        print(f"SKIP {config_name} {issue_date} (exists)")
        return

    images = load_issue_images(issue_dir)
    page_count = len(images)
    print(f"running {config_name} on {issue_date} ({page_count} pages)", flush=True)

    t0 = time.time()

    if cfg.get("preprocess"):
        print("  preprocessing images...", flush=True)
        images = preprocess_images(images)

    mode = cfg["mode"]
    vlm = cfg["vlm"]
    prompt = cfg["prompt"]
    max_tokens = cfg["max_tokens"]
    max_model_len = cfg.get("max_model_len", 16384)
    backend = cfg["backend"]

    if mode.startswith("yolo"):
        print("  running YOLO column detection...", flush=True)
        crops = yolo_crop_columns(images)
        print(f"  {len(crops)} regions detected across {page_count} pages", flush=True)
        crop_images = [c[0] for c in crops]
        crop_pages = [c[1] for c in crops]

        if backend == "vllm":
            crop_texts = run_vllm(vlm, crop_images, prompt, max_tokens=max_tokens, max_model_len=max_model_len)
        else:
            crop_texts = run_hf(vlm, crop_images, prompt, max_tokens=max_tokens)

        if "structured" in mode:
            all_articles = []
            for text, page_num in zip(crop_texts, crop_pages):
                try:
                    data = json.loads(_strip_markdown(text))
                    arts = data.get("articles", []) if isinstance(data, dict) else []
                except json.JSONDecodeError:
                    arts = [{"unit_type": "article", "headline": None, "text": text}]
                for a in arts:
                    a["page_span"] = [page_num]
                    all_articles.append(a)
            result_json = json.dumps({"articles": all_articles})
        else:
            page_groups: dict[int, list[str]] = {}
            for text, page_num in zip(crop_texts, crop_pages):
                page_groups.setdefault(page_num, []).append(text)
            page_texts = ["\n\n".join(page_groups.get(i + 1, [])) for i in range(page_count)]
            raw_text = merge_raw(page_texts)

            if "cleanup" in mode:
                result_json = run_vllm_text(cfg["cleanup_model"], raw_text, cfg["cleanup_prompt"], max_tokens=8192, page_count=str(page_count))
            else:
                result_json = json.dumps({"articles": [{"unit_type": "article", "headline": None, "paragraphs": [{"text": raw_text}]}]})
    else:
        if backend == "vllm":
            page_texts = run_vllm(vlm, images, prompt, max_tokens=max_tokens, max_model_len=max_model_len)
        else:
            page_texts = run_hf(vlm, images, prompt, max_tokens=max_tokens)

        if mode == "structured":
            result_json = merge_structured(page_texts)
        elif mode == "raw_cleanup":
            raw_text = merge_raw(page_texts)
            result_json = run_vllm_text(cfg["cleanup_model"], raw_text, cfg["cleanup_prompt"], max_tokens=8192, page_count=str(page_count))
        elif mode == "raw_postcorrect":
            raw_text = merge_raw(page_texts)
            corrected = run_vllm_text(cfg["correction_model"], raw_text, prompts.LLM_POST_CORRECTION, max_tokens=8192)
            result_json = json.dumps({"articles": [{"unit_type": "article", "headline": None, "paragraphs": [{"text": corrected}]}]})
        else:
            raw_text = merge_raw(page_texts)
            result_json = json.dumps({"articles": [{"unit_type": "article", "headline": None, "paragraphs": [{"text": raw_text}]}]})

    elapsed = time.time() - t0
    try:
        save_result(config_name, issue_date, result_json, page_count, elapsed)
    except Exception as e:
        print(f"FAILED saving: {e}")
        import traceback; traceback.print_exc()


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: run_single_hf.py <config|all> [date|all]")
        print(f"configs: {', '.join(sorted(PIPELINES.keys()))}")
        return

    config_name = sys.argv[1]
    issue_date = sys.argv[2] if len(sys.argv) > 2 else "all"
    dates = ISSUE_DATES if issue_date == "all" else [issue_date]

    if config_name == "all":
        for name in PIPELINES:
            for d in dates:
                try:
                    run_pipeline(name, d)
                except Exception as e:
                    print(f"FAILED {name} {d}: {e}")
    else:
        for d in dates:
            try:
                run_pipeline(config_name, d)
            except Exception as e:
                print(f"FAILED {config_name} {d}: {e}")


if __name__ == "__main__":
    main()
