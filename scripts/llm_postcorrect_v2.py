"""LLM post-correction v2: multi-source consensus merge.

For each ensemble article, find 2-3 OTHER source predictions that match it (by
text-overlap). Feed all 3-4 versions to Qwen2.5-7B and ask it to produce the
MOST CORRECT consensus version. Reject LLM output if char-edit-distance from
the original ensemble article exceeds a threshold (prevents over-paraphrasing).

Why consensus: if 3 different OCR sources all agree on "Giordano" but the
ensemble has "Taroni", the LLM has clear evidence which is right.

Usage:
    uv run --no-project python scripts/llm_postcorrect_v2.py <ens.json> <date> <out.json>
"""
from __future__ import annotations

import difflib
import json
import os
import pathlib as pl
import re
import sys
import typing as tp


SOURCE_NAMES = [
    "exp_045_qwen3vl_vllm",
    "col3_qwen3_8b_v2_structured",
    "col4_qwen3_8b_v2_structured",
    "exp_055_col6_ads_prompt",
    "exp_010_yolo_qwen3_8b",
    "exp_108_col3_qwen25vl",
    "exp_098_col5_qwen3vl_vllm",
    "exp_099_col2_qwen3vl_vllm",
    "exp_111_col2_qwen25vl",
    "yolo_qwen25_7b_v2_structured",
    "exp_052_col6_vllm",
    "exp_028_yolo_smallregion",
    "qwen3b_structured",
    "qwen_vl_3b_structured",
]


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _text_overlap(a: str, b: str) -> float:
    a_words = set(_normalize(a).split())
    b_words = set(_normalize(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _edit_ratio(a: str, b: str) -> float:
    if not a:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1.0 - sm.ratio()


def find_alternates(target: dict[str, tp.Any], sources: list[dict[str, tp.Any]], k: int = 3, min_overlap: float = 0.45) -> list[str]:
    target_text = _article_text(target)
    if len(target_text) < 200:
        return []
    target_norm = _normalize(target_text)
    matches: list[tuple[float, str]] = []
    for src in sources:
        for art in src.get("articles", []):
            text = _article_text(art)
            if abs(len(text) - len(target_text)) / max(len(target_text), 1) > 0.5:
                continue
            ov = _text_overlap(target_norm, _normalize(text))
            if ov >= min_overlap:
                matches.append((ov, text))
    matches.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    result: list[str] = []
    for ov, t in matches:
        key = t[:100]
        if key in seen:
            continue
        seen.add(key)
        result.append(t)
        if len(result) >= k:
            break
    return result


CONSENSUS_PROMPT = """You are an expert at reconciling multiple OCR transcriptions of the same Italian historical newspaper article (1880-1945) into a single best version.

Below are {N} different OCR'd versions of the SAME article (each with different transcription errors). Your task: produce the BEST single transcription by:
- Picking the most common reading at each character/word
- Preserving the original archaic Italian wording
- Keeping the same approximate length
- NOT modernizing, NOT summarizing, NOT paraphrasing

DO NOT add interpretation. DO NOT explain. Output ONLY the corrected text directly.

{VERSIONS}

Output the consensus version below (no preamble):"""


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: llm_postcorrect_v2.py <ens.json> <date> <out.json>")
        sys.exit(1)

    ens = json.loads(open(sys.argv[1]).read())
    date = sys.argv[2]
    out_path = sys.argv[3]

    cache = pl.Path("eval/predictions")
    sources: list[dict[str, tp.Any]] = []
    for name in SOURCE_NAMES:
        path = cache / f"{name}_{date}.json"
        if path.exists():
            sources.append(json.loads(path.read_text()))

    articles = list(ens.get("articles", []))
    candidates: list[tuple[int, str, list[str]]] = []
    for i, art in enumerate(articles):
        text = _article_text(art)
        if not (400 <= len(text) <= 2500):
            continue
        alts = find_alternates(art, sources, k=3)
        if len(alts) >= 2:
            candidates.append((i, text, alts))

    print(f"Articles total: {len(articles)} | Consensus candidates: {len(candidates)}", flush=True)
    if not candidates:
        with open(out_path, "w") as f:
            json.dump(ens, f, indent=2, ensure_ascii=False)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=12288,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    prompts: list[str] = []
    for idx, original, alts in candidates:
        all_versions = [original] + alts
        version_block = "\n\n".join(f"VERSION {k+1}:\n\"\"\"\n{v}\n\"\"\"" for k, v in enumerate(all_versions))
        prompt_text = CONSENSUS_PROMPT.format(N=len(all_versions), VERSIONS=version_block)
        msg = [{"role": "user", "content": prompt_text}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params)

    new_articles = list(articles)
    accepted = 0
    rejected_drift = 0
    rejected_length = 0
    rejected_empty = 0
    for (idx, original, alts), output in zip(candidates, outputs):
        cleaned = output.outputs[0].text.strip()
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip()
        if not cleaned:
            rejected_empty += 1
            continue
        ratio = len(cleaned) / max(len(original), 1)
        if ratio < 0.7 or ratio > 1.4:
            rejected_length += 1
            continue
        edit = _edit_ratio(original, cleaned)
        if edit > 0.30:
            rejected_drift += 1
            continue
        new_a = dict(articles[idx])
        new_a["paragraphs"] = [{"text": cleaned, "is_continuation": False}]
        new_articles[idx] = new_a
        accepted += 1

    print(f"Accepted: {accepted} | Rejected drift: {rejected_drift} | length: {rejected_length} | empty: {rejected_empty}", flush=True)
    out = {**ens, "articles": new_articles}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
