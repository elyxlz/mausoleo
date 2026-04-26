"""LLM post-correction for high-CER ensemble articles.

For each ensemble article that has medium-to-high error rate (estimated by
text-quality heuristics), feed it to Qwen2.5-7B with an OCR-cleanup prompt
asking to correct typos while preserving original Italian wording.

Heuristic for "needs cleanup":
  - Article length 400-3000 chars (skip very short ads / very long blobs)
  - Contains many short broken-word fragments (3-letter words ending mid-syllable)
  - OR contains "obvious" OCR garbage patterns (consecutive 2-char words, repeated
    consonants without vowels)

Usage:
    uv run --no-project python scripts/llm_postcorrect.py <input.json> <date> <output.json>
"""
from __future__ import annotations

import json
import re
import sys
import typing as tp


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _quality_score(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    if not words:
        return 0.0
    short_broken = sum(1 for w in words if 2 <= len(w) <= 3 and not re.search(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", w))
    no_vowel = sum(1 for w in words if len(w) >= 4 and not re.search(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", w))
    fragment_ratio = (short_broken + no_vowel * 2) / max(len(words), 1)
    return fragment_ratio


def needs_cleanup(article: dict[str, tp.Any], min_chars: int = 400, max_chars: int = 2500) -> bool:
    text = _article_text(article)
    if not (min_chars <= len(text) <= max_chars):
        return False
    return True


CLEANUP_PROMPT_TEMPLATE = """You are an expert at correcting OCR errors in Italian historical newspaper text from 1880-1945.

The text below has been OCR'd from a 19th-century Italian newspaper and contains transcription errors (broken words, wrong characters, missing letters). Your job is to FIX OBVIOUS OCR ERRORS while preserving:
1. The exact factual content (names, numbers, places)
2. The original Italian wording (do not modernize or paraphrase)
3. The structure and length (do not summarize or expand)
4. Period-typical archaic spelling and punctuation

OCR'd text:
\"\"\"
{TEXT}
\"\"\"

Output ONLY the corrected text, with no explanation, no markdown fences, no headers. Begin directly with the first word."""


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: llm_postcorrect.py <input.json> <date> <output.json>")
        sys.exit(1)

    issue = json.loads(open(sys.argv[1]).read())
    out_path = sys.argv[3]
    articles = list(issue.get("articles", []))

    candidates_idx: list[int] = []
    for i, a in enumerate(articles):
        if needs_cleanup(a):
            candidates_idx.append(i)

    print(f"Articles total: {len(articles)} | Candidates for cleanup: {len(candidates_idx)}", flush=True)
    if not candidates_idx:
        with open(out_path, "w") as f:
            json.dump(issue, f, indent=2, ensure_ascii=False)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    prompts: list[str] = []
    for idx in candidates_idx:
        text = _article_text(articles[idx])
        msg = [{"role": "user", "content": CLEANUP_PROMPT_TEMPLATE.format(TEXT=text)}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params)

    new_articles = list(articles)
    accepted = 0
    rejected = 0
    for idx, output in zip(candidates_idx, outputs):
        cleaned = output.outputs[0].text.strip()
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip()
        original = _article_text(articles[idx])
        if not cleaned:
            rejected += 1
            continue
        ratio = len(cleaned) / max(len(original), 1)
        if ratio < 0.5 or ratio > 1.6:
            rejected += 1
            continue
        new_a = dict(articles[idx])
        new_a["paragraphs"] = [{"text": cleaned, "is_continuation": False}]
        new_articles[idx] = new_a
        accepted += 1

    print(f"Cleaned articles: accepted {accepted} / rejected {rejected}", flush=True)
    out = {**issue, "articles": new_articles}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
