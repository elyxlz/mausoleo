"""LLM-based cross-page article stitcher.

For each pair (article A on page N, article B on page N+1) that meet criteria:
  - A is "long enough" (>400 chars)
  - A's last paragraph doesn't end in . ! ? » " ) ]
  - B starts with lowercase OR has no headline OR is unusually short (<800 chars)

Asks Qwen2.5-7B to judge "Is article B the continuation of article A?"
If yes, merges paragraphs and updates page_span.

Usage:
    uv run --no-project python scripts/llm_stitch_crosspage.py <input.json> <output.json>
"""
from __future__ import annotations

import json
import re
import sys
import typing as tp


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _ends_unfinished(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] not in ".!?»\"')]}…"


def _is_continuation_candidate(b: dict[str, tp.Any]) -> bool:
    text = _article_text(b).lstrip()
    if not text:
        return False
    if not text[0].isalpha():
        return False
    if text[0].islower():
        return True
    if not b.get("headline"):
        return True
    return False


def find_candidate_pairs(articles: list[dict[str, tp.Any]]) -> list[tuple[int, int]]:
    sorted_idx = sorted(
        range(len(articles)),
        key=lambda i: (
            articles[i].get("page_span", [999])[0] if articles[i].get("page_span") else 999,
            i,
        ),
    )
    sorted_articles = [articles[i] for i in sorted_idx]
    sorted_to_orig = {si: oi for si, oi in enumerate(sorted_idx)}

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, a in enumerate(sorted_articles):
        if i in used:
            continue
        pa = a.get("page_span") or []
        if not pa:
            continue
        text_a = _article_text(a)
        if len(text_a) < 400:
            continue
        if not _ends_unfinished(text_a):
            continue
        last_page_a = pa[-1]

        for j in range(i + 1, len(sorted_articles)):
            if j in used:
                continue
            b = sorted_articles[j]
            pb = b.get("page_span") or []
            if not pb:
                continue
            if pb[0] > last_page_a + 1:
                break
            if pb[0] != last_page_a + 1:
                continue
            if not _is_continuation_candidate(b):
                continue
            pairs.append((sorted_to_orig[i], sorted_to_orig[j]))
            used.add(j)
            break
    return pairs


def build_prompt(article_a: dict[str, tp.Any], article_b: dict[str, tp.Any]) -> str:
    a_text = _article_text(article_a)
    b_text = _article_text(article_b)
    a_tail = a_text[-400:]
    b_head = b_text[:400]
    a_headline = article_a.get("headline") or "(no headline)"
    b_headline = article_b.get("headline") or "(no headline)"
    return (
        "You are analyzing OCR output from a 19th-20th century Italian newspaper. "
        "Determine whether one article's text continues into another article on the next page.\n\n"
        f"ARTICLE A (page {article_a.get('page_span')}, headline: {a_headline}):\n"
        f"... {a_tail}\n\n"
        f"ARTICLE B (page {article_b.get('page_span')}, headline: {b_headline}):\n"
        f"{b_head} ...\n\n"
        "Is ARTICLE B the continuation of ARTICLE A? "
        "Continuations show: (a) ARTICLE A's last sentence is incomplete and B starts mid-sentence, "
        "(b) same topic/characters carry over, (c) ARTICLE B has no headline OR a continuation marker.\n"
        "Reply with exactly one word: YES or NO"
    )


def stitch_with_llm(
    issue: dict[str, tp.Any],
    model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> tuple[dict[str, tp.Any], int, int]:
    articles = issue.get("articles", [])
    pairs = find_candidate_pairs(articles)
    if not pairs:
        return issue, 0, 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    prompts: list[str] = []
    for ai, bi in pairs:
        msg = [{"role": "user", "content": build_prompt(articles[ai], articles[bi])}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params)

    matched: list[tuple[int, int]] = []
    for (ai, bi), output in zip(pairs, outputs):
        text = output.outputs[0].text.strip().upper()
        if text.startswith("YES"):
            matched.append((ai, bi))

    if not matched:
        return issue, len(pairs), 0

    consumed: set[int] = set()
    merge_map: dict[int, int] = {}
    for ai, bi in matched:
        if ai in consumed or bi in consumed:
            continue
        merge_map[ai] = bi
        consumed.add(bi)

    new_articles: list[dict[str, tp.Any]] = []
    for i, a in enumerate(articles):
        if i in consumed:
            continue
        if i in merge_map:
            j = merge_map[i]
            successor = articles[j]
            new_a = dict(a)
            new_paras = list(a.get("paragraphs", [])) + list(successor.get("paragraphs", []))
            new_a["paragraphs"] = new_paras
            pa = a.get("page_span", [])
            pb = successor.get("page_span", [])
            new_a["page_span"] = sorted(set(pa + pb))
            new_articles.append(new_a)
        else:
            new_articles.append(a)

    out = dict(issue)
    out["articles"] = new_articles
    return out, len(pairs), len(matched)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: llm_stitch_crosspage.py <input.json> <output.json> [model]")
        sys.exit(1)

    issue = json.loads(open(sys.argv[1]).read())
    model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-7B-Instruct"
    before = len(issue.get("articles", []))
    out, candidates, stitched = stitch_with_llm(issue, model=model)
    after = len(out.get("articles", []))
    print(f"Before: {before} | After: {after} | Candidates: {candidates} | Stitched: {stitched}")
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
