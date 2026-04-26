from __future__ import annotations

import json
import re
import sys
import typing as tp


REPEAT_PATTERN = re.compile(r"(\s*[\.\-\=\_\*])\s*((?:[\.\-\=\_\*]\s*){6,})")


def trim_trailing_garbage(text: str) -> str:
    if not text:
        return text
    trimmed = text
    trimmed = re.sub(r"(\s*\.\s*){5,}\s*$", ".", trimmed)
    trimmed = re.sub(r"(\s*\-\s*){5,}\s*$", "-", trimmed)
    trimmed = re.sub(r"(\s*\*\s*){5,}\s*$", "", trimmed)
    trimmed = re.sub(r"(\s*\"\s*){3,}\s*$", "", trimmed)
    words = trimmed.rstrip().split()
    if words and all(len(w) <= 2 for w in words[-30:]):
        while words and len(words[-1]) <= 2 and words[-1] in {".", "-", "_", "=", "*", ",", ":", ";"}:
            words.pop()
        trimmed = " ".join(words)
    return trimmed


def looks_like_json_blob(text: str) -> bool:
    if not text:
        return False
    first200 = text[:200].strip()
    if first200.startswith("```"):
        first200 = first200.lstrip("`").lstrip("json").lstrip("JSON").strip()
    if first200.startswith("{") and '"articles"' in first200:
        return True
    if first200.startswith("{") and ('"unit_type"' in first200 or '"paragraphs"' in first200):
        return True
    if first200.startswith("[") and '{' in first200 and '"text"' in first200:
        return True
    if '"unit_type"' in text[:500] and '"paragraphs"' in text[:500]:
        return True
    return False


def trim_predictions(pred: dict[str, tp.Any]) -> dict[str, tp.Any]:
    articles = list(pred.get("articles", []))
    trimmed_count = 0
    dropped_count = 0
    new_articles = []
    for art in articles:
        paragraphs = list(art.get("paragraphs", []))
        if not paragraphs:
            new_articles.append(art)
            continue
        combined = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in paragraphs)
        if looks_like_json_blob(combined):
            dropped_count += 1
            continue
        new_paragraphs = []
        changed = False
        for p in paragraphs:
            if not isinstance(p, dict):
                new_paragraphs.append({"text": str(p)})
                continue
            t = p.get("text", "")
            t_new = trim_trailing_garbage(t)
            if t_new != t:
                changed = True
            new_paragraphs.append({**p, "text": t_new})
        if changed:
            trimmed_count += 1
        new_art = dict(art)
        new_art["paragraphs"] = new_paragraphs
        new_articles.append(new_art)
    out = dict(pred)
    out["articles"] = new_articles
    print(f"  Trimmed {trimmed_count} articles, dropped {dropped_count} JSON blobs ({len(articles)} -> {len(new_articles)})")
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: trim_repetitive.py <input> <output>")
        sys.exit(1)
    pred = json.loads(open(sys.argv[1]).read())
    out = trim_predictions(pred)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
