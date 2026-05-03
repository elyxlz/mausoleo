"""§5.3 quality-assessment spot-check artefacts.

Runs three sub-tasks needed by `references/section_5_spotcheck.md`:
  1. Pull 10 random day-summaries from July 1943 (seed=1943, exclude 07-26),
     extract top-5 named entities per day via Claude over OAuth, then check
     entity presence in the day's transcription. Verdict per day.
  2. For 1 random day (seed=1944), trace entities at article (top-3 most-
     summarised), day, week, month levels. Quantify survival.
  3. Pull side-by-side 25-July representations at day / week / month.

Output: writes `eval/section_5_spotcheck.json` with the raw evidence the
markdown drafter cites.

Auth: OAuth via ~/.claude/.credentials.json + oauth-2025-04-20 header.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import random
import re
import sys
import time
import typing as tp
import unicodedata

import anthropic


REPO = pathlib.Path("/tmp/mausoleo")
DAY = REPO / "eval/summaries/day"
WEEK = REPO / "eval/summaries/week"
MONTH = REPO / "eval/summaries/month/1943-07.json"
ARTICLE = REPO / "eval/summaries/article"
TRANS = REPO / "eval/transcriptions"
OUT = REPO / "eval/section_5_spotcheck.json"

OAUTH_BETA = "oauth-2025-04-20"
MODEL = "claude-sonnet-4-5-20250929"
CC_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

NER_PROMPT = (
    "You are a named-entity extractor for a 1943 Italian newspaper "
    "summary (Il Messaggero, Rome). From the summary text below, "
    "extract the FIVE most prominent named entities — people, places, "
    "or organisations. Return as a JSON array of exactly 5 objects "
    "with keys 'entity' (string, the surface form as written) and "
    "'type' (one of PERSON, PLACE, ORG). Order by salience (most "
    "salient first). Use the exact spelling from the text. No prose "
    "outside the JSON array.\n\n"
    "SUMMARY:\n{summary}\n\n"
    "JSON ARRAY:"
)

NER_FULL_PROMPT = (
    "You are a named-entity extractor for a 1943 Italian newspaper "
    "summary (Il Messaggero, Rome). From the text below, extract ALL "
    "distinct named entities — people, places, organisations. Return "
    "as a JSON array of objects with keys 'entity' and 'type' (one of "
    "PERSON, PLACE, ORG). Use the exact spelling from the text. "
    "Deduplicate. No prose outside the JSON.\n\n"
    "TEXT:\n{summary}\n\n"
    "JSON ARRAY:"
)

TOPICS_PROMPT = (
    "You are a topic extractor for a 1943 Italian newspaper summary "
    "(Il Messaggero, Rome). From the text below, list 3-6 distinct "
    "topics (short noun phrases in Italian, e.g. 'fronte siciliano', "
    "'bombardamento di Bologna'). Return as a JSON array of strings. "
    "No prose outside the JSON.\n\n"
    "TEXT:\n{text}\n\n"
    "JSON ARRAY:"
)


def _load_token() -> str:
    p = os.path.expanduser("~/.claude/.credentials.json")
    return json.load(open(p))["claudeAiOauth"]["accessToken"]


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        auth_token=_load_token(),
        default_headers={"anthropic-beta": OAUTH_BETA},
    )


def _call(
    client: anthropic.Anthropic, user: str, *, max_tokens: int = 800,
    attempt: int = 0,
) -> str:
    try:
        r = client.messages.create(
            model=MODEL,
            system=CC_SYSTEM,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIStatusError as e:
        if attempt < 2:
            time.sleep(2 + attempt * 3)
            return _call(client, user, max_tokens=max_tokens, attempt=attempt + 1)
        raise
    return "".join(b.text for b in r.content if hasattr(b, "text"))


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_json_array(text: str) -> list:
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def _normalise(s: str) -> str:
    """Lowercase + strip diacritics for fuzzy substring matching."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _entity_present(entity: str, haystack_norm: str) -> tuple[bool, str]:
    """Substring match after normalisation. For multi-word entities we
    also allow each token to match independently (covers OCR-fragmented
    surface forms like 'San Francesco' vs 'San Francesc')."""
    norm = _normalise(entity)
    if not norm:
        return False, "empty"
    if norm in haystack_norm:
        return True, "exact-norm"
    # Fall back: shortest meaningful substring (drop short tokens) — try
    # the longest token as a weaker hit.
    toks = [t for t in norm.split(" ") if len(t) >= 4]
    if toks:
        # Require at least one token >=5 chars to count as partial.
        long_toks = [t for t in toks if len(t) >= 5]
        for t in long_toks:
            if t in haystack_norm:
                return True, f"partial:{t}"
    return False, "absent"


def _day_haystack(date: str) -> str:
    """Concatenate every paragraph + headline from the transcription
    file for that date, normalised."""
    p = TRANS / f"{date}.json"
    d = json.load(open(p))
    parts: list[str] = []
    for art in d["articles"]:
        if art.get("headline"):
            parts.append(art["headline"])
        for para in (art.get("paragraphs") or []):
            t = para.get("text") or ""
            if t:
                parts.append(t)
    return _normalise(" ".join(parts))


def _verdict(hits: int, total: int) -> str:
    if total == 0:
        return "FAIL"
    frac = hits / total
    if frac >= 0.8:
        return "PASS"
    if frac >= 0.4:
        return "PARTIAL"
    return "FAIL"


def task_1_spotcheck(client: anthropic.Anthropic) -> dict:
    rng = random.Random(1943)
    candidates = [d for d in range(1, 32) if d != 26]
    picks = sorted(rng.sample(candidates, 10))
    results = []
    for day in picks:
        date = f"1943-07-{day:02d}"
        print(f"  spot-check day {date}", flush=True)
        day_doc = json.load(open(DAY / f"{date}.json"))
        summary = day_doc["summary"]
        raw = _call(client, NER_PROMPT.format(summary=summary), max_tokens=500)
        ents = _parse_json_array(raw)
        # Keep only the first 5 well-formed entries.
        clean = []
        for e in ents:
            if isinstance(e, dict) and e.get("entity") and e.get("type"):
                clean.append({
                    "entity": str(e["entity"]),
                    "type": str(e["type"]).upper(),
                })
            if len(clean) == 5:
                break
        haystack = _day_haystack(date)
        checks = []
        hits = 0
        for e in clean:
            present, how = _entity_present(e["entity"], haystack)
            checks.append({**e, "present_in_source": present, "match": how})
            if present:
                hits += 1
        results.append({
            "date": date,
            "child_count": day_doc.get("child_count"),
            "summary": summary,
            "top5_entities": checks,
            "hits": hits,
            "out_of": len(checks),
            "verdict": _verdict(hits, len(checks)),
        })
    agg = {
        "PASS": sum(1 for r in results if r["verdict"] == "PASS"),
        "PARTIAL": sum(1 for r in results if r["verdict"] == "PARTIAL"),
        "FAIL": sum(1 for r in results if r["verdict"] == "FAIL"),
    }
    return {"days": results, "aggregate": agg, "seed": 1943}


def _article_text(date: str, art_id: str) -> str:
    p = TRANS / f"{date}.json"
    d = json.load(open(p))
    for art in d["articles"]:
        if art["id"] == art_id:
            head = art.get("headline") or ""
            paras = " ".join(p.get("text") or "" for p in (art.get("paragraphs") or []))
            return (head + "\n" + paras).strip()
    return ""


def _entity_set(items: list[dict]) -> set:
    return {_normalise(e["entity"]) for e in items if e.get("entity")}


def task_2_infoloss(client: anthropic.Anthropic) -> dict:
    rng = random.Random(1944)
    spot = sorted(random.Random(1943).sample(
        [d for d in range(1, 32) if d != 26], 10
    ))
    candidates = [d for d in range(1, 32) if d != 26 and d not in spot]
    day_n = rng.choice(candidates)
    date = f"1943-07-{day_n:02d}"
    print(f"  info-loss day {date}", flush=True)
    # Top 3 articles by article-summary length (proxy for "most-summarised").
    art_files = sorted(ARTICLE.glob(f"{date}_a*.json"))
    sized = []
    for f in art_files:
        d = json.load(open(f))
        sized.append((len(d.get("summary") or ""), f, d))
    sized.sort(reverse=True)
    top3 = sized[:3]

    article_payload = []
    article_ent_union: list[dict] = []
    for sz, f, doc in top3:
        art_id = doc["node_id"]
        text = _article_text(date, art_id)
        # NER on the article TRANSCRIPTION (full text), not the summary,
        # so we get a faithful "what entities exist at article level".
        full = text if text else (doc.get("summary") or "")
        raw = _call(client, NER_FULL_PROMPT.format(summary=full[:6000]),
                    max_tokens=900)
        ents = _parse_json_array(raw)
        clean = []
        for e in ents:
            if isinstance(e, dict) and e.get("entity"):
                clean.append({
                    "entity": str(e["entity"]),
                    "type": str(e.get("type", "")).upper(),
                })
        topics_raw = _call(client, TOPICS_PROMPT.format(text=full[:6000]),
                           max_tokens=300)
        topics = _parse_json_array(topics_raw)
        article_payload.append({
            "article_id": art_id,
            "headline": doc.get("summary", "").splitlines()[0][:200],
            "char_len": sz,
            "entities": clean,
            "topics": topics,
        })
        article_ent_union.extend(clean)

    # Day, week, month NER + topics.
    day_doc = json.load(open(DAY / f"{date}.json"))
    day_raw = _call(client, NER_FULL_PROMPT.format(summary=day_doc["summary"]),
                    max_tokens=900)
    day_ents = _parse_json_array(day_raw)
    day_topics = _parse_json_array(_call(
        client, TOPICS_PROMPT.format(text=day_doc["summary"]), max_tokens=300))

    # Find covering week.
    week_node_id = day_doc["parent_id"]
    week_doc = json.load(open(WEEK / f"{week_node_id}.json"))
    week_raw = _call(client, NER_FULL_PROMPT.format(summary=week_doc["summary"]),
                     max_tokens=900)
    week_ents = _parse_json_array(week_raw)
    week_topics = _parse_json_array(_call(
        client, TOPICS_PROMPT.format(text=week_doc["summary"]), max_tokens=300))

    month_doc = json.load(open(MONTH))
    month_raw = _call(client, NER_FULL_PROMPT.format(summary=month_doc["summary"]),
                      max_tokens=900)
    month_ents = _parse_json_array(month_raw)
    month_topics = _parse_json_array(_call(
        client, TOPICS_PROMPT.format(text=month_doc["summary"]), max_tokens=300))

    art_set = {_normalise(e["entity"]) for e in article_ent_union if e.get("entity")}
    day_set = {_normalise(e.get("entity", "")) for e in day_ents}
    week_set = {_normalise(e.get("entity", "")) for e in week_ents}
    month_set = {_normalise(e.get("entity", "")) for e in month_ents}

    # "Survives to X" = article-level entity that also appears at level X.
    surv_day = art_set & day_set
    surv_week = art_set & week_set
    surv_month = art_set & month_set

    # Type breakdown for which types compress out.
    by_type: dict[str, dict] = {}
    art_typed = {(_normalise(e["entity"]), e.get("type", "")) for e in article_ent_union if e.get("entity")}
    for ent_norm, typ in art_typed:
        rec = by_type.setdefault(typ or "UNK", {"article": 0, "day": 0, "week": 0, "month": 0})
        rec["article"] += 1
        if ent_norm in day_set:
            rec["day"] += 1
        if ent_norm in week_set:
            rec["week"] += 1
        if ent_norm in month_set:
            rec["month"] += 1

    return {
        "date": date,
        "seed": 1944,
        "article_top3": article_payload,
        "day": {
            "node_id": day_doc["node_id"],
            "entities": day_ents,
            "topics": day_topics,
        },
        "week": {
            "node_id": week_doc["node_id"],
            "entities": week_ents,
            "topics": week_topics,
        },
        "month": {
            "node_id": month_doc["node_id"],
            "entities": month_ents,
            "topics": month_topics,
        },
        "survival_counts": {
            "article": len(art_set),
            "day": len(surv_day),
            "week": len(surv_week),
            "month": len(surv_month),
        },
        "survival_by_type": by_type,
        "surviving_entities": {
            "day": sorted(surv_day),
            "week": sorted(surv_week),
            "month": sorted(surv_month),
        },
    }


def task_3_july25() -> dict:
    day = json.load(open(DAY / "1943-07-25.json"))
    week = json.load(open(WEEK / f"{day['parent_id']}.json"))
    month = json.load(open(MONTH))
    return {
        "day": {
            "node_id": day["node_id"],
            "child_count": day.get("child_count"),
            "summary": day["summary"],
        },
        "week": {
            "node_id": week["node_id"],
            "date_start": week["date_start"],
            "date_end": week["date_end"],
            "child_count": week.get("child_count"),
            "summary": week["summary"],
        },
        "month": {
            "node_id": month["node_id"],
            "child_count": month.get("child_count"),
            "summary": month["summary"],
        },
    }


def main() -> None:
    client = _client()
    print("[1/3] spot-check 10 days...", flush=True)
    t1 = task_1_spotcheck(client)
    print("[2/3] info-loss trace...", flush=True)
    t2 = task_2_infoloss(client)
    print("[3/3] 25-July side-by-side (no LLM)...", flush=True)
    t3 = task_3_july25()
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": MODEL,
        "task_1_spotcheck": t1,
        "task_2_infoloss": t2,
        "task_3_july25": t3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
