"""Build the case-3 oracle ratio table.

Classifies every July-1943 article in `eval/transcriptions/` as:
  - war      (Sicilian campaign, Axis military bulletins, Allied operations,
              war-economy mobilisation)
  - domestic (regime politics, Badoglio transition, civilian-front editorial,
              social policy)
  - other    (sport, culture, non-war economy, classifieds, etc.)

Uses Claude Sonnet 4.5 over OAuth (Claude Max subscription).
Articles batched 10 per call. Output written to
`eval/case_studies/case3_oracle_ratios.json`.

The output records:
  per-article classification, per-day counts, and per-week war:domestic
  ratios for the five ISO weeks that overlap July 1943.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic


REPO = pathlib.Path("/tmp/mausoleo")
TRANS = REPO / "eval/transcriptions"
OUT = REPO / "eval/case_studies/case3_oracle_ratios.json"

OAUTH_BETA = "oauth-2025-04-20"
MODEL = "claude-sonnet-4-5-20250929"
CC_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

CLASSIFIER_PROMPT = (
    "You are classifying short article snippets from Il Messaggero (Rome), "
    "July 1943. For each item, return exactly one label:\n"
    "  - WAR      : Sicilian campaign news, Axis military bulletins, Allied "
    "operations, raids/bombings, war-economy or war-mobilisation pieces.\n"
    "  - DOMESTIC : regime politics, Mussolini, Fascist Party, Badoglio "
    "transition, civilian-front editorial, social policy, domestic "
    "administration.\n"
    "  - OTHER    : sport, culture/arts, theatre, non-war economy, finance "
    "tables, weather, classifieds, obituaries, science.\n\n"
    "Output: one line per item, format `<index>:<LABEL>`. No prose. "
    "Example output for 3 items:\n"
    "1:WAR\n2:DOMESTIC\n3:OTHER\n"
)

BATCH_SIZE = 10
MAX_WORKERS = 6


def _load_token() -> str:
    p = os.path.expanduser("~/.claude/.credentials.json")
    return json.load(open(p))["claudeAiOauth"]["accessToken"]


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        auth_token=_load_token(),
        default_headers={"anthropic-beta": OAUTH_BETA},
    )


def _truncate(s: str, n: int = 600) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_batch(batch: list[dict[str, tp.Any]]) -> str:
    lines = []
    for i, art in enumerate(batch, start=1):
        head = (art.get("headline") or "").strip() or "(no headline)"
        # Concatenate first paragraph to give the LLM enough to classify on.
        text = ""
        for p in art.get("paragraphs", []) or []:
            text += " " + (p.get("text") or "")
            if len(text) > 600:
                break
        lines.append(f"[{i}] HEADLINE: {head}\nTEXT: {_truncate(text, 500)}")
    return "\n\n".join(lines)


_LINE_RE = re.compile(r"^(\d+)\s*[:\-]\s*(WAR|DOMESTIC|OTHER)", re.IGNORECASE)


def _parse_response(text: str, n: int) -> list[str]:
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            lab = m.group(2).upper()
            if 1 <= idx <= n:
                out[idx] = lab
    # Default unknown to OTHER (conservative) so we always emit n labels.
    return [out.get(i + 1, "OTHER") for i in range(n)]


def _classify_batch(
    client: anthropic.Anthropic, batch: list[dict[str, tp.Any]],
    *, attempt: int = 0,
) -> tuple[list[str], int, int]:
    user = (
        CLASSIFIER_PROMPT
        + "\n\nClassify these "
        + str(len(batch))
        + " items:\n\n"
        + _format_batch(batch)
    )
    try:
        r = client.messages.create(
            model=MODEL,
            system=CC_SYSTEM,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIStatusError as e:
        if attempt < 2:
            time.sleep(1.5 + attempt * 2)
            return _classify_batch(client, batch, attempt=attempt + 1)
        return ["OTHER"] * len(batch), 0, 0
    text = "".join(getattr(b, "text", "") for b in r.content)
    labs = _parse_response(text, len(batch))
    return labs, int(r.usage.input_tokens or 0), int(r.usage.output_tokens or 0)


def main() -> None:
    files = sorted(TRANS.glob("1943-07-*.json"))
    days: list[tuple[str, list[dict[str, tp.Any]]]] = []
    total_articles = 0
    for f in files:
        d = json.loads(f.read_text())
        days.append((d["date"], d["articles"]))
        total_articles += len(d["articles"])
    print(f"loaded {len(days)} days, {total_articles} articles", flush=True)

    # Build flat list with stable indices so we can reassemble.
    flat: list[tuple[str, str, dict[str, tp.Any]]] = []  # (date, art_id, art)
    for date, arts in days:
        for art in arts:
            flat.append((date, art["id"], art))

    # Build batches.
    batches: list[list[tuple[str, str, dict[str, tp.Any]]]] = []
    for i in range(0, len(flat), BATCH_SIZE):
        batches.append(flat[i : i + BATCH_SIZE])
    print(f"{len(batches)} batches of <= {BATCH_SIZE}", flush=True)

    client = _client()
    labels: dict[str, str] = {}
    in_tok = 0
    out_tok = 0
    t0 = time.time()
    done = 0

    def _job(idx: int, batch: list[tuple[str, str, dict[str, tp.Any]]]) -> tuple[int, list[str], int, int]:
        labs, it, ot = _classify_batch(client, [b[2] for b in batch])
        return idx, labs, it, ot

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_job, i, b) for i, b in enumerate(batches)]
        for fut in as_completed(futures):
            idx, labs, it, ot = fut.result()
            in_tok += it
            out_tok += ot
            for (date, aid, _), lab in zip(batches[idx], labs):
                labels[aid] = lab
            done += 1
            if done % 10 == 0 or done == len(batches) or done <= 3:
                el = time.time() - t0
                rate = done / max(el, 1e-3)
                eta = (len(batches) - done) / max(rate, 1e-3)
                print(
                    f"  {done}/{len(batches)} batches | "
                    f"{len(labels)}/{total_articles} arts | "
                    f"{el:.0f}s elapsed | eta {eta:.0f}s | "
                    f"tok in/out {in_tok}/{out_tok}",
                    flush=True,
                )

    # Aggregate per-day counts.
    per_day: dict[str, dict[str, int]] = {}
    for date, aid, _ in flat:
        d = per_day.setdefault(date, {"war": 0, "domestic": 0, "other": 0})
        lab = labels.get(aid, "OTHER").lower()
        d[lab] = d.get(lab, 0) + 1

    # ISO-week aggregation. ISO weeks 26..30 cover July 1943.
    per_week: dict[str, dict[str, tp.Any]] = {}
    for date, counts in sorted(per_day.items()):
        d = dt.date.fromisoformat(date)
        iso_year, iso_week, _ = d.isocalendar()
        wkey = f"{iso_year}-W{iso_week:02d}"
        b = per_week.setdefault(wkey, {"war": 0, "domestic": 0, "other": 0, "days": []})
        b["war"] += counts["war"]
        b["domestic"] += counts["domestic"]
        b["other"] += counts["other"]
        b["days"].append(date)

    # Compute ratios. ratio = war / (war + domestic) (so it's bounded in
    # [0,1] and "other" is excluded — what fraction of the political
    # foreground is war coverage, vs domestic coverage).
    for w, b in per_week.items():
        denom = b["war"] + b["domestic"]
        b["ratio_war_over_war_plus_domestic"] = (b["war"] / denom) if denom else None
        b["war_to_domestic"] = (b["war"] / b["domestic"]) if b["domestic"] else None

    out = {
        "model": MODEL,
        "classifier_prompt_summary": "WAR / DOMESTIC / OTHER, batched 10/call",
        "n_articles_total": total_articles,
        "n_articles_labelled": len(labels),
        "labels_per_article": labels,
        "per_day": per_day,
        "per_week": per_week,
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "wall_sec": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"wrote {OUT} | {len(labels)} labels | tokens in/out {in_tok}/{out_tok}")


if __name__ == "__main__":
    main()
