"""Researcher agent loop.

Uses Claude Sonnet 4.5 over OAuth (~/.claude/.credentials.json + the
``oauth-2025-04-20`` beta header). The same model and system prompt are
shared across both the Mausoleo and baseline runs; only the toolset differs.

Tracks: tool-call count, total characters returned by tools (the agent's
"context-side" read budget), final compiled answer, and per-call token
usage so the runner can compute spend.
"""
from __future__ import annotations

import dataclasses as dc
import json
import os
import time
import typing as tp

import anthropic

from mausoleo.case_studies import tools as t

OAUTH_BETA = "oauth-2025-04-20"
RESEARCHER_MODEL = "claude-sonnet-4-5-20250929"

# Prices per 1M tokens, USD. Sonnet 4.5 = $3 in / $15 out.
PRICING = {
    "claude-sonnet-4-5-20250929": {"in": 3.0, "out": 15.0},
    "claude-opus-4-5-20251101": {"in": 15.0, "out": 75.0},
    "claude-opus-4-5": {"in": 15.0, "out": 75.0},
}

CC_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."


def _load_token() -> str:
    p = os.path.expanduser("~/.claude/.credentials.json")
    with open(p) as f:
        creds = json.load(f)
    return creds["claudeAiOauth"]["accessToken"]


SYSTEM_PROMPT = (
    "You are a researcher agent investigating Il Messaggero (Rome), the daily "
    "newspaper of fascist-era Italy. The corpus is the 30 daily issues of July "
    "1943 (with the issue of 26 July absent from the source archive). Your job "
    "is to answer the user's research question by querying the corpus through "
    "the tools available, then to compile a single final answer.\n\n"
    "Guidelines:\n"
    "- Be efficient: you have at most 30 tool calls per task.\n"
    "- Explain your reasoning briefly between tool calls.\n"
    "- Cite specific dates, headlines and article ids in your final answer.\n"
    "- If a tool returns nothing useful, try a different query before giving up.\n"
    "- When you are ready, produce a single compiled answer of 250-600 words "
    "  in clear English. Do NOT call further tools after the final answer.\n"
    "- If the user's question requires structured numeric output (e.g. "
    "  per-week ratios), include those structured lines exactly in the format "
    "  the user requested, then add prose.\n"
    "- The final answer must be the last text message and should not include "
    "  the words 'I will now' or other deferral phrases — just the answer.\n"
)


@dc.dataclass
class TrialUsage:
    tool_calls: int = 0
    chars_read: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    final_answer: str = ""
    tool_call_log: list[dict[str, tp.Any]] = dc.field(default_factory=list)
    article_ids_touched: list[str] = dc.field(default_factory=list)
    error: str | None = None
    stopped_at_cap: bool = False


def _add_cost(model: str, usage: tp.Any, target: TrialUsage) -> None:
    p = PRICING.get(model, {"in": 3.0, "out": 15.0})
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    target.input_tokens += in_tok
    target.output_tokens += out_tok
    target.cost_usd += (in_tok / 1_000_000) * p["in"] + (out_tok / 1_000_000) * p["out"]


def _extract_article_ids(payload: tp.Any) -> list[str]:
    """Pull article ids out of any tool result, for completeness scoring."""
    out: list[str] = []
    if isinstance(payload, dict):
        # baseline_search results
        if "results" in payload and isinstance(payload["results"], list):
            for r in payload["results"]:
                if isinstance(r, dict):
                    aid = r.get("article_id") or r.get("node_id")
                    if isinstance(aid, str) and "_a" in aid:
                        out.append(aid)
        # node-level lookups
        if isinstance(payload.get("node_id"), str) and "_a" in payload["node_id"]:
            out.append(payload["node_id"])
        # children of a day → articles
        if "children" in payload and isinstance(payload["children"], list):
            for c in payload["children"]:
                if isinstance(c, dict):
                    cid = c.get("node_id")
                    if isinstance(cid, str) and "_a" in cid:
                        out.append(cid)
        # text on an article
        if payload.get("level") == "article" and isinstance(payload.get("node_id"), str):
            out.append(payload["node_id"])
        # baseline read_article
        if isinstance(payload.get("article_id"), str):
            out.append(payload["article_id"])
    return out


def run_trial(
    question: str,
    system: str,           # "mausoleo" | "baseline"
    *,
    seed: int = 0,
    max_tool_calls: int = 30,
    temperature: float = 0.7,
) -> TrialUsage:
    """Run one researcher-agent trial against one system.

    The seed is folded into the user message so different trials genuinely
    explore different paths under temperature > 0.
    """
    client = anthropic.Anthropic(
        auth_token=_load_token(),
        default_headers={"anthropic-beta": OAUTH_BETA},
    )
    if system == "mausoleo":
        schemas = t.MAUSOLEO_TOOL_SCHEMAS
        dispatch = t.dispatch_mausoleo
    elif system == "baseline":
        schemas = t.BASELINE_TOOL_SCHEMAS
        dispatch = t.dispatch_baseline
    else:
        raise ValueError(f"unknown system {system}")

    usage = TrialUsage()
    t0 = time.time()

    # Inject the historical-research SYSTEM_PROMPT as the first user-message
    # context block so OAuth still accepts the API system field as the
    # Claude Code identity (per the OAuth restriction).
    seeded_question = f"[trial-seed {seed}]\n\n{question}"
    user_first = f"{SYSTEM_PROMPT}\n\n---\n\n{seeded_question}"
    messages: list[dict[str, tp.Any]] = [
        {"role": "user", "content": user_first}
    ]

    for _ in range(max_tool_calls + 4):  # +4 to give room for final wrap-up
        try:
            resp = client.messages.create(
                model=RESEARCHER_MODEL,
                system=CC_SYSTEM,
                max_tokens=2200,
                temperature=temperature,
                tools=schemas,
                messages=messages,
            )
        except Exception as e:
            usage.error = f"{type(e).__name__}: {str(e)[:300]}"
            break
        _add_cost(RESEARCHER_MODEL, resp.usage, usage)
        # Append assistant turn unchanged.
        messages.append({"role": "assistant", "content": resp.content})

        # If model is done (no tool_use blocks), capture text and stop.
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            text_parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"]
            usage.final_answer = "\n\n".join(p for p in text_parts if p).strip()
            break

        # Resolve tool calls and append a single user turn carrying all
        # tool_result blocks; this MUST happen before the cap-check, otherwise
        # the conversation ends on an unresolved tool_use which the API rejects.
        tool_result_blocks: list[dict[str, tp.Any]] = []
        for tu in tool_uses:
            usage.tool_calls += 1
            try:
                payload = dispatch(tu.name, dict(tu.input))
            except Exception as e:
                payload = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            payload_json = json.dumps(payload, ensure_ascii=False, default=str)
            usage.chars_read += len(payload_json)
            usage.tool_call_log.append({
                "name": tu.name,
                "input": dict(tu.input),
                "output_chars": len(payload_json),
            })
            usage.article_ids_touched.extend(_extract_article_ids(payload))
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": payload_json,
            })
        messages.append({"role": "user", "content": tool_result_blocks})

        if usage.tool_calls >= max_tool_calls:
            usage.stopped_at_cap = True
            # Force compile: append the cap notice in a follow-up user turn.
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the tool-call cap. Stop calling tools "
                    "and produce your final compiled answer now (250-450 words, "
                    "English)."
                ),
            })
            try:
                final_resp = client.messages.create(
                    model=RESEARCHER_MODEL,
                    system=CC_SYSTEM,
                    max_tokens=2200,
                    temperature=temperature,
                    tools=schemas,
                    tool_choice={"type": "none"},
                    messages=messages,
                )
                _add_cost(RESEARCHER_MODEL, final_resp.usage, usage)
                text_parts = [getattr(b, "text", "") for b in final_resp.content if getattr(b, "type", None) == "text"]
                usage.final_answer = "\n\n".join(p for p in text_parts if p).strip()
            except Exception as e:
                usage.error = f"finalise: {type(e).__name__}: {str(e)[:200]}"
            break

    usage.elapsed_sec = round(time.time() - t0, 2)
    # De-dup but preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for a in usage.article_ids_touched:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    usage.article_ids_touched = uniq
    return usage
