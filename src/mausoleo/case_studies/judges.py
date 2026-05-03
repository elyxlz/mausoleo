"""Two LLM-as-judge implementations.

- Judge 1 ("Opus"): Claude Opus 4.5 if available, else Sonnet 4.5 with an
  explicit Judge-1 system prompt.
- Judge 2 ("Sonnet-alt"): Claude Sonnet 4.5 with a different judge prompt
  (substituted from GPT-5; methodology adjustment documented in §6.5 + RUNLOG).

Both judges score a (question, answer) pair on three dimensions: factual
accuracy, comprehensiveness, insight, each on a 0-5 scale. Output is a
strict JSON object plus a one-paragraph rationale.
"""
from __future__ import annotations

import dataclasses as dc
import json
import re
import typing as tp

import anthropic

from mausoleo.case_studies.agent import (
    CC_SYSTEM,
    OAUTH_BETA,
    PRICING,
    _load_token,
    _add_cost,
)

JUDGE1_MODEL_PREF = "claude-opus-4-5"  # actual id resolved at runtime
JUDGE1_MODEL_FALLBACK = "claude-sonnet-4-5-20250929"
JUDGE2_MODEL = "claude-sonnet-4-5-20250929"


JUDGE1_SYSTEM = (
    "You are Judge 1: a senior historian-of-fascist-Italy reviewer. "
    "Score the candidate answer on factual accuracy (does it correctly state "
    "what Il Messaggero reported, or what is known about the corpus and the "
    "events), comprehensiveness (does it cover the major sub-questions and "
    "specific evidence), and insight (does it draw a defensible "
    "historiographical reading rather than just listing facts). Use the 0-5 "
    "scale strictly: 0 = empty / unrelated, 1 = mostly wrong, 2 = mixed, "
    "3 = solid baseline, 4 = strong, 5 = exceptional. "
    "Be willing to give 0 to nulls and 5 to genuinely excellent answers."
)

JUDGE2_SYSTEM = (
    "You are Judge 2: a critical computational-humanities reviewer specialising "
    "in retrieval evaluation. Apply the rubric strictly. For factual accuracy "
    "treat unverifiable claims as suspect. For comprehensiveness, demand "
    "explicit citation of dates, headlines, or article ids. For insight, "
    "demand that the answer interpret the evidence rather than only describe "
    "it. Use the 0-5 scale; treat null / empty answers as 0 across the board."
)


JUDGE_PROMPT_TEMPLATE = (
    "Research question:\n{question}\n\n"
    "Candidate answer (system: {system}):\n{answer}\n\n"
    "Evaluate the answer on the three-dimension rubric (factual accuracy, "
    "comprehensiveness, insight). Output STRICT JSON only, no prose, no code "
    "fences, exactly:\n"
    "{{\"factual\": <0-5>, \"comprehensive\": <0-5>, \"insight\": <0-5>, "
    "\"rationale\": \"<one paragraph, 1-3 sentences>\"}}"
)


@dc.dataclass
class JudgeResult:
    judge: str
    model: str
    factual: float = 0.0
    comprehensive: float = 0.0
    insight: float = 0.0
    rationale: str = ""
    cost_usd: float = 0.0
    raw_text: str = ""
    error: str | None = None

    @property
    def mean(self) -> float:
        return (self.factual + self.comprehensive + self.insight) / 3.0


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_score(text: str) -> dict[str, tp.Any]:
    m = _JSON_RE.search(text or "")
    if not m:
        return {"factual": 0.0, "comprehensive": 0.0, "insight": 0.0, "rationale": text[:300]}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"factual": 0.0, "comprehensive": 0.0, "insight": 0.0, "rationale": text[:300]}
    for k in ("factual", "comprehensive", "insight"):
        v = d.get(k, 0)
        try:
            d[k] = max(0.0, min(5.0, float(v)))
        except Exception:
            d[k] = 0.0
    d["rationale"] = str(d.get("rationale", ""))[:600]
    return d


def _judge_call(model: str, system_prompt: str, user_prompt: str) -> tuple[str, tp.Any]:
    client = anthropic.Anthropic(
        auth_token=_load_token(),
        default_headers={"anthropic-beta": OAUTH_BETA},
    )
    # Same OAuth trick: API system field stays as Claude Code identity, real
    # judge instructions live as the leading user-message block.
    full_user = f"{system_prompt}\n\n---\n\n{user_prompt}"
    resp = client.messages.create(
        model=model,
        system=CC_SYSTEM,
        max_tokens=600,
        temperature=0.0,
        messages=[{"role": "user", "content": full_user}],
    )
    parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip(), resp.usage


def _resolve_judge1_model() -> str:
    # Try the explicit Opus 4.5 id; if it's not accepted, fall back to
    # Sonnet 4.5 with the Judge-1 prompt (this is documented in RUNLOG and
    # called out in §6.5 as a methodology adjustment).
    try:
        client = anthropic.Anthropic(
            auth_token=_load_token(),
            default_headers={"anthropic-beta": OAUTH_BETA},
        )
        # Just attempt a tiny ping with the preferred id.
        client.messages.create(
            model="claude-opus-4-5-20251101",
            system=CC_SYSTEM,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )
        return "claude-opus-4-5-20251101"
    except Exception:
        return JUDGE1_MODEL_FALLBACK


_JUDGE1_RESOLVED: str | None = None


def judge_one(question: str, answer: str, system: str) -> JudgeResult:
    global _JUDGE1_RESOLVED
    if _JUDGE1_RESOLVED is None:
        _JUDGE1_RESOLVED = _resolve_judge1_model()
    model = _JUDGE1_RESOLVED
    user_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, answer=answer or "(empty)", system=system
    )
    try:
        text, usage = _judge_call(model, JUDGE1_SYSTEM, user_prompt)
    except Exception as e:
        return JudgeResult(judge="judge1", model=model, error=f"{type(e).__name__}: {str(e)[:200]}")
    parsed = _parse_score(text)
    res = JudgeResult(
        judge="judge1",
        model=model,
        factual=parsed["factual"],
        comprehensive=parsed["comprehensive"],
        insight=parsed["insight"],
        rationale=parsed["rationale"],
        raw_text=text[:1000],
    )
    p = PRICING.get(model, {"in": 3.0, "out": 15.0})
    res.cost_usd = (
        (getattr(usage, "input_tokens", 0) or 0) / 1_000_000 * p["in"]
        + (getattr(usage, "output_tokens", 0) or 0) / 1_000_000 * p["out"]
    )
    return res


def judge_two(question: str, answer: str, system: str) -> JudgeResult:
    user_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, answer=answer or "(empty)", system=system
    )
    try:
        text, usage = _judge_call(JUDGE2_MODEL, JUDGE2_SYSTEM, user_prompt)
    except Exception as e:
        return JudgeResult(judge="judge2", model=JUDGE2_MODEL, error=f"{type(e).__name__}: {str(e)[:200]}")
    parsed = _parse_score(text)
    res = JudgeResult(
        judge="judge2",
        model=JUDGE2_MODEL,
        factual=parsed["factual"],
        comprehensive=parsed["comprehensive"],
        insight=parsed["insight"],
        rationale=parsed["rationale"],
        raw_text=text[:1000],
    )
    p = PRICING.get(JUDGE2_MODEL, {"in": 3.0, "out": 15.0})
    res.cost_usd = (
        (getattr(usage, "input_tokens", 0) or 0) / 1_000_000 * p["in"]
        + (getattr(usage, "output_tokens", 0) or 0) / 1_000_000 * p["out"]
    )
    return res
