"""Orchestrate the §6 case-study runs end-to-end.

Three cases × two systems (mausoleo, baseline) × three trials × two judges.
Persists each (case, system, trial, judge) result as a separate JSON file
under ``eval/case_studies/runs/`` and aggregates to ``aggregate.json``.

Token tracking only: the previous run's "spend" cap was a phantom dollar
figure derived from API list prices, but Mausoleo bills against the Claude
Max subscription rate-limit quota over OAuth, so the dollar figure is
meaningless. We log token totals (input + output) per trial in RUNLOG and
in aggregate.json instead, and never abort on a dollar threshold.

Case-3 metric: cases 1 and 2 still use article-id-touched recall against
hand-annotated GT. Case 3 instead reports MAE / RMSE of the agent's
declared per-week war-vs-domestic ratio against the oracle ratios at
``eval/case_studies/case3_oracle_ratios.json``. The agent is asked to
explicitly state its per-week ratio in its final answer; the runner
parses that and scores it.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import pathlib
import time
import typing as tp

from mausoleo.case_studies import agent, judges, stats, tools


RUNS_DIR = pathlib.Path("/tmp/mausoleo/eval/case_studies/runs")
GT_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/relevance_gt.json")
ORACLE_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/case3_oracle_ratios.json")
AGG_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/aggregate.json")
RUNLOG_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/RUNLOG.md")
DONE_FLAG = pathlib.Path("/tmp/mausoleo_phase2_done.txt")

# No dollar cap. Phantom budget cap removed in 2026-05-03 rerun: OAuth
# Sonnet/Opus calls bill against subscription rate-limit quota, not USD.
N_TRIALS = 3
SYSTEMS = ("mausoleo", "baseline")
TEMPERATURE = 0.7

CASES: list[dict[str, tp.Any]] = [
    {
        "case_id": "case1",
        "label": "case1_missing_07-26",
        "question": (
            "What was reported on 26 July 1943, the day after Mussolini's arrest, "
            "in Il Messaggero? Read the surrounding days if helpful, then compile "
            "your answer."
        ),
        "lead": True,
    },
    {
        "case_id": "case2",
        "label": "case2_july25_regime_change",
        "question": (
            "How did Il Messaggero cover the fall of Mussolini and the transition "
            "to the Badoglio government? Cite specific dates, headlines and the "
            "editorial-register shift across 25-27 July 1943."
        ),
        "lead": False,
    },
    {
        "case_id": "case3",
        "label": "case3_comparative_coverage",
        "question": (
            "How does the balance of war coverage vs domestic-politics coverage "
            "shift over July 1943 in Il Messaggero? Treat 'war' = Sicilian "
            "campaign + Axis military bulletins + Allied operations + war-economy "
            "mobilisation; 'domestic' = regime politics + Badoglio transition + "
            "civilian-front editorial + social policy.\n\n"
            "REQUIRED in your final answer: a per-ISO-week numeric estimate of "
            "the war fraction, defined as war / (war + domestic) (a value in "
            "[0,1]), in EXACTLY this format on its own line for each of the five "
            "ISO weeks 26-30 that overlap July 1943:\n\n"
            "  WEEK 1943-W26: war_fraction=<float>\n"
            "  WEEK 1943-W27: war_fraction=<float>\n"
            "  WEEK 1943-W28: war_fraction=<float>\n"
            "  WEEK 1943-W29: war_fraction=<float>\n"
            "  WEEK 1943-W30: war_fraction=<float>\n\n"
            "These five lines must appear verbatim. Then provide the rest of "
            "your prose: shifts observed, dates, headlines."
        ),
        "lead": False,
    },
]


_WEEK_LINE_RE = __import__("re").compile(
    r"WEEK\s+(1943-W\d{2})\s*[:\-]\s*war[_ ]fraction\s*=\s*([0-9]*\.?[0-9]+)",
    __import__("re").IGNORECASE,
)


def parse_week_ratios(answer: str) -> dict[str, float]:
    """Pull `WEEK 1943-WNN: war_fraction=X` triples out of agent prose."""
    out: dict[str, float] = {}
    if not answer:
        return out
    for m in _WEEK_LINE_RE.finditer(answer):
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        # Clip to [0,1] to be robust.
        if v > 1.0:
            v = v / 100.0 if v <= 100.0 else 1.0
        v = max(0.0, min(1.0, v))
        out[m.group(1).upper()] = v
    return out


def case3_ratio_score(
    answer: str, oracle: dict[str, dict[str, tp.Any]]
) -> dict[str, tp.Any]:
    """Score the agent's per-week war-fraction estimates against oracle."""
    target_weeks = ["1943-W26", "1943-W27", "1943-W28", "1943-W29", "1943-W30"]
    pred = parse_week_ratios(answer)
    gold: dict[str, float] = {}
    for w in target_weeks:
        if w in oracle:
            r = oracle[w].get("ratio_war_over_war_plus_domestic")
            if isinstance(r, (int, float)):
                gold[w] = float(r)
    diffs: list[float] = []
    per_week: list[dict[str, tp.Any]] = []
    for w in target_weeks:
        g = gold.get(w)
        p = pred.get(w)
        d = abs(p - g) if (g is not None and p is not None) else None
        per_week.append({"week": w, "gold": g, "pred": p, "abs_err": d})
        if d is not None:
            diffs.append(d)
    if not diffs:
        return {
            "metric": "ratio_war_over_war_plus_domestic",
            "weeks": target_weeks,
            "per_week": per_week,
            "n_weeks_scored": 0,
            "mae": None,
            "rmse": None,
            "parsed_ratios": pred,
        }
    mae = sum(diffs) / len(diffs)
    rmse = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
    return {
        "metric": "ratio_war_over_war_plus_domestic",
        "weeks": target_weeks,
        "per_week": per_week,
        "n_weeks_scored": len(diffs),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "parsed_ratios": pred,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _save_trial(case_id: str, system: str, trial: int,
                payload: dict[str, tp.Any]) -> pathlib.Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    p = RUNS_DIR / f"{case_id}_{system}_t{trial}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return p


def _save_judge(case_id: str, system: str, trial: int, judge_name: str,
                payload: dict[str, tp.Any]) -> pathlib.Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    p = RUNS_DIR / f"{case_id}_{system}_t{trial}_{judge_name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return p


def _gt_for(case_id: str) -> dict[str, tp.Any]:
    gt = json.loads(GT_PATH.read_text())
    return gt[case_id]


def _oracle_per_week() -> dict[str, dict[str, tp.Any]]:
    if not ORACLE_PATH.exists():
        return {}
    return json.loads(ORACLE_PATH.read_text()).get("per_week", {})


def _completeness(touched: list[str], gt_articles: list[dict[str, str]]) -> dict[str, tp.Any]:
    gt_ids = {g["article_id"] for g in gt_articles}
    touched_set = set(touched)
    hits = touched_set & gt_ids
    recall = len(hits) / max(len(gt_ids), 1)
    return {
        "gt_size": len(gt_ids),
        "touched_relevant": len(hits),
        "recall": recall,
        "touched_total": len(touched_set),
    }


def _append_runlog(line: str) -> None:
    RUNLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNLOG_PATH.open("a") as f:
        f.write(line.rstrip() + "\n")


# ---------------------------------------------------------------------------
# Spend bookkeeping
# ---------------------------------------------------------------------------

@dc.dataclass
class Spend:
    researcher_usd: float = 0.0
    judges_usd: float = 0.0

    @property
    def total(self) -> float:
        return self.researcher_usd + self.judges_usd


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(*, only_cases: tuple[str, ...] | None = None) -> dict[str, tp.Any]:
    spend = Spend()
    tok_in = 0
    tok_out = 0
    t0 = time.time()
    _append_runlog(f"\n## run started {_now()} (no-budget-cap rerun)")

    # Force-load the embedder so semantic + hybrid use real vectors.
    embed_status = tools.ensure_embedder()
    _append_runlog(f"  embedder status: {json.dumps(embed_status, default=str)}")

    aggregate: dict[str, tp.Any] = {
        "started": _now(),
        "embedder_status": embed_status,
        "trials": [],
    }
    oracle_per_week = _oracle_per_week()

    for case in CASES:
        if only_cases and case["case_id"] not in only_cases:
            continue
        gt = _gt_for(case["case_id"])
        for system in SYSTEMS:
            for trial in range(1, N_TRIALS + 1):
                seed = trial * 1009 + hash(case["case_id"]) % 997
                _append_runlog(
                    f"- starting {case['case_id']} / {system} / trial {trial} "
                    f"(seed={seed}) at tok_in={tok_in} tok_out={tok_out}"
                )
                trial_t0 = time.time()
                u = agent.run_trial(
                    question=case["question"],
                    system=system,
                    seed=seed,
                    max_tool_calls=30,
                    temperature=TEMPERATURE,
                )
                spend.researcher_usd += u.cost_usd
                tok_in += u.input_tokens
                tok_out += u.output_tokens
                if case["case_id"] == "case3":
                    # Article-touched recall is meaningless here; we keep
                    # the touched set for diagnostics but score on ratios.
                    comp = _completeness(u.article_ids_touched, gt.get("articles", []))
                    ratio = case3_ratio_score(u.final_answer, oracle_per_week)
                    comp["ratio_metric"] = ratio
                else:
                    comp = _completeness(u.article_ids_touched, gt["articles"])
                trial_payload: dict[str, tp.Any] = {
                    "case_id": case["case_id"],
                    "system": system,
                    "trial": trial,
                    "seed": seed,
                    "question": case["question"],
                    "tool_calls": u.tool_calls,
                    "stopped_at_cap": u.stopped_at_cap,
                    "chars_read": u.chars_read,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost_usd_phantom": round(u.cost_usd, 5),
                    "elapsed_sec": u.elapsed_sec,
                    "completeness": comp,
                    "final_answer": u.final_answer,
                    "tool_call_log": u.tool_call_log,
                    "article_ids_touched": u.article_ids_touched,
                    "error": u.error,
                    "timestamp": _now(),
                }
                _save_trial(case["case_id"], system, trial, trial_payload)

                # Judges (skip if final answer is empty AND we're not the
                # baseline-null case, which we want scored as 0).
                judge_payloads: list[dict[str, tp.Any]] = []
                if not u.error or u.final_answer:
                    j1 = judges.judge_one(case["question"], u.final_answer, system)
                    spend.judges_usd += j1.cost_usd
                    j1d = dc.asdict(j1)
                    j1d.update({"case_id": case["case_id"], "system": system, "trial": trial})
                    _save_judge(case["case_id"], system, trial, "judge1", j1d)
                    judge_payloads.append(j1d)

                    j2 = judges.judge_two(case["question"], u.final_answer, system)
                    spend.judges_usd += j2.cost_usd
                    j2d = dc.asdict(j2)
                    j2d.update({"case_id": case["case_id"], "system": system, "trial": trial})
                    _save_judge(case["case_id"], system, trial, "judge2", j2d)
                    judge_payloads.append(j2d)
                else:
                    # If the agent crashed before producing any answer, still
                    # score 0 across the board so the trial appears in stats.
                    for jn in ("judge1", "judge2"):
                        empty = {
                            "judge": jn, "model": "n/a",
                            "factual": 0.0, "comprehensive": 0.0, "insight": 0.0,
                            "rationale": "no answer produced (agent error)",
                            "cost_usd": 0.0, "raw_text": "",
                            "case_id": case["case_id"], "system": system, "trial": trial,
                            "error": u.error,
                        }
                        _save_judge(case["case_id"], system, trial, jn, empty)
                        judge_payloads.append(empty)

                trial_row: dict[str, tp.Any] = {
                    "case_id": case["case_id"],
                    "system": system,
                    "trial": trial,
                    "tool_calls": u.tool_calls,
                    "chars_read": u.chars_read,
                    "completeness_recall": comp.get("recall", 0.0),
                    "judge1_mean": (judge_payloads[0]["factual"] + judge_payloads[0]["comprehensive"] + judge_payloads[0]["insight"]) / 3,
                    "judge2_mean": (judge_payloads[1]["factual"] + judge_payloads[1]["comprehensive"] + judge_payloads[1]["insight"]) / 3,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "trial_elapsed_sec": round(time.time() - trial_t0, 2),
                }
                if case["case_id"] == "case3":
                    rmse = comp.get("ratio_metric", {}).get("rmse")
                    mae = comp.get("ratio_metric", {}).get("mae")
                    n_w = comp.get("ratio_metric", {}).get("n_weeks_scored")
                    trial_row["case3_rmse"] = rmse
                    trial_row["case3_mae"] = mae
                    trial_row["case3_n_weeks_scored"] = n_w
                aggregate["trials"].append(trial_row)
                ratio_log = ""
                if case["case_id"] == "case3":
                    ratio_log = (
                        f" rmse={trial_row.get('case3_rmse')} "
                        f"mae={trial_row.get('case3_mae')} "
                        f"weeks={trial_row.get('case3_n_weeks_scored')}/5"
                    )
                _append_runlog(
                    f"  done {case['case_id']}/{system}/t{trial}: "
                    f"calls={u.tool_calls} chars={u.chars_read} "
                    f"recall={comp.get('recall', 0):.2f}{ratio_log} "
                    f"j1={judge_payloads[0]['factual']:.1f}/{judge_payloads[0]['comprehensive']:.1f}/{judge_payloads[0]['insight']:.1f} "
                    f"j2={judge_payloads[1]['factual']:.1f}/{judge_payloads[1]['comprehensive']:.1f}/{judge_payloads[1]['insight']:.1f} "
                    f"tok_in={u.input_tokens} tok_out={u.output_tokens} ({u.elapsed_sec}s)"
                )
                # Persist running aggregate after every trial so we never lose
                # data on a crash.
                aggregate["tokens_in_total"] = tok_in
                aggregate["tokens_out_total"] = tok_out
                aggregate["phantom_cost_usd"] = round(spend.total, 4)
                AGG_PATH.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, default=str))

    aggregate["finished"] = _now()
    aggregate["wall_time_sec"] = round(time.time() - t0, 2)
    aggregate["per_case_stats"] = compute_per_case_stats(aggregate["trials"])
    aggregate["tokens_in_total"] = tok_in
    aggregate["tokens_out_total"] = tok_out
    aggregate["phantom_cost_usd"] = round(spend.total, 4)
    AGG_PATH.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, default=str))
    DONE_FLAG.write_text("DONE: rerun complete\n")
    _append_runlog(
        f"\nfinished tokens in/out {tok_in}/{tok_out} "
        f"(phantom $={spend.total:.2f}) wall={aggregate['wall_time_sec']:.0f}s"
    )
    return aggregate


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_per_case_stats(trials: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
    by_case: dict[str, dict[str, list[dict[str, tp.Any]]]] = {}
    for tr in trials:
        by_case.setdefault(tr["case_id"], {"mausoleo": [], "baseline": []})
        by_case[tr["case_id"]][tr["system"]].append(tr)

    out: dict[str, tp.Any] = {}
    for case_id, sysmap in by_case.items():
        m = sysmap.get("mausoleo", [])
        b = sysmap.get("baseline", [])
        case_block: dict[str, tp.Any] = {}
        metric_keys = [
            ("efficiency_tool_calls", "tool_calls"),
            ("efficiency_chars_read", "chars_read"),
            ("completeness_recall", "completeness_recall"),
            ("quality_judge1_mean", "judge1_mean"),
            ("quality_judge2_mean", "judge2_mean"),
        ]
        if case_id == "case3":
            metric_keys.append(("case3_ratio_rmse", "case3_rmse"))
            metric_keys.append(("case3_ratio_mae", "case3_mae"))
        for metric_name, key in metric_keys:
            mausoleo_vals = [tr[key] for tr in m if tr.get(key) is not None]
            baseline_vals = [tr[key] for tr in b if tr.get(key) is not None]

            case_block[metric_name] = {
                "mausoleo": _summary(mausoleo_vals),
                "baseline": _summary(baseline_vals),
            }

        # Sign test on the *quality* axis (judge mean). Pair judge1 + judge2
        # observations: 3 trials × 2 judges = 6 paired observations.
        pairs_quality: list[tuple[float, float]] = []
        for trial_idx in range(min(len(m), len(b))):
            for jk in ("judge1_mean", "judge2_mean"):
                pairs_quality.append((m[trial_idx][jk], b[trial_idx][jk]))
        case_block["sign_test_quality"] = stats.sign_test(pairs_quality)

        if case_id == "case3":
            # Case 3 paired sign test on RMSE (lower is better; treat
            # baseline-as-x, mausoleo-as-y so we preserve the "mausoleo
            # wins => mausoleo's value < baseline's value" semantics by
            # negating).
            pairs_rmse: list[tuple[float, float]] = []
            for ti in range(min(len(m), len(b))):
                mr = m[ti].get("case3_rmse")
                br = b[ti].get("case3_rmse")
                if mr is None or br is None:
                    continue
                # Re-frame so larger=better for mausoleo: pass (-mausoleo, -baseline)
                pairs_rmse.append((-mr, -br))
            case_block["sign_test_case3_rmse_lower_better"] = stats.sign_test(pairs_rmse)
        else:
            # Sign test on completeness recall.
            pairs_complete: list[tuple[float, float]] = []
            for trial_idx in range(min(len(m), len(b))):
                pairs_complete.append((m[trial_idx]["completeness_recall"],
                                       b[trial_idx]["completeness_recall"]))
            case_block["sign_test_completeness"] = stats.sign_test(pairs_complete)

        # Inter-judge κ (discretised 0-5, paired across (system, trial)).
        if m and b:
            cats = list(range(6))
            j1_disc = []
            j2_disc = []
            for tr in m + b:
                j1_disc.append(stats.discretise_score(tr["judge1_mean"]))
                j2_disc.append(stats.discretise_score(tr["judge2_mean"]))
            case_block["inter_judge_kappa_quality_mean"] = stats.cohen_kappa(j1_disc, j2_disc, cats)

        out[case_id] = case_block
    return out


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


if __name__ == "__main__":
    import sys
    only = tuple(sys.argv[1:]) if len(sys.argv) > 1 else None
    print(json.dumps(run_all(only_cases=only), default=str, indent=2)[:2000])
