"""Render the §6.5 results section + RUNLOG once the runner finishes.

Updated 2026-05-03 for the rerun: case-3 metric is ratio-RMSE instead of
article-id-touched recall; embedding is now loaded; no dollar cap.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import textwrap
import typing as tp


AGG_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/aggregate.json")
RUNLOG_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/RUNLOG.md")
ORACLE_PATH = pathlib.Path("/tmp/mausoleo/eval/case_studies/case3_oracle_ratios.json")
OUT_PATH = pathlib.Path("/tmp/mausoleo/references/section_6_5_results.md")


def _fmt(v: tp.Any, n: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def _summary_str(d: dict[str, tp.Any], n_decimals: int = 2) -> str:
    if not d or d.get("n", 0) == 0:
        return "n/a"
    return (
        f"{_fmt(d['mean'], n_decimals)} "
        f"(min {_fmt(d['min'], n_decimals)}, max {_fmt(d['max'], n_decimals)})"
    )


def _row(case_label: str, metric: str, m: dict[str, tp.Any], b: dict[str, tp.Any], n: int = 2) -> str:
    return (
        f"| {case_label} | {metric} | "
        f"{_summary_str(m, n)} | {_summary_str(b, n)} |"
    )


def _quality_combined(j1: dict[str, tp.Any], j2: dict[str, tp.Any]) -> dict[str, tp.Any]:
    if not j1 or not j2:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": (j1["mean"] + j2["mean"]) / 2,
        "min": min(j1["min"], j2["min"]),
        "max": max(j1["max"], j2["max"]),
        "n": j1.get("n", 0) + j2.get("n", 0),
    }


CASE_LABELS = {
    "case1": "Case 1 (07-26 absent)",
    "case2": "Case 2 (07-25 regime change)",
    "case3": "Case 3 (comparative coverage)",
}


def render_table(agg: dict[str, tp.Any]) -> str:
    per_case = agg.get("per_case_stats", {})
    rows: list[str] = []
    for case_id in ("case1", "case2", "case3"):
        if case_id not in per_case:
            continue
        cb = per_case[case_id]
        label = CASE_LABELS[case_id]
        rows.append(_row(label, "Tool calls",
                          cb["efficiency_tool_calls"]["mausoleo"],
                          cb["efficiency_tool_calls"]["baseline"], n=1))
        rows.append(_row(label, "Chars read",
                          cb["efficiency_chars_read"]["mausoleo"],
                          cb["efficiency_chars_read"]["baseline"], n=0))
        if case_id == "case3":
            rows.append(_row(label, "Ratio RMSE (lower=better)",
                              cb.get("case3_ratio_rmse", {}).get("mausoleo", {}),
                              cb.get("case3_ratio_rmse", {}).get("baseline", {}), n=3))
            rows.append(_row(label, "Ratio MAE (lower=better)",
                              cb.get("case3_ratio_mae", {}).get("mausoleo", {}),
                              cb.get("case3_ratio_mae", {}).get("baseline", {}), n=3))
        else:
            rows.append(_row(label, "Recall vs GT",
                              cb["completeness_recall"]["mausoleo"],
                              cb["completeness_recall"]["baseline"], n=2))
        rows.append(_row(label, "Quality (judge mean)",
                          _quality_combined(cb["quality_judge1_mean"]["mausoleo"],
                                              cb["quality_judge2_mean"]["mausoleo"]),
                          _quality_combined(cb["quality_judge1_mean"]["baseline"],
                                              cb["quality_judge2_mean"]["baseline"]),
                          n=2))
    header_table = (
        "| Case | Metric | Mausoleo (mean, min, max) | Baseline (mean, min, max) |\n"
        "|---|---|---|---|\n"
    ) + "\n".join(rows)
    return header_table


def render_sign_tests(agg: dict[str, tp.Any]) -> str:
    per_case = agg.get("per_case_stats", {})
    sign_lines: list[str] = []
    for case_id in ("case1", "case2", "case3"):
        if case_id not in per_case:
            continue
        cb = per_case[case_id]
        sq = cb.get("sign_test_quality", {})
        sign_lines.append(
            f"- {CASE_LABELS[case_id]} quality (n=6, 3 trials × 2 judges): "
            f"M wins {sq.get('wins', 0)}, B wins {sq.get('losses', 0)}, "
            f"ties {sq.get('ties', 0)}; two-sided sign-test p = {_fmt(sq.get('p_value'), 3)}."
        )
        if case_id == "case3":
            sr = cb.get("sign_test_case3_rmse_lower_better", {})
            sign_lines.append(
                f"- {CASE_LABELS[case_id]} ratio-RMSE (n=3 trials, lower=better): "
                f"M wins {sr.get('wins', 0)}, B wins {sr.get('losses', 0)}, "
                f"ties {sr.get('ties', 0)}; two-sided sign-test p = {_fmt(sr.get('p_value'), 3)}."
            )
        else:
            sc = cb.get("sign_test_completeness", {})
            sign_lines.append(
                f"- {CASE_LABELS[case_id]} completeness (n=3 trials): "
                f"M wins {sc.get('wins', 0)}, B wins {sc.get('losses', 0)}, "
                f"ties {sc.get('ties', 0)}; two-sided sign-test p = {_fmt(sc.get('p_value'), 3)}."
            )
    return "\n".join(sign_lines)


def render_kappa(agg: dict[str, tp.Any]) -> str:
    per_case = agg.get("per_case_stats", {})
    out: list[str] = []
    for case_id in ("case1", "case2", "case3"):
        if case_id not in per_case:
            continue
        cb = per_case[case_id]
        k = cb.get("inter_judge_kappa_quality_mean", float("nan"))
        if isinstance(k, float):
            out.append(
                f"- {CASE_LABELS[case_id]}: Cohen's κ on discretised 0-5 quality means = {_fmt(k, 2)}."
            )
    return "\n".join(out)


def render_runlog_summary(agg: dict[str, tp.Any]) -> str:
    n_trials = len(agg.get("trials", []))
    tok_in = agg.get("tokens_in_total", 0)
    tok_out = agg.get("tokens_out_total", 0)
    wall = agg.get("wall_time_sec", 0.0)
    embed = agg.get("embedder_status", {})
    return textwrap.dedent(f"""

## Rerun 2026-05-03 final summary {dt.datetime.utcnow().isoformat()}Z

- **trials completed**: {n_trials} / 18 (3 cases × 2 systems × 3 trials)
- **wall time**: {wall/60:.1f} min ({wall:.0f} s)
- **tokens (Anthropic, OAuth subscription quota; dollar cost is meaningless)**:
  input {tok_in:,}, output {tok_out:,}.
- **embedder loaded**: {embed.get('loaded', False)} ({embed.get('model', 'n/a')}, dim {embed.get('dim', 'n/a')}); smoke-test nearest day to "Mussolini" = {embed.get('nearest_day_to_mussolini')}.
- **case-3 metric**: replaced article-id-touched recall with per-ISO-week
  war-fraction MAE/RMSE against an LLM-built oracle
  (`eval/case_studies/case3_oracle_ratios.json`, 6480 articles classified).
- **cost cap**: removed. Phantom dollar figure no longer used for control flow.
""")


def main() -> None:
    agg = json.loads(AGG_PATH.read_text())
    table = render_table(agg)
    sign = render_sign_tests(agg)
    kappa = render_kappa(agg)
    print(table)
    print()
    print(sign)
    print()
    print(kappa)
    with RUNLOG_PATH.open("a") as f:
        f.write(render_runlog_summary(agg))
    print(f"appended summary to {RUNLOG_PATH}")


if __name__ == "__main__":
    main()
