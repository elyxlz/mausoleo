"""Render the four figures embedded in the dissertation main text.

Re-rendered with Seaborn for v132 final packaging. Page-fit width ~6 in
to sit cleanly inside the A4 main column at the 65 % pandoc width
attribute. DPI 160. No text overlay: labels, legends, value annotations
are checked at this width.

Figure 1 - calendar-tree architecture (matplotlib + seaborn whitegrid axes;
            Seaborn does not natively render tree diagrams)
Figure 2 - sns.barplot, tool calls per case, 3 cases x 2 systems
Figure 3 - sns.catplot/grouped barplot, quality rubric per case
Figure 4 - sns.barplot, OCR composite by pipeline pass
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = pathlib.Path("/tmp/mausoleo/references/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Consistent palette: tree neutral; results pair = teal vs amber.
MAUSOLEO_C = "#2a6f97"  # blue-teal
BASELINE_C = "#d68c45"  # warm amber
TREE_LEAF = "#cad2c5"
TREE_MID = "#84a98c"
TREE_TOP = "#52796f"

# Apply Seaborn theme globally before any plotting.
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="DejaVu Serif",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
)


# -------------------- Figure 1: calendar tree --------------------
def fig1_calendar_tree() -> pathlib.Path:
    """Hierarchy diagram. Seaborn cannot draw trees directly; we use
    matplotlib patches on Seaborn-styled axes."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 7.6)
    ax.axis("off")  # hide the whitegrid for a clean diagram.

    # Levels (y) from leaf at bottom to root at top.
    levels = [
        (6.6, "month: 1943-07", "1 node", TREE_TOP),
        (5.4, "weeks", "5 nodes (W26 tail - W30)", TREE_MID),
        (4.2, "days", "31 nodes (incl. 1943-07-26 absent)", TREE_MID),
        (3.0, "articles", "6,480 nodes", TREE_LEAF),
        (1.8, "paragraphs", "leaves: raw text + embedding", TREE_LEAF),
    ]
    for y, label, count, colour in levels:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.6, y - 0.30), 3.6, 0.60,
            boxstyle="round,pad=0.04", linewidth=0.7,
            facecolor=colour, edgecolor="#333"))
        ax.text(2.4, y, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="#1a1a1a")
        ax.text(4.5, y, count, ha="left", va="center", fontsize=8.0,
                color="#333")

    # Vertical "child" arrows (top-down).
    for y_top, y_bot in [(6.6, 5.4), (5.4, 4.2), (4.2, 3.0), (3.0, 1.8)]:
        ax.annotate("", xy=(2.4, y_bot + 0.32), xytext=(2.4, y_top - 0.32),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9))

    # Recursive-summarisation flow on the right (bottom-up arrows + label).
    for y_bot, y_top in [(1.8, 3.0), (3.0, 4.2), (4.2, 5.4), (5.4, 6.6)]:
        ax.annotate("", xy=(8.9, y_top - 0.32), xytext=(8.9, y_bot + 0.32),
                    arrowprops=dict(arrowstyle="-|>", color="#a4133c", lw=1.2))
    ax.text(8.9, 7.25, "recursive\nsummarisation\n(bottom-up)",
            ha="center", va="bottom", fontsize=7.5, color="#a4133c",
            fontweight="bold")

    # Top-left annotation - kept short to avoid overlap.
    ax.text(0.0, 7.25,
            "calendar-shaped\nhierarchical index",
            ha="left", va="bottom", fontsize=8.5, color="#1a1a1a",
            fontweight="bold")
    ax.text(0.0, -0.55,
            "leaf store: per-paragraph OCR + 384-d MiniLM embedding\n"
            "ClickHouse: single nodes table; 6,517 nodes total",
            ha="left", va="bottom", fontsize=7.0, color="#444")

    p = OUT / "fig1_calendar_tree.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# -------------------- Figure 2: tool-call bar chart --------------------
def fig2_tool_calls() -> pathlib.Path:
    cases = [
        "Case 1\n(26 July absent)",
        "Case 2\n(25 July regime change)",
        "Case 3\n(comparative coverage)",
    ]
    rows = []
    for case, m, b in zip(cases, [13.3, 12.3, 8.3], [27.0, 29.7, 28.3]):
        rows.append({"case": case, "system": "Mausoleo", "calls": m})
        rows.append({"case": case, "system": "Keyword baseline", "calls": b})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    sns.barplot(
        data=df,
        x="case",
        y="calls",
        hue="system",
        palette={"Mausoleo": MAUSOLEO_C, "Keyword baseline": BASELINE_C},
        ax=ax,
        edgecolor="#222",
        linewidth=0.6,
    )

    # Value annotations above each bar.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=7.5)

    ax.axhline(30, color="#555", linestyle=":", linewidth=0.8)
    ax.text(2.45, 30.4, "30-call cap", color="#555",
            fontsize=7.0, ha="right", va="bottom")

    ax.set_ylabel("Mean tool calls per trial (3 trials)")
    ax.set_xlabel("")
    ax.set_ylim(0, 35)
    ax.set_title("Tool-call cost per case study")
    ax.legend(loc="upper left", frameon=False)

    p = OUT / "fig2_tool_calls.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# -------------------- Figure 3: quality rubric per case --------------------
def fig3_quality_rubric() -> pathlib.Path:
    """Three small panels, one per case, grouped sns.barplot with
    dimension on x and system on hue."""
    runs_dir = pathlib.Path("/tmp/mausoleo/eval/case_studies/runs")
    per_dim: dict[str, dict[str, dict[str, list[float]]]] = {}

    def add(case: str, sysname: str, dim: str, val: float) -> None:
        per_dim.setdefault(case, {}).setdefault(sysname, {}).setdefault(dim, []).append(val)

    if runs_dir.exists():
        for jf in runs_dir.glob("*.json"):
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            cid = d.get("case_id")
            sysname = d.get("system")
            if not cid or not sysname:
                continue
            for jkey in ("judge1", "judge2"):
                jdat = d.get(jkey) or {}
                for dim in ("factual", "comprehensive", "insight"):
                    v = jdat.get(dim)
                    if v is not None:
                        add(cid, sysname, dim, float(v))

    cases = ["case1", "case2", "case3"]
    case_titles = {
        "case1": "Case 1\n26 July absent",
        "case2": "Case 2\n25 July regime change",
        "case3": "Case 3\ncomparative coverage",
    }
    dims = ["factual", "comprehensive", "insight"]
    dim_labels = {
        "factual": "Factual",
        "comprehensive": "Compreh.",
        "insight": "Insight",
    }

    fallback = {
        "case1": {"mausoleo": [4.78, 4.39, 4.50], "baseline": [4.50, 4.06, 4.11]},
        "case2": {"mausoleo": [4.94, 4.83, 4.78], "baseline": [4.61, 4.39, 4.33]},
        "case3": {"mausoleo": [4.22, 4.00, 3.94], "baseline": [3.50, 3.11, 3.22]},
    }

    rows = []
    for cid in cases:
        if per_dim.get(cid, {}).get("mausoleo") and per_dim[cid].get("baseline"):
            for i, d in enumerate(dims):
                m = float(np.mean(per_dim[cid]["mausoleo"].get(d, [fallback[cid]["mausoleo"][i]])))
                b = float(np.mean(per_dim[cid]["baseline"].get(d, [fallback[cid]["baseline"][i]])))
                rows.append({"case": cid, "dim": dim_labels[d], "system": "Mausoleo", "score": m})
                rows.append({"case": cid, "dim": dim_labels[d], "system": "Baseline", "score": b})
        else:
            for i, d in enumerate(dims):
                rows.append({"case": cid, "dim": dim_labels[d], "system": "Mausoleo",
                             "score": fallback[cid]["mausoleo"][i]})
                rows.append({"case": cid, "dim": dim_labels[d], "system": "Baseline",
                             "score": fallback[cid]["baseline"][i]})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 3.4), sharey=True)
    palette = {"Mausoleo": MAUSOLEO_C, "Baseline": BASELINE_C}
    for ax, cid in zip(axes, cases):
        sub = df[df["case"] == cid]
        sns.barplot(
            data=sub, x="dim", y="score", hue="system",
            palette=palette, ax=ax, edgecolor="#222", linewidth=0.4,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=1, fontsize=6.5)
        ax.set_title(case_titles[cid], fontsize=8.0)
        ax.set_xlabel("")
        ax.set_ylim(0, 5.6)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.tick_params(axis="x", labelsize=7.5)
        if ax is axes[0]:
            ax.set_ylabel("Mean judge score (0-5)")
        else:
            ax.set_ylabel("")
        # Only keep one legend (rightmost).
        if ax is axes[-1]:
            ax.legend(loc="lower right", frameon=False, fontsize=7.0)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.suptitle("Three-dimension judge rubric, Mausoleo vs keyword baseline",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()

    p = OUT / "fig3_quality_rubric.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# -------------------- Figure 4: OCR composite per pass --------------------
def fig4_ocr_composite() -> pathlib.Path:
    passes = [
        "30-min v1\nbaseline",
        "+col5\nQwen3",
        "+col2\nQwen2.5",
        "+YOLO\nQwen2.5",
        "+col6\nQwen3",
        "+fullpage\nstack",
        "post-corr v1\nLLM",
        "post-corr v2\nLLM",
    ]
    scores = [0.8824, 0.8854, 0.8865, 0.8880, 0.8893, 0.9057, 0.8997, 0.8950]
    is_postcorr = [False] * 6 + [True, True]
    df = pd.DataFrame({
        "pass": passes,
        "score": scores,
        "kind": [
            "LLM post-correction (regression)" if p else "Ensemble add (cumulative gain)"
            for p in is_postcorr
        ],
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    sns.barplot(
        data=df, x="pass", y="score", hue="kind",
        palette={
            "Ensemble add (cumulative gain)": "#3a86ff",
            "LLM post-correction (regression)": "#9d4edd",
        },
        ax=ax, edgecolor="#222", linewidth=0.5, dodge=False,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=1.5, fontsize=6.5)

    ax.set_ylim(0.86, 0.92)
    ax.set_ylabel("Composite OCR score (0-1)")
    ax.set_xlabel("")
    ax.set_title("OCR composite by ensemble pass; LLM post-correction regression")
    ax.tick_params(axis="x", labelsize=6.5)

    ax.axhline(0.89878, color="#a4133c", linestyle="--", linewidth=0.8)
    ax.text(-0.4, 0.89878 + 0.0008, "headline 0.899",
            color="#a4133c", fontsize=7.0, ha="left", va="bottom")

    ax.legend(loc="lower left", frameon=False, fontsize=7.0)

    p = OUT / "fig4_ocr_composite.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    paths = [
        fig1_calendar_tree(),
        fig2_tool_calls(),
        fig3_quality_rubric(),
        fig4_ocr_composite(),
    ]
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"{p.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
