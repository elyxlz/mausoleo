from __future__ import annotations

import base64
import json
import pathlib as pl
import re
from collections import defaultdict

from mausoleo.eval.evaluate import compute_cer, compute_wer
from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict

BOOTSTRAP_DIR = pl.Path("eval/bootstrap_gt")
PREDICTIONS_DIR = pl.Path("eval/predictions")
GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
ISSUE_DATES = ["1885-06-15", "1910-06-15", "1940-04-01"]
OUTPUT = pl.Path("eval/eval_report.html")


def load_issue(path: pl.Path) -> Issue:
    return issue_from_dict(json.loads(path.read_text()))


def issue_to_markdown(issue: Issue) -> str:
    parts: list[str] = []
    for art in issue.articles:
        headline = art.headline or "(no headline)"
        parts.append(f"### {headline}")
        parts.append(f"*{art.unit_type} | pages {art.page_span}*")
        for para in art.paragraphs:
            parts.append(para.text)
        parts.append("")
    return "\n\n".join(parts)


def image_to_data_uri(path: pl.Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{data}"


def compute_metrics(gt_text: str, pred_text: str) -> dict[str, float]:
    if not pred_text.strip():
        return {"cer": 1.0, "wer": 1.0}
    return {
        "cer": compute_cer(gt_text, pred_text),
        "wer": compute_wer(gt_text, pred_text),
    }


def generate_html() -> str:
    config_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    config_texts: dict[str, dict[str, str]] = defaultdict(dict)

    gt_data: dict[str, dict] = {}
    for date in ISSUE_DATES:
        gt_path = BOOTSTRAP_DIR / date / "ground_truth.json"
        if not gt_path.exists():
            continue
        gt_issue = load_issue(gt_path)
        gt_text = extract_full_text(gt_issue)
        gt_data[date] = {
            "issue": gt_issue,
            "text": gt_text,
            "markdown": issue_to_markdown(gt_issue),
        }

    for pred_path in sorted(PREDICTIONS_DIR.glob("*.json")):
        m = re.match(r"(.+)_(18\d\d-\d\d-\d\d|19\d\d-\d\d-\d\d)\.json", pred_path.name)
        if not m:
            continue
        cfg_name = m.group(1)
        date = m.group(2)
        if date not in gt_data:
            continue
        try:
            pred_issue = load_issue(pred_path)
        except Exception:
            continue
        pred_text = extract_full_text(pred_issue)
        metrics = compute_metrics(gt_data[date]["text"], pred_text)
        config_metrics[cfg_name].append(metrics)
        config_texts[cfg_name][date] = issue_to_markdown(pred_issue)

    ranked: list[tuple[str, float, float, int]] = []
    for cfg, metrics_list in config_metrics.items():
        avg_cer = sum(m["cer"] for m in metrics_list) / len(metrics_list)
        avg_wer = sum(m["wer"] for m in metrics_list) / len(metrics_list)
        ranked.append((cfg, avg_cer, avg_wer, len(metrics_list)))
    ranked.sort(key=lambda x: x[1])

    parts: list[str] = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR Evaluation Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Georgia, serif; background: #faf9f6; color: #222; max-width: 1400px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 2em; margin-bottom: 10px; border-bottom: 3px solid #333; padding-bottom: 10px; }
  h2 { font-size: 1.5em; margin: 30px 0 10px; color: #444; }
  h3 { font-size: 1.1em; margin: 15px 0 5px; color: #555; }
  .date-section { margin: 30px 0; }
  .images { display: flex; gap: 10px; overflow-x: auto; padding: 10px 0; }
  .images img { max-height: 500px; border: 1px solid #ccc; }
  .gt-text { background: #f0f7e6; border: 2px solid #8cb369; border-radius: 8px; padding: 20px; margin: 15px 0; white-space: pre-wrap; font-size: 0.9em; line-height: 1.6; max-height: 600px; overflow-y: auto; }
  .leaderboard { width: 100%; border-collapse: collapse; margin: 20px 0; }
  .leaderboard th { background: #333; color: #fff; padding: 10px; text-align: left; position: sticky; top: 0; cursor: pointer; }
  .leaderboard td { padding: 8px 10px; border-bottom: 1px solid #ddd; }
  .leaderboard tr:hover { background: #eef; }
  .leaderboard tr.selected { background: #ddf; }
  .rank { font-weight: bold; color: #888; }
  .metric { font-family: monospace; }
  .good { color: #2a7; }
  .ok { color: #a80; }
  .bad { color: #c44; }
  .config-detail { display: none; margin: 20px 0; border: 2px solid #aac; border-radius: 8px; padding: 20px; background: #f8f8ff; }
  .config-detail.active { display: block; }
  .config-detail .pred-text { background: #fff; border: 1px solid #ccc; border-radius: 4px; padding: 15px; margin: 10px 0; white-space: pre-wrap; font-size: 0.85em; line-height: 1.5; max-height: 500px; overflow-y: auto; }
  .tabs { display: flex; gap: 5px; margin: 10px 0; }
  .tab { padding: 8px 16px; border: 1px solid #ccc; border-radius: 4px 4px 0 0; cursor: pointer; background: #eee; }
  .tab.active { background: #fff; border-bottom-color: #fff; font-weight: bold; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
  .badge-dates { background: #e0e0e0; }
</style>
</head>
<body>
<h1>Mausoleo di Roma — OCR Evaluation Report</h1>
<p>Comparing OCR pipeline configs against bootstrapped ground truth. Lower CER/WER = better.</p>
""")

    for date in ISSUE_DATES:
        if date not in gt_data:
            continue
        parts.append(f'<div class="date-section">')
        parts.append(f"<h2>Ground Truth: {date}</h2>")
        parts.append('<div class="images">')
        img_dir = GROUND_TRUTH_DIR / date
        for img_path in sorted(img_dir.glob("*.jpeg"), key=lambda p: int(p.stem)):
            uri = image_to_data_uri(img_path)
            parts.append(f'<img src="{uri}" alt="Page {img_path.stem}">')
        parts.append("</div>")
        parts.append(f'<div class="gt-text">{gt_data[date]["markdown"]}</div>')
        parts.append("</div>")

    parts.append("<h2>Leaderboard</h2>")
    parts.append('<p>Click a row to see its full OCR output for each date.</p>')
    parts.append('<table class="leaderboard" id="leaderboard">')
    parts.append("<thead><tr><th>#</th><th>Config</th><th>Dates</th><th>Avg CER ↑↓</th><th>Avg WER</th></tr></thead>")
    parts.append("<tbody>")
    for i, (cfg, cer, wer, n_dates) in enumerate(ranked):
        cer_class = "good" if cer < 0.5 else "ok" if cer < 1.0 else "bad"
        wer_class = "good" if wer < 0.5 else "ok" if wer < 1.0 else "bad"
        parts.append(f'<tr onclick="showConfig(\'{cfg}\')" data-cfg="{cfg}">')
        parts.append(f'<td class="rank">{i+1}</td>')
        parts.append(f"<td>{cfg}</td>")
        parts.append(f'<td><span class="badge badge-dates">{n_dates}</span></td>')
        parts.append(f'<td class="metric {cer_class}">{cer:.4f}</td>')
        parts.append(f'<td class="metric {wer_class}">{wer:.4f}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")

    for cfg, _, _, _ in ranked:
        parts.append(f'<div class="config-detail" id="detail-{cfg}">')
        parts.append(f"<h2>{cfg}</h2>")
        parts.append('<div class="tabs">')
        for j, date in enumerate(ISSUE_DATES):
            active = "active" if j == 0 else ""
            parts.append(f'<div class="tab {active}" onclick="switchTab(this, \'{cfg}\', \'{date}\')">{date}</div>')
        parts.append("</div>")
        for j, date in enumerate(ISSUE_DATES):
            active = "active" if j == 0 else ""
            text = config_texts.get(cfg, {}).get(date, "(no prediction for this date)")
            safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'<div class="tab-content {active}" id="tab-{cfg}-{date}">')
            parts.append(f'<div class="pred-text">{safe_text}</div>')
            parts.append("</div>")
        parts.append("</div>")

    parts.append("""
<script>
function showConfig(cfg) {
  document.querySelectorAll('.config-detail').forEach(d => d.classList.remove('active'));
  document.querySelectorAll('#leaderboard tr').forEach(r => r.classList.remove('selected'));
  const detail = document.getElementById('detail-' + cfg);
  if (detail) { detail.classList.add('active'); detail.scrollIntoView({behavior: 'smooth', block: 'start'}); }
  const row = document.querySelector('tr[data-cfg=\"' + cfg + '\"]');
  if (row) row.classList.add('selected');
}
function switchTab(el, cfg, date) {
  el.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  const parent = el.closest('.config-detail');
  parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  const content = document.getElementById('tab-' + cfg + '-' + date);
  if (content) content.classList.add('active');
}
</script>
</body></html>""")

    return "\n".join(parts)


if __name__ == "__main__":
    html = generate_html()
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT} ({len(html)} bytes)")
