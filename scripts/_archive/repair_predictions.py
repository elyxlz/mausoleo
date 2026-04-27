import dataclasses as dc
import json
import pathlib as pl

from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict
from mausoleo.ocr.operators.merge import _strip_markdown

PREDICTIONS_DIR = pl.Path("eval/predictions")


def repair_article(article: dict) -> dict:
    new_paras = []
    for para in article.get("paragraphs", []):
        text = para.get("text") or ""
        text = text.strip()

        if text.startswith("```") or text.startswith("{"):
            extracted_articles = extract_from_json_blob(text)
            if extracted_articles:
                return_articles = []
                for ea in extracted_articles:
                    ea_paras = ea.get("paragraphs", [])
                    if not ea_paras and "text" in ea:
                        ea_paras = [{"text": ea["text"]}]
                    return_articles.append({
                        "unit_type": ea.get("unit_type", article.get("unit_type", "article")),
                        "headline": ea.get("headline", article.get("headline")),
                        "paragraphs": ea_paras,
                        "page_span": ea.get("page_span", article.get("page_span", [])),
                    })
                return return_articles
            else:
                new_paras.append({"id": para.get("id", ""), "text": text})
        else:
            new_paras.append(para)

    article["paragraphs"] = new_paras
    return article


def extract_from_json_blob(text: str) -> list[dict] | None:
    cleaned = _strip_markdown(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "articles" in data:
            return data["articles"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    for suffix in ['"}]}', '"}]', '"]', '"}', '}', ']', '"} ]}', '"}  ]}']:
        try:
            data = json.loads(cleaned + suffix)
            if isinstance(data, dict) and "articles" in data:
                return data["articles"]
        except json.JSONDecodeError:
            continue

    return None


def repair_issue(issue_data: dict) -> dict:
    new_articles = []
    for article in issue_data.get("articles", []):
        result = repair_article(article)
        if isinstance(result, list):
            new_articles.extend(result)
        else:
            new_articles.append(result)

    for i, art in enumerate(new_articles):
        date = issue_data["date"]
        art["id"] = f"{date}_a{i:02d}"
        art["position_in_issue"] = i
        if "paragraphs" not in art:
            art["paragraphs"] = []
        for j, para in enumerate(art["paragraphs"]):
            if "id" not in para:
                para["id"] = f"{date}_a{i:02d}_p{j:02d}"
            if "text" not in para:
                para["text"] = ""

    issue_data["articles"] = new_articles
    return issue_data


def is_clean(issue: Issue) -> bool:
    for article in issue.articles:
        for para in article.paragraphs:
            t = (para.text or "").strip()
            if t.startswith("{") or t.startswith("```"):
                return False
    return True


def main() -> None:
    files = sorted(PREDICTIONS_DIR.glob("*.json"))
    repaired = 0
    for f in files:
        data = json.loads(f.read_text())
        issue = issue_from_dict(data)

        if is_clean(issue):
            n_arts = len(issue.articles)
            chars = len(extract_full_text(issue))
            print(f"  [OK ] {f.name:<50s} {n_arts:>3d} arts  {chars:>7d} chars")
            continue

        fixed = repair_issue(data)
        fixed_issue = issue_from_dict(fixed)

        if is_clean(fixed_issue):
            f.write_text(json.dumps(fixed, indent=2, ensure_ascii=False))
            n_arts = len(fixed_issue.articles)
            chars = len(extract_full_text(fixed_issue))
            print(f"  [FIX] {f.name:<50s} {n_arts:>3d} arts  {chars:>7d} chars")
            repaired += 1
        else:
            n_arts = len(fixed_issue.articles)
            chars = len(extract_full_text(fixed_issue))
            n_bad = sum(1 for a in fixed_issue.articles for p in a.paragraphs if (p.text or "").strip().startswith(("{", "```")))
            print(f"  [BAD] {f.name:<50s} {n_arts:>3d} arts  {chars:>7d} chars  ({n_bad} still bad)")

    print(f"\n  Repaired {repaired} files")


if __name__ == "__main__":
    main()
