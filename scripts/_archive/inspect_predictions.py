import json
import pathlib as pl

PRED_DIR = pl.Path("eval/predictions")

for f in sorted(PRED_DIR.glob("*.json")):
    d = json.loads(f.read_text())
    arts = d.get("articles", [])
    n_bad = 0
    total_chars = 0
    for a in arts:
        for p in a.get("paragraphs", []):
            t = p.get("text") or ""
            total_chars += len(t)
            if t.strip().startswith("{") or t.strip().startswith("```"):
                n_bad += 1

    status = "BAD" if n_bad > 0 else "OK"
    print(f"  [{status:3s}] {f.name:<50s} {len(arts):>3d} arts  {total_chars:>7d} chars  {n_bad} bad paras")
