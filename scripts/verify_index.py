"""Validate the hierarchical index output: every node valid JSON, parent links
intact, embeddings present, counts consistent. Run after build_index.py."""

from __future__ import annotations

import json
import pathlib
import sys

OUT = pathlib.Path("/tmp/mausoleo/eval/summaries")
LEVELS = ("paragraph", "article", "day", "week", "month")
EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2; was 1024 for BGE-M3


def main() -> int:
    nodes: dict[str, dict] = {}
    counts: dict[str, int] = {}
    errors: list[str] = []

    for lvl in LEVELS:
        files = sorted((OUT / lvl).glob("*.json"))
        counts[lvl] = len(files)
        for f in files:
            try:
                d = json.loads(f.read_text())
            except Exception as e:
                errors.append(f"json load failed {f}: {e}")
                continue
            for required in ("node_id", "level", "parent_id", "summary", "date_start", "date_end"):
                if required not in d:
                    errors.append(f"{f}: missing field {required}")
            if d["level"] != lvl:
                errors.append(f"{f}: level mismatch ({d['level']} vs dir {lvl})")
            if d["node_id"] in nodes:
                errors.append(f"duplicate node_id {d['node_id']}")
            nodes[d["node_id"]] = d
            emb = d.get("embedding") or []
            if len(emb) != EMBED_DIM:
                errors.append(f"{d['node_id']}: embedding dim {len(emb)} != {EMBED_DIM}")
            if not d.get("summary") or len(d["summary"].strip()) < 20:
                errors.append(f"{d['node_id']}: empty/short summary")

    # parent integrity
    for nid, d in nodes.items():
        pid = d.get("parent_id") or ""
        if d["level"] == "month":
            if pid != "":
                errors.append(f"{nid}: month should have empty parent_id")
            continue
        if not pid:
            errors.append(f"{nid}: missing parent_id")
            continue
        if pid not in nodes:
            errors.append(f"{nid}: parent {pid} not found")

    print(f"Counts: {counts}")
    print(f"Total nodes: {sum(counts.values())}")
    print(f"Errors: {len(errors)}")
    for e in errors[:30]:
        print(f"  ! {e}")
    if len(errors) > 30:
        print(f"  ... and {len(errors) - 30} more")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
