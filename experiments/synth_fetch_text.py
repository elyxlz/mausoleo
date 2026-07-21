from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import io
import json
import pathlib as pl
import re
import sys
import urllib.error
import urllib.request

REPO = pl.Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "eval" / "autoresearch" / "synth" / "corpus"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
USER_AGENT = "mausoleo-synth-fetch/1.0"
MIN_BIRTH = 1800
MAX_BIRTH = 1900
MAX_DEATH = 1962
YEARS_RE = re.compile(r"(\d{4})-(\d{4})")
START_RE = re.compile(r"\*\*\* ?START OF.*$", re.MULTILINE)
END_RE = re.compile(r"\*\*\* ?END OF.*$", re.MULTILINE)
ALLOWED_RE = re.compile(r"[a-zA-ZàèéìíîòóùúÀÈÉÌÒÙçÇ0-9\s.,;:!?'’‘\"«»()\[\]—–\--]")
BANNED_SUBSTRINGS = ("gutenberg", "www.", "http", "ebook", "copyright", "trascri")

SEED_PASSAGES = [
    "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti, tutto a seni e a golfi, a seconda dello sporgere e del rientrare di quelli, vien, quasi a un tratto, a ristringersi, e a prender corso e figura di fiume, tra un promontorio a destra, e un'ampia costiera dall'altra parte.",
    "C'era una volta un pezzo di legno. Non era un legno di lusso, ma un semplice pezzo da catasta, di quelli che d'inverno si mettono nelle stufe e nei caminetti per accendere il fuoco e per riscaldare le stanze.",
    "Un tempo i Malavoglia erano stati numerosi come i sassi della strada vecchia di Trezza; ce n'erano persino ad Ognina, e ad Aci Castello, tutti buona e brava gente di mare, proprio all'opposto di quel che sembrava dal nomignolo, come dev'essere.",
    "Il mattino seguente, appena giorno, la gente cominciò ad affollarsi nella piazza del paese, davanti alla chiesa, per udire le ultime notizie giunte nella notte dai messi del governo.",
    "La sera scendeva lenta sulla città, e i lampioni a gas si accendevano l'uno dopo l'altro lungo il corso, mentre le carrozze passavano rumorose sul selciato e i venditori ambulanti gridavano le loro merci agli angoli delle vie.",
]


@dc.dataclass(frozen=True)
class CatalogEntry:
    book_id: int
    title: str
    authors: str


def http_get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def author_years(authors: str) -> tuple[int, int] | None:
    m = YEARS_RE.search(authors)
    return (int(m.group(1)), int(m.group(2))) if m else None


def entry_in_period(row: dict[str, str]) -> bool:
    if row["Language"] != "it" or row["Type"] != "Text":
        return False
    years = author_years(row["Authors"])
    if years is None:
        return False
    birth, death = years
    return MIN_BIRTH <= birth <= MAX_BIRTH and death <= MAX_DEATH


def fetch_catalog(cache_path: pl.Path) -> list[CatalogEntry]:
    if not cache_path.exists():
        cache_path.write_bytes(http_get(CATALOG_URL))
    reader = csv.DictReader(io.StringIO(cache_path.read_text(encoding="utf-8")))
    rows = [r for r in reader if entry_in_period(r)]
    return [CatalogEntry(int(r["Text#"]), r["Title"], r["Authors"]) for r in rows]


def strip_boilerplate(raw: str) -> str:
    m_start = START_RE.search(raw)
    m_end = END_RE.search(raw)
    start = m_start.end() if m_start else 0
    end = m_end.start() if m_end else len(raw)
    return raw[start:end]


def decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def download_book(entry: CatalogEntry) -> str | None:
    path = CORPUS_DIR / f"pg{entry.book_id}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    try:
        text = strip_boilerplate(decode_text(http_get(TEXT_URL.format(id=entry.book_id))))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    path.write_text(text, encoding="utf-8")
    return text


def merge_wrapped_lines(block: str) -> str:
    return re.sub(r"\s+", " ", block).strip()


def paragraph_ok(text: str, min_chars: int, max_chars: int) -> bool:
    if not min_chars <= len(text) <= max_chars:
        return False
    lower = text.lower()
    if any(b in lower for b in BANNED_SUBSTRINGS):
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / len(text) < 0.72:
        return False
    upper = sum(c.isupper() for c in text)
    if upper / max(1, letters) > 0.3:
        return False
    return len(ALLOWED_RE.findall(text)) / len(text) > 0.985


def segment_paragraphs(text: str, min_chars: int, max_chars: int) -> list[str]:
    blocks = [merge_wrapped_lines(b) for b in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if paragraph_ok(b, min_chars, max_chars)]


def fallback_chunks(min_chars: int, max_chars: int) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for i, passage in enumerate(SEED_PASSAGES * 40):
        if paragraph_ok(passage, min_chars, max_chars):
            chunks.append({"book_id": -1, "idx": i, "text": passage})
    return chunks


def build_chunks(entries: list[CatalogEntry], min_chars: int, max_chars: int) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for entry in entries:
        text = download_book(entry)
        if text is None:
            continue
        for idx, para in enumerate(segment_paragraphs(text, min_chars, max_chars)):
            chunks.append({"book_id": entry.book_id, "idx": idx, "text": para})
        print(f"pg{entry.book_id} {entry.title[:60]}: total chunks {len(chunks)}")
    return chunks


def write_outputs(chunks: list[dict[str, object]], source: str, n_books: int) -> None:
    chunks_path = CORPUS_DIR / "chunks.jsonl"
    chunks_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n")
    manifest = {"source": source, "books": n_books, "chunks": len(chunks)}
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_chars = sum(len(str(c["text"])) for c in chunks)
    print(f"source={source} books={n_books} chunks={len(chunks)} total_chars={total_chars}")
    print(f"wrote {chunks_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", type=int, default=40)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=1600)
    args = parser.parse_args()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        entries = fetch_catalog(CORPUS_DIR / "pg_catalog.csv")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"network unavailable ({exc}), using bundled seed passages", file=sys.stderr)
        write_outputs(fallback_chunks(args.min_chars, args.max_chars), "bundled_seed", 0)
        return
    selected = entries[: args.books]
    chunks = build_chunks(selected, args.min_chars, args.max_chars)
    if not chunks:
        write_outputs(fallback_chunks(args.min_chars, args.max_chars), "bundled_seed", 0)
        return
    write_outputs(chunks, "gutenberg", len(selected))


if __name__ == "__main__":
    main()
