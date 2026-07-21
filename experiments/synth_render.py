from __future__ import annotations

import argparse
import dataclasses as dc
import functools
import json
import pathlib as pl
import random
import re
import typing as tp

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

FloatArr = npt.NDArray[np.float32]
U8Arr = npt.NDArray[np.uint8]

REPO = pl.Path(__file__).resolve().parents[1]
SYNTH_DIR = REPO / "eval" / "autoresearch" / "synth"
CHUNKS_PATH = SYNTH_DIR / "corpus" / "chunks.jsonl"
PROMPT = "OCR:"
SCALE = 2
VOWELS = set("aeiouàèéìíòóùúAEIOUÀÈÉÌÒÙ")
URW = "/usr/share/fonts/opentype/urw-base35"
Era = tp.Literal["old", "mid", "late"]

FAMILIES: dict[str, dict[str, str]] = {
    "c059": {
        "regular": f"{URW}/C059-Roman.otf",
        "bold": f"{URW}/C059-Bold.otf",
        "italic": f"{URW}/C059-Italic.otf",
    },
    "nimbus": {
        "regular": f"{URW}/NimbusRoman-Regular.otf",
        "bold": f"{URW}/NimbusRoman-Bold.otf",
        "italic": f"{URW}/NimbusRoman-Italic.otf",
    },
    "p052": {
        "regular": f"{URW}/P052-Roman.otf",
        "bold": f"{URW}/P052-Bold.otf",
        "italic": f"{URW}/P052-Italic.otf",
    },
    "bookman": {
        "regular": f"{URW}/URWBookman-Light.otf",
        "bold": f"{URW}/URWBookman-Demi.otf",
        "italic": f"{URW}/URWBookman-LightItalic.otf",
    },
    "liberation": {
        "regular": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "bold": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "italic": "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    },
    "freeserif": {
        "regular": "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "bold": "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
        "italic": "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
    },
    "dejavu_cond": {
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
        "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Bold.ttf",
        "italic": "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf",
    },
}

ERA_WEIGHTS: list[tuple[Era, float]] = [("old", 0.35), ("mid", 0.3), ("late", 0.35)]
HEADLINE_STYLES = ["bold", "bold_upper", "smallcaps", "bold_italic"]


@dc.dataclass(frozen=True)
class Spec:
    family: str
    body_px: int
    leading: float
    col_width: int
    n_cols: int
    indent_em: float
    justify: bool
    hyphenate: bool
    headline_style: str | None
    subhead: bool
    era: Era
    n_paragraphs: int


@dc.dataclass(frozen=True)
class Chunk:
    book_id: int
    idx: int
    text: str


@dc.dataclass(frozen=True)
class LineItem:
    tokens: tuple[str, ...]
    justified: bool
    indent: int
    gap_before: int


@functools.cache
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def weighted_era(rng: random.Random) -> Era:
    r = rng.random()
    acc = 0.0
    for era, w in ERA_WEIGHTS:
        acc += w
        if r < acc:
            return era
    return "late"


def sample_spec(rng: random.Random) -> Spec:
    n_cols = 2 if rng.random() < 0.15 else 1
    has_headline = rng.random() < 0.55
    return Spec(
        family=rng.choice(list(FAMILIES)),
        body_px=rng.randint(12, 17),
        leading=rng.uniform(1.12, 1.38),
        col_width=rng.randint(300, 390) if n_cols == 2 else rng.randint(380, 600),
        n_cols=n_cols,
        indent_em=rng.choice([0.0, 1.0, 1.2, 1.5]),
        justify=rng.random() < 0.9,
        hyphenate=rng.random() < 0.85,
        headline_style=rng.choice(HEADLINE_STYLES) if has_headline else None,
        subhead=has_headline and rng.random() < 0.4,
        era=weighted_era(rng),
        n_paragraphs=rng.randint(1, 6),
    )


def load_chunks(path: pl.Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        chunks.append(Chunk(int(r["book_id"]), int(r["idx"]), str(r["text"])))
    return chunks


def archaize(text: str, rng: random.Random) -> str:
    if rng.random() < 0.75:
        text = re.sub(r"é\b", "è", text)
    if rng.random() < 0.1:
        text = re.sub(r"\bE\b", "E'", text, count=1)
    return text


def make_headline(chunks: list[Chunk], rng: random.Random) -> str:
    words = rng.choice(chunks).text.split()
    start = rng.randrange(max(1, len(words) - 8))
    picked = words[start : start + rng.randint(3, 7)]
    return " ".join(picked).strip(" .,;:!?'\"«»()-")


def sample_paragraphs(chunks: list[Chunk], spec: Spec, rng: random.Random) -> list[str]:
    start = rng.randrange(len(chunks))
    book = chunks[start].book_id
    picked: list[str] = []
    for c in chunks[start:]:
        if c.book_id != book or len(picked) >= spec.n_paragraphs:
            break
        picked.append(archaize(c.text, rng))
    return picked


def compose_gt(headline: str | None, subhead: str | None, paragraphs: list[str]) -> str:
    parts = [p for p in (headline, subhead) if p]
    return "\n".join(parts + paragraphs)


def hyphen_split(word: str, font: ImageFont.FreeTypeFont, avail: float) -> tuple[str, str] | None:
    if len(word) < 6 or not word.isalpha():
        return None
    for cut in range(len(word) - 3, 2, -1):
        if word[cut - 1] in VOWELS and word[cut] not in VOWELS and cut + 1 < len(word) and word[cut + 1] in VOWELS:
            head = word[:cut] + "-"
            if font.getlength(head) <= avail:
                return head, word[cut:]
    return None


def wrap_paragraph(text: str, font: ImageFont.FreeTypeFont, width: int, indent: int, hyphenate: bool) -> list[LineItem]:
    space = font.getlength(" ")
    lines: list[LineItem] = []
    cur: list[str] = []
    cur_w = 0.0
    queue = text.split()
    while queue:
        word = queue.pop(0)
        avail = width - (indent if not lines else 0)
        w = font.getlength(word)
        if not cur or cur_w + space + w <= avail:
            cur_w += w if not cur else space + w
            cur.append(word)
            continue
        split = hyphen_split(word, font, avail - cur_w - space) if hyphenate else None
        if split:
            cur.append(split[0])
            queue.insert(0, split[1])
        else:
            queue.insert(0, word)
        lines.append(LineItem(tuple(cur), True, indent if not lines else 0, 0))
        cur, cur_w = [], 0.0
    if cur:
        lines.append(LineItem(tuple(cur), False, indent if not lines else 0, 0))
    return lines


def layout_body(paragraphs: list[str], font: ImageFont.FreeTypeFont, spec: Spec, para_gap: int) -> list[LineItem]:
    indent = int(spec.indent_em * font.size)
    items: list[LineItem] = []
    for pi, para in enumerate(paragraphs):
        lines = wrap_paragraph(para, font, spec.col_width * SCALE, indent * SCALE, spec.hyphenate)
        if lines and pi > 0:
            lines[0] = dc.replace(lines[0], gap_before=para_gap)
        items.extend(lines)
    return items


def draw_line(draw: ImageDraw.ImageDraw, x0: int, y: int, item: LineItem, font: ImageFont.FreeTypeFont, width: int, justify: bool) -> None:
    space = font.getlength(" ")
    words_w = sum(font.getlength(t) for t in item.tokens)
    n_gaps = len(item.tokens) - 1
    gap = space
    if justify and item.justified and n_gaps > 0:
        gap = min(space * 3.2, (width - item.indent - words_w) / n_gaps)
        gap = max(space * 0.75, gap)
    x = float(x0 + item.indent)
    for token in item.tokens:
        draw.text((x, y), token, font=font, fill=0)
        x += font.getlength(token) + gap


def draw_smallcaps(draw: ImageDraw.ImageDraw, x: float, y: int, text: str, path: str, size: int) -> float:
    big = load_font(path, size)
    small = load_font(path, int(size * 0.78))
    for ch in text:
        font = big if (ch.isupper() or not ch.isalpha()) else small
        up = ch.upper()
        dy = big.getbbox("A")[1] - font.getbbox("A")[1] if ch.isalpha() else 0
        draw.text((x, y + dy + (0 if font is big else int(size * 0.22))), up, font=font, fill=0)
        x += font.getlength(up)
    return x


def smallcaps_width(text: str, path: str, size: int) -> float:
    big = load_font(path, size)
    small = load_font(path, int(size * 0.78))
    return sum((big if (c.isupper() or not c.isalpha()) else small).getlength(c.upper()) for c in text)


def headline_font_path(spec: Spec) -> str:
    fam = FAMILIES[spec.family]
    if spec.headline_style == "bold_italic":
        return fam["italic"]
    return fam["bold"]


def headline_text(text: str, style: str) -> str:
    return text.upper() if style == "bold_upper" else text


def wrap_centered(text: str, measure: tp.Callable[[str], float], width: int) -> list[str]:
    lines: list[str] = []
    cur: list[str] = []
    for word in text.split():
        cand = " ".join(cur + [word])
        if cur and measure(cand) > width:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_headline_block(draw: ImageDraw.ImageDraw, y: int, text: str, spec: Spec, width: int, x0: int) -> int:
    style = spec.headline_style or "bold"
    size = int(spec.body_px * SCALE * 1.55)
    path = headline_font_path(spec)
    shown = headline_text(text, style)
    if style == "smallcaps":
        measure: tp.Callable[[str], float] = lambda s: smallcaps_width(s, path, size)
    else:
        measure = lambda s: load_font(path, size).getlength(s)
    for line in wrap_centered(shown, measure, width):
        x = x0 + (width - measure(line)) / 2
        if style == "smallcaps":
            draw_smallcaps(draw, x, y, line, path, size)
        else:
            draw.text((x, y), line, font=load_font(path, size), fill=0)
        y += int(size * 1.25)
    return y


def draw_subhead_block(draw: ImageDraw.ImageDraw, y: int, text: str, spec: Spec, width: int, x0: int) -> int:
    size = int(spec.body_px * SCALE * 1.1)
    font = load_font(FAMILIES[spec.family]["italic"], size)
    for line in wrap_centered(text, font.getlength, width):
        draw.text((x0 + (width - font.getlength(line)) / 2, y), line, font=font, fill=0)
        y += int(size * 1.3)
    return y


def column_slices(items: list[LineItem], n_cols: int) -> list[list[LineItem]]:
    if n_cols == 1:
        return [items]
    half = (len(items) + 1) // 2
    return [items[:half], items[half:]]


def body_height(items: list[LineItem], line_h: int) -> int:
    return sum(line_h + it.gap_before for it in items)


def draw_edge_clutter(draw: ImageDraw.ImageDraw, spec: Spec, rng: random.Random, w: int, h: int, chunks: list[Chunk]) -> None:
    font = load_font(FAMILIES[spec.family]["regular"], spec.body_px * SCALE)
    line_h = int(spec.body_px * SCALE * spec.leading)
    if rng.random() < 0.55:
        x = rng.randint(2, 6) * SCALE
        draw.line([(x, 0), (x, h)], fill=rng.randint(30, 90), width=SCALE)
    if rng.random() < 0.55:
        x = w - rng.randint(2, 6) * SCALE
        draw.line([(x, 0), (x, h)], fill=rng.randint(30, 90), width=SCALE)
    if rng.random() < 0.45:
        words = rng.choice(chunks).text.split()
        side_left = rng.random() < 0.5
        for i, y in enumerate(range(rng.randint(0, line_h), h, line_h)):
            token = words[i % len(words)]
            tw = font.getlength(token)
            x_pos = -tw + rng.randint(2, 5) * SCALE if side_left else w - rng.randint(2, 5) * SCALE
            draw.text((x_pos, y), token, font=font, fill=0)


def render_page(headline: str | None, subhead: str | None, paragraphs: list[str], spec: Spec, rng: random.Random, chunks: list[Chunk]) -> Image.Image:
    font = load_font(FAMILIES[spec.family]["regular"], spec.body_px * SCALE)
    line_h = int(spec.body_px * SCALE * spec.leading)
    items = layout_body(paragraphs, font, spec, para_gap=rng.randint(0, line_h // 2))
    cols = column_slices(items, spec.n_cols)
    gutter = rng.randint(10, 18) * SCALE if spec.n_cols == 2 else 0
    margin = rng.randint(8, 22) * SCALE
    page_w = spec.n_cols * spec.col_width * SCALE + (spec.n_cols - 1) * gutter + 2 * margin
    head_h = (int(spec.body_px * SCALE * 4.4) if headline else 0) + (int(spec.body_px * SCALE * 2.2) if subhead else 0)
    page_h = head_h + max(body_height(c, line_h) for c in cols) + 2 * margin + line_h
    img = Image.new("L", (page_w, page_h), 255)
    draw = ImageDraw.Draw(img)
    y = margin
    if headline:
        y = draw_headline_block(draw, y, headline, spec, page_w - 2 * margin, margin) + int(line_h * 0.4)
    if subhead:
        y = draw_subhead_block(draw, y, subhead, spec, page_w - 2 * margin, margin) + int(line_h * 0.3)
    for ci, col in enumerate(cols):
        x0 = margin + ci * (spec.col_width * SCALE + gutter)
        cy = y
        for item in col:
            cy += item.gap_before
            draw_line(draw, x0, cy, item, font, spec.col_width * SCALE, spec.justify)
            cy += line_h
        if ci > 0:
            rx = x0 - gutter // 2
            draw.line([(rx, y), (rx, cy)], fill=40, width=SCALE)
    draw_edge_clutter(draw, spec, rng, page_w, page_h, chunks)
    return img


def render_bleed(chunks: list[Chunk], spec: Spec, rng: random.Random, size: tuple[int, int]) -> FloatArr:
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    font = load_font(FAMILIES[spec.family]["regular"], spec.body_px * SCALE)
    line_h = int(spec.body_px * SCALE * spec.leading)
    words = rng.choice(chunks).text.split() * 8
    wi = 0
    for y in range(0, size[1], line_h):
        x = 0.0
        while x < size[0]:
            token = words[wi % len(words)]
            draw.text((x, y), token, font=font, fill=0)
            x += font.getlength(token + " ")
            wi += 1
    return np.asarray(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT), dtype=np.float32) / 255.0


def as_f32(x: tp.Any) -> FloatArr:
    return np.asarray(x, dtype=np.float32)


def apply_ink_spread(a: FloatArr, rng: random.Random) -> FloatArr:
    if rng.random() < 0.15:
        return a
    return as_f32(cv2.erode(a, np.ones((3, 3), dtype=np.uint8), iterations=1))


def apply_ink_wear(a: FloatArr, nprng: np.random.Generator, era: Era) -> FloatArr:
    strength = {"old": 0.4, "mid": 0.28, "late": 0.2}[era]
    field = cv2.GaussianBlur(nprng.random(a.shape, dtype=np.float32), (0, 0), 1.2)
    wear = np.clip((field - (1 - strength * 0.5)) * 2.2, 0, 0.7)
    return as_f32(np.clip(a + wear * (a < 0.6), 0, 1))


def apply_bleed(a: FloatArr, bleed: FloatArr, rng: random.Random) -> FloatArr:
    alpha = rng.uniform(0.08, 0.3)
    soft = cv2.GaussianBlur(bleed, (0, 0), rng.uniform(1.0, 2.5))
    return as_f32(np.clip(a - alpha * (1 - soft), 0, 1))


def apply_illumination(a: FloatArr, nprng: np.random.Generator) -> FloatArr:
    h, w = a.shape
    field = nprng.random((max(2, h // 220), max(2, w // 220))).astype(np.float32)
    smooth = as_f32(cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC))
    return as_f32(np.clip(a * (0.82 + 0.22 * smooth), 0, 1))


def apply_blur(a: FloatArr, rng: random.Random, era: Era) -> FloatArr:
    sigma = {"old": rng.uniform(0.7, 1.5), "mid": rng.uniform(0.5, 1.1), "late": rng.uniform(0.3, 0.9)}[era]
    a = as_f32(cv2.GaussianBlur(a, (0, 0), sigma))
    if rng.random() < 0.2:
        k = rng.choice([3, 5])
        kernel = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        a = as_f32(cv2.filter2D(a, -1, kernel))
    return a


def apply_noise(a: FloatArr, nprng: np.random.Generator, era: Era) -> FloatArr:
    sigma = {"old": 0.05, "mid": 0.045, "late": 0.06}[era]
    a = a + nprng.normal(0, sigma, a.shape).astype(np.float32)
    salt = nprng.random(a.shape) < 0.0012
    pepper = nprng.random(a.shape) < 0.0012
    a = np.where(salt, 1.0, a)
    a = np.where(pepper, a * 0.25, a)
    return as_f32(np.clip(a, 0, 1))


def apply_tone(a: FloatArr, rng: random.Random, era: Era) -> FloatArr:
    if era == "late":
        black, white = rng.uniform(0.05, 0.18), rng.uniform(0.75, 0.92)
        gamma = rng.uniform(0.75, 1.0)
    else:
        black, white = rng.uniform(0.0, 0.1), rng.uniform(0.82, 0.98)
        gamma = rng.uniform(0.85, 1.1)
    a = as_f32(np.clip((a - black) / max(1e-3, white - black), 0, 1))
    return as_f32(np.power(a, gamma))


def apply_rotation(a: FloatArr, rng: random.Random) -> FloatArr:
    angle = rng.uniform(-1.2, 1.2)
    h, w = a.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return as_f32(cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def apply_jpeg(u8: U8Arr, rng: random.Random) -> U8Arr:
    quality = rng.randint(45, 88)
    ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return np.asarray(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)


def apply_tint(u8: U8Arr, rng: random.Random, era: Era) -> U8Arr:
    if era == "old":
        factors = (1.0, rng.uniform(0.93, 0.98), rng.uniform(0.8, 0.9))
    elif era == "mid":
        factors = (1.0, rng.uniform(0.96, 1.0), rng.uniform(0.88, 0.97))
    else:
        factors = (1.0, 1.0, rng.uniform(0.96, 1.0))
    rgb = np.stack([np.clip(u8 * f, 0, 255) for f in factors], axis=-1)
    return rgb.astype(np.uint8)


def degrade(page: Image.Image, bleed: FloatArr | None, spec: Spec, rng: random.Random, nprng: np.random.Generator) -> U8Arr:
    a = np.asarray(page, dtype=np.float32) / 255.0
    a = apply_ink_spread(a, rng)
    a = apply_ink_wear(a, nprng, spec.era)
    if bleed is not None:
        a = apply_bleed(a, bleed, rng)
    out_w = max(1, page.width // SCALE)
    out_h = max(1, page.height // SCALE)
    a = as_f32(cv2.resize(a, (out_w, out_h), interpolation=cv2.INTER_AREA))
    a = apply_illumination(a, nprng)
    a = apply_blur(a, rng, spec.era)
    a = apply_rotation(a, rng)
    a = apply_noise(a, nprng, spec.era)
    a = apply_tone(a, rng, spec.era)
    u8 = apply_jpeg((a * 255).astype(np.uint8), rng)
    return apply_tint(u8, rng, spec.era)


def build_record(gt: str, image_path: pl.Path) -> dict[str, tp.Any]:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{PROMPT}"},
            {"role": "assistant", "content": gt},
        ],
        "images": [str(image_path)],
    }


def generate_pair(i: int, seed: int, chunks: list[Chunk], crops_dir: pl.Path) -> tuple[dict[str, tp.Any], Spec]:
    rng = random.Random(f"{seed}:{i}")
    nprng = np.random.default_rng(seed * 1_000_003 + i)
    spec = sample_spec(rng)
    paragraphs = sample_paragraphs(chunks, spec, rng)
    headline = make_headline(chunks, rng) if spec.headline_style else None
    if headline is not None:
        headline = headline_text(headline, spec.headline_style or "bold")
    subhead = make_headline(chunks, rng).lower().capitalize() if spec.subhead else None
    page = render_page(headline, subhead, paragraphs, spec, rng, chunks)
    bleed = render_bleed(chunks, spec, rng, page.size) if rng.random() < 0.5 else None
    img = degrade(page, bleed, spec, rng, nprng)
    path = crops_dir / f"synth_{seed}_{i:05d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return build_record(compose_gt(headline, subhead, paragraphs), path), spec


def print_stats(records: list[dict[str, tp.Any]], specs: list[Spec]) -> None:
    chars = [len(r["messages"][1]["content"]) for r in records]
    print(f"generated {len(records)} pairs")
    print(f"gt chars: mean {np.mean(chars):.0f}, p50 {np.percentile(chars, 50):.0f}, p95 {np.percentile(chars, 95):.0f}")
    for field in ("family", "era", "headline_style", "n_cols"):
        counts: dict[str, int] = {}
        for s in specs:
            key = str(getattr(s, field))
            counts[key] = counts.get(key, 0) + 1
        print(f"{field}: {dict(sorted(counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl", type=pl.Path, default=SYNTH_DIR / "synth_pairs.jsonl")
    parser.add_argument("--crops-dir", type=pl.Path, default=SYNTH_DIR / "crops")
    args = parser.parse_args()
    chunks = load_chunks(CHUNKS_PATH)
    args.crops_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, tp.Any]] = []
    specs: list[Spec] = []
    for i in range(args.count):
        record, spec = generate_pair(i, args.seed, chunks, args.crops_dir)
        records.append(record)
        specs.append(spec)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{args.count}")
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print_stats(records, specs)
    print(f"wrote {args.jsonl}")


if __name__ == "__main__":
    main()
