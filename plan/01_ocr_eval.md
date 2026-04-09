# Phase 1: OCR Evaluation Suite

## Goal

Build a ground-truth dataset and evaluation framework to determine the best OCR pipeline for historical Italian newspaper scans before committing to any approach.

## 1.1 Ground Truth Dataset

### Sampling Strategy

Select ~100 pages spanning the full archive range, stratified by era to capture varying print quality:

- 1880–1899: ~20 pages (earliest, worst quality, letterpress)
- 1900–1919: ~20 pages (early 20th century, slightly better)
- 1920–1935: ~20 pages (fascist era, varying quality)
- 1935–1945: ~20 pages (war era, often degraded paper/ink)
- Mixed difficulty: ~20 pages specifically chosen for hard cases (faded ink, complex layouts, overlapping columns, ads mixed with articles, small print)

### What Each Ground Truth Page Needs

For each sampled page, produce:

1. **Raw transcription**: every text block transcribed exactly as printed
2. **Semantic units**: each page segmented into information units (articles, ads, obituaries, etc.) with a label
3. **Reading order**: the correct sequence of semantic units as a human would read them
4. **Cross-page links**: if an article continues from/to another page, note it

### Ground Truth Creation Process

1. Run a strong VLM (Claude, GPT-4o) on each page to produce a draft transcription
2. Manually verify and correct against the original image
3. Structure the output as a JSON file per page:

```json
{
  "page_id": "1923-03-15_p3",
  "date": "1923-03-15",
  "page_number": 3,
  "units": [
    {
      "id": "u1",
      "type": "article",
      "headline": "Il Duce visita Roma",
      "text": "...",
      "bounding_box": {"x1": 0, "y1": 0, "x2": 500, "y2": 300},
      "continues_from": null,
      "continues_to": "p4_u3"
    }
  ],
  "reading_order": ["u1", "u2", "u3"]
}
```

### Also Create Issue-Level Ground Truth

For ~20 complete daily issues (all pages from one day), produce the full issue-level ground truth — all articles stitched across pages in correct order. This tests the whole-issue LLM approach.

Select issues from different eras: ~5 per era bucket.

## 1.2 Evaluation Metrics

### Two separate metrics, measured independently:

**Metric 1: Transcription Quality (per text block)**
- CER (Character Error Rate): edit distance at character level, normalized by reference length
- WER (Word Error Rate): edit distance at word level
- Measured per semantic unit in isolation (ignoring reading order)
- Report mean, median, p95, and breakdown by era bucket

**Metric 2: Reading Order Quality (per page and per issue)**
- Kendall's tau rank correlation between predicted and ground-truth unit sequences
- Normalized edit distance on the sequence of unit IDs
- Bonus: article segmentation accuracy — did the pipeline identify the same semantic units as ground truth? (Measured as F1 over unit boundaries)

### Reporting

For each pipeline configuration tested, produce a score card:
- Overall CER/WER
- CER/WER by era bucket
- Reading order correlation (page-level)
- Reading order correlation (issue-level, for the 20 full issues)
- Segmentation F1
- Throughput (pages/second on reference hardware)

## 1.3 Pipeline Configurations to Test

### OCR Models (first pass — the "rough draft" VLM)

Research and benchmark these categories:

1. **Dedicated OCR VLMs**: GOT-OCR2, Nougat, Donut, TrOCR, etc.
2. **General VLMs with OCR capability**: Qwen2.5-VL (various sizes), InternVL2.5, Florence-2, Phi-3.5-vision, Llama-3.2-vision
3. **Traditional OCR + enhancement**: Tesseract/EasyOCR as absolute baseline (expect poor results)
4. **Emerging models**: check HuggingFace and papers from late 2025 / early 2026 for any new releases

For each, test multiple sizes if available (e.g., Qwen2.5-VL 3B vs 7B vs 72B).

### Ensemble Configurations

Test these pipeline variants:

- **VLM only**: single VLM does everything (OCR + segmentation + reading order)
- **YOLO + VLM**: DocLayout-YOLO for layout detection / bounding boxes, then VLM for OCR on each segment
- **VLM + LLM cleanup**: VLM for raw OCR, then a text-only or vision LLM for cleanup and reordering
- **YOLO + VLM + LLM cleanup**: full ensemble
- **Whole-issue VLM**: feed all pages of a day's issue to a long-context VLM at once

### LLM Cleanup Models (second pass)

- Qwen2.5 (various sizes, text-only for cost if OCR provides good enough text)
- Qwen2.5-VL (if feeding images in the cleanup pass)
- Llama-3 variants
- Mistral/Mixtral variants
- Whatever is best available at time of implementation

### Key Variables to Test

- Image preprocessing: does deskewing/contrast enhancement help?
- Image resolution: what resolution to feed the VLM?
- Prompt engineering: how much does the OCR/cleanup prompt matter?
- Single page vs. whole issue processing

## 1.4 Implementation

### Directory Structure

```
eval/
  ground_truth/
    pages/           # individual page ground truth JSONs
    issues/          # full issue ground truth JSONs
  predictions/       # pipeline outputs in same format as ground truth
  metrics/           # computed scores per pipeline
  scripts/
    create_ground_truth.py   # helper to generate drafts via API
    evaluate.py              # compute all metrics
    compare.py               # generate comparison tables/charts
    sample_pages.py          # select stratified sample from archive
```

### Steps

1. Write `sample_pages.py` to select the ~100 stratified pages from the scraped data
2. Write `create_ground_truth.py` to generate draft transcriptions using a strong API model
3. Manually review and correct all ground truth files
4. Write `evaluate.py` implementing CER, WER, reading order metrics, segmentation F1
5. For each pipeline configuration: run it on the 100 pages, save predictions, compute metrics
6. Write `compare.py` to produce comparison tables and decide on the winning pipeline

### Definition of Done

- Ground truth dataset complete for ~100 pages + ~20 full issues
- At least 5 pipeline configurations benchmarked
- Clear winner identified with evidence
- Score card documenting the chosen pipeline's performance per era
