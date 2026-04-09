# Phase 2: OCR Pipeline (Ray Data)

## Goal

Build a production OCR pipeline using the winning configuration from Phase 1. Process the entire Il Messaggero archive (1880–~1945) at scale using Ray Data. Output: structured text for every daily issue, ready for hierarchical summarization.

## 2.1 Pipeline Architecture

### Processing Unit: The Daily Issue

The pipeline processes one day's newspaper issue as a single unit:

1. Load all page images for a given date (typically 4–10 pages)
2. Run OCR VLM on each page (first pass — rough draft)
3. Feed all pages + rough OCR to the cleanup LLM as a single call (second pass)
4. LLM outputs structured, ordered text with semantic unit boundaries
5. Save structured output to disk

If the eval in Phase 1 shows that a different approach wins (e.g., YOLO + VLM without a second LLM pass, or single-pass VLM on the whole issue), adapt accordingly. The pipeline shape follows the eval results.

### Output Format Per Issue

```json
{
  "date": "1923-03-15",
  "source": "il_messaggero",
  "page_count": 6,
  "units": [
    {
      "id": "1923-03-15_a01",
      "type": "article",
      "headline": "Il Duce visita Roma",
      "paragraphs": [
        {"id": "1923-03-15_a01_p01", "text": "..."},
        {"id": "1923-03-15_a01_p02", "text": "..."}
      ],
      "page_span": [1, 2],
      "position_in_issue": 1
    },
    {
      "id": "1923-03-15_a02",
      "type": "obituary",
      "headline": null,
      "paragraphs": [...],
      "page_span": [3],
      "position_in_issue": 2
    }
  ]
}
```

### ID Scheme

Deterministic, human-readable IDs:
- Issue: `{date}` e.g. `1923-03-15`
- Article: `{date}_a{nn}` e.g. `1923-03-15_a01`
- Paragraph: `{date}_a{nn}_p{nn}` e.g. `1923-03-15_a01_p01`

## 2.2 Ray Data Pipeline

### Why Ray Data

- Handles data loading, batching, and GPU scheduling
- Scales from single GPU to multi-GPU/multi-node
- Built-in fault tolerance and checkpointing
- Can resume from where it left off if interrupted

### Pipeline Stages

```
Ray Data pipeline:
  read_images(date_range)          # yields (date, [page_images])
  -> map_batches(ocr_first_pass)   # VLM inference on individual pages, GPU
  -> map(assemble_issue)           # group pages into issue + OCR text
  -> map_batches(llm_cleanup)      # LLM cleanup/reorder per issue, GPU
  -> map(validate_output)          # schema validation
  -> write_json(output_dir)        # save structured JSON per issue
```

### Key Design Decisions

**Batching**: The OCR first pass can batch across pages from different issues (just image -> text). The LLM cleanup must process one full issue at a time (needs all pages together).

**GPU allocation**: If running on multiple GPUs, OCR and LLM cleanup can run on separate GPUs in parallel (pipeline parallelism). Ray Data handles this via resource annotations.

**Checkpointing**: Save progress per-date so the pipeline can resume. Before processing a date, check if output JSON already exists. Skip if it does.

**Error handling**: If a date fails (corrupted images, LLM timeout, etc.), log it and continue. Don't block the whole pipeline. Collect failed dates for retry.

### vLLM Integration

Use vLLM as the inference backend for both the OCR VLM and the cleanup LLM:
- Run vLLM as a server (OpenAI-compatible API)
- Ray Data workers send requests to the vLLM server
- This decouples data orchestration from model serving
- Can run multiple vLLM instances for different models (OCR vs cleanup)

Alternative: use vLLM's offline batched inference directly within Ray Data map_batches. Test both approaches for throughput.

## 2.3 Prompt Engineering

### OCR First Pass Prompt

Design a prompt that tells the VLM to:
- Transcribe all visible text on the newspaper page
- Preserve the spatial layout (columns, headlines, body text)
- Mark uncertain characters or words
- Output in a structured format (not just raw text)

The exact prompt will be refined during the eval phase.

### LLM Cleanup Prompt

Design a prompt that tells the LLM to:
- Take the raw OCR output + page images for the full issue
- Correct OCR errors using context and visual confirmation
- Segment into semantic units (articles, ads, obituaries, etc.)
- Determine reading order across the full issue
- Stitch articles that span multiple pages
- Output in the structured JSON format defined above
- Label each unit with a type

This is the most important prompt in the system. It should be heavily tested and refined.

## 2.4 Image Preprocessing

Test during eval phase whether these help:
- Deskewing (straighten rotated scans)
- Contrast normalization
- Binarization (convert to black and white)
- Denoising
- Resolution scaling (what DPI to feed the VLM)

If preprocessing helps, add it as a Ray Data stage before OCR.

## 2.5 Output Storage

### Intermediate Output

Save one JSON file per date to disk:
```
ocr_output/
  1880/
    1880-01-01.json
    1880-01-02.json
    ...
  1881/
    ...
```

This serves as the input to Phase 3 (hierarchical index building) and as a cache so the OCR pipeline doesn't need to be re-run.

### Quality Spot-Check

After running the full pipeline:
- Randomly sample ~50 issues across eras
- Manually review OCR quality
- Run the eval metrics from Phase 1 on any ground truth pages that went through the pipeline
- Document quality observations per era

## 2.6 Implementation Steps

1. Set up vLLM serving for the chosen OCR model and cleanup LLM
2. Write the Ray Data pipeline skeleton (read images → write JSON)
3. Implement OCR first pass stage (calls vLLM)
4. Implement issue assembly stage (groups pages, attaches OCR)
5. Implement LLM cleanup stage (calls vLLM with full issue)
6. Implement output validation (check JSON schema, flag anomalies)
7. Add checkpointing (skip already-processed dates)
8. Add error handling and retry logic
9. Run on a small subset (~1 month) and validate quality
10. Run on full archive
11. Quality spot-check

### Definition of Done

- Full archive (1880–~1945) processed into structured JSON
- Every daily issue has a well-formed output file
- Failed dates documented and retried where possible
- Quality spot-check shows acceptable results across all eras
