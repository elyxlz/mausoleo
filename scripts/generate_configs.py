from __future__ import annotations

import pathlib as pl

CONFIGS_DIR = pl.Path("configs/ocr")


def w(name: str, imports: str, operators: str) -> None:
    path = CONFIGS_DIR / f"{name}.py"
    path.write_text(f"from mausoleo.ocr import prompts\nfrom mausoleo.ocr.config import OcrPipelineConfig\n{imports}\n\nconfig = OcrPipelineConfig(\n    name=\"{name}\",\n    operators=[\n{operators}\n    ],\n)\n")
    print(f"  {name}")


def vlm(model: str, prompt: str, backend: str = "vllm", max_tokens: int = 8192, max_model_len: int = 16384) -> str:
    return f'        VlmOcr(model="{model}", prompt=prompts.{prompt}, backend="{backend}", max_tokens={max_tokens}, max_model_len={max_model_len}),'


def cleanup(model: str, backend: str = "vllm") -> str:
    return f'        LlmCleanup(model="{model}", prompt=prompts.LLM_CLEANUP_V2, backend="{backend}", max_tokens=8192, max_model_len=16384),'


def postcorrect(model: str, backend: str = "vllm") -> str:
    return f'        LlmPostCorrect(model="{model}", prompt=prompts.LLM_POST_CORRECTION, backend="{backend}", max_tokens=8192, max_model_len=16384),'


def colsplit(n: int, overlap: float = 0.03) -> str:
    return f"        ColumnSplit(num_columns={n}, overlap_pct={overlap}),"


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for f in CONFIGS_DIR.glob("*.py"):
        f.unlink()

    VLMS = {
        "qwen25_7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "vllm"),
        "qwen25_3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "vllm"),
        "qwen3_8b": ("Qwen/Qwen3-VL-8B-Instruct", "transformers"),
        "qwen3_4b": ("Qwen/Qwen3-VL-4B-Instruct", "transformers"),
        "qwen3_2b": ("Qwen/Qwen3-VL-2B-Instruct", "transformers"),
        "gemma3_9b": ("google/gemma-3-9b-vision", "transformers"),
        "gemma3_2b": ("google/gemma-3-2b-vision", "transformers"),
        "internvl3_6b": ("OpenGVLab/InternVL3-6B", "transformers"),
        "internvl3_2b": ("OpenGVLab/InternVL3-2B", "transformers"),
    }

    LLMS = {
        "qwen25_3b": ("Qwen/Qwen2.5-3B-Instruct", "vllm"),
        "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "vllm"),
    }

    imp_base = "from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr"
    imp_cleanup = "from mausoleo.ocr.operators import LlmCleanup, ParseIssue, VlmOcr"
    imp_postcorrect = "from mausoleo.ocr.operators import LlmPostCorrect, MergePages, ParseIssue, VlmOcr"
    imp_col = "from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr"
    imp_col_cleanup = "from mausoleo.ocr.operators import ColumnSplit, LlmCleanup, ParseIssue, VlmOcr"
    imp_preproc = "from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr"
    imp_preproc_col = "from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, Preprocess, VlmOcr"
    imp_yolo = "from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop"

    print("=== 1. Full-page structured (every VLM) ===")
    for vtag, (vmodel, vbackend) in VLMS.items():
        for pv, prompt, maxtok in [("v1", "VLM_OCR_STRUCTURED", 4096), ("v2", "VLM_OCR_STRUCTURED_V2", 8192)]:
            name = f"{vtag}_{pv}_structured"
            w(name, imp_base, f"{vlm(vmodel, prompt, vbackend, maxtok)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 2. Full-page raw + cleanup (main VLMs) ===")
    for vtag in ["qwen25_7b", "qwen25_3b", "qwen3_8b", "gemma3_9b"]:
        vmodel, vbackend = VLMS[vtag]
        lmodel, lbackend = LLMS["qwen25_3b"]
        name = f"{vtag}_v2_raw_cleanup"
        w(name, imp_cleanup, f"{vlm(vmodel, 'VLM_OCR_RAW_V2', vbackend)}\n{cleanup(lmodel, lbackend)}\n        ParseIssue(),")

    print("\n=== 3. Full-page structured + post-correction ===")
    for vtag in ["qwen25_7b", "qwen3_8b"]:
        vmodel, vbackend = VLMS[vtag]
        lmodel, lbackend = LLMS["qwen25_3b"]
        name = f"{vtag}_v2_postcorrect"
        w(name, imp_postcorrect, f"{vlm(vmodel, 'VLM_OCR_STRUCTURED_V2', vbackend)}\n{postcorrect(lmodel, lbackend)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 4. Preprocessing + full-page ===")
    for vtag in ["qwen25_7b"]:
        vmodel, vbackend = VLMS[vtag]
        name = f"preproc_{vtag}_v2_structured"
        w(name, imp_preproc, f"        Preprocess(grayscale=True, max_dimension=2000),\n{vlm(vmodel, 'VLM_OCR_STRUCTURED_V2', vbackend)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 5. Column split + structured (main VLMs, different col counts) ===")
    for ncols in [3, 4, 5]:
        for vtag in ["qwen25_7b", "qwen25_3b", "qwen3_8b", "gemma3_9b"]:
            vmodel, vbackend = VLMS[vtag]
            name = f"col{ncols}_{vtag}_v2_structured"
            w(name, imp_col, f"{colsplit(ncols)}\n{vlm(vmodel, 'VLM_OCR_STRUCTURED_V2', vbackend)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 6. Column split + raw column OCR + cleanup ===")
    for ncols in [4]:
        for vtag in ["qwen25_7b"]:
            vmodel, vbackend = VLMS[vtag]
            lmodel, lbackend = LLMS["qwen25_3b"]
            name = f"col{ncols}_{vtag}_v2_column_cleanup"
            w(name, imp_col_cleanup, f"{colsplit(ncols)}\n{vlm(vmodel, 'VLM_OCR_COLUMN', vbackend)}\n{cleanup(lmodel, lbackend)}\n        ParseIssue(),")

    print("\n=== 7. Column split + raw (no cleanup) ===")
    for ncols in [4]:
        for vtag in ["qwen25_7b"]:
            vmodel, vbackend = VLMS[vtag]
            name = f"col{ncols}_{vtag}_v2_raw"
            w(name, imp_col, f"{colsplit(ncols)}\n{vlm(vmodel, 'VLM_OCR_RAW_V2', vbackend)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 8. Column split no overlap ===")
    for ncols in [4]:
        for vtag in ["qwen25_7b"]:
            vmodel, vbackend = VLMS[vtag]
            name = f"col{ncols}_nolap_{vtag}_v2_structured"
            w(name, imp_col, f"{colsplit(ncols, 0.0)}\n{vlm(vmodel, 'VLM_OCR_STRUCTURED_V2', vbackend)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 9. Preprocess + column split ===")
    for ncols in [4]:
        for vtag in ["qwen25_7b"]:
            vmodel, vbackend = VLMS[vtag]
            name = f"preproc_col{ncols}_{vtag}_v2_structured"
            w(name, imp_preproc_col, f"        Preprocess(grayscale=True, max_dimension=2000),\n{colsplit(ncols)}\n{vlm(vmodel, 'VLM_OCR_STRUCTURED_V2', vbackend)}\n        MergePages(),\n        ParseIssue(),")

    print("\n=== 10. YOLO crop + VLM ===")
    for vtag in ["qwen25_7b"]:
        vmodel, vbackend = VLMS[vtag]
        name = f"yolo_{vtag}_v2_structured"
        w(name, imp_yolo, f"        YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, merge_vertical_gap=80, merge_horizontal_overlap=0.7),\n{vlm(vmodel, 'VLM_OCR_STRUCTURED_V2', vbackend)}\n        MergePages(),\n        ParseIssue(),")

    configs = sorted(f.stem for f in CONFIGS_DIR.glob("*.py"))
    print(f"\n=== TOTAL: {len(configs)} configs ===")


if __name__ == "__main__":
    main()
