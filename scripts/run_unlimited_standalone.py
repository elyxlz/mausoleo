from __future__ import annotations

import argparse
import json
import pathlib as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baidu/Unlimited-OCR")
    parser.add_argument("--prompt", default="<image>Multi page parsing.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=1024)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()

    out_dir = pl.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    returned = model.infer_multi(
        tokenizer,
        prompt=args.prompt,
        image_files=list(args.images),
        output_path=str(out_dir),
        image_size=args.image_size,
        max_length=args.max_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
        save_results=True,
    )
    result = {"returned_text": returned if isinstance(returned, str) else None, "output_dir": str(out_dir)}
    (out_dir / "runner_result.json").write_text(json.dumps(result))
    print(json.dumps({"saved_files": sorted(str(p) for p in out_dir.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
