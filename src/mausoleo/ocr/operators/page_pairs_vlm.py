from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class PagePairVlm(BaseOperatorConfig):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 12288
    temperature: float = 0.0
    gpu_fraction: float = 1.0
    max_model_len: int = 32768
    window_size: int = 2
    stride: int = 1


@register_operator(PagePairVlm, operation=OperatorType.MAP_BATCHES)
class PagePairVlmOperator(StatefulOperator[PagePairVlm]):
    def __init__(self, config: PagePairVlm) -> None:
        self.config = config
        if config.mock:
            return
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=config.model,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=config.max_model_len,
            limit_mm_per_prompt={"image": config.window_size},
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(temperature=config.temperature, max_tokens=config.max_tokens)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)

        from PIL import Image
        from transformers import AutoProcessor

        if not hasattr(self, "_vllm_processor"):
            self._vllm_processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]
        pil_images = [Image.open(io.BytesIO(img)) for img in raw_images]
        page_count = len(pil_images)

        windows: list[tuple[int, list[tp.Any]]] = []
        w = self.config.window_size
        s = self.config.stride
        i = 0
        while i < page_count:
            end = min(i + w, page_count)
            windows.append((i + 1, pil_images[i:end]))
            if end >= page_count:
                break
            i += s

        window_articles: list[dict[str, tp.Any]] = []
        window_page_lists: list[list[int]] = []

        prompts: list[dict[str, tp.Any]] = []
        for window_start, imgs in windows:
            content: list[dict[str, tp.Any]] = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": self.config.prompt})
            messages = [{"role": "user", "content": content}]
            prompt_text = self._vllm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append({"prompt": prompt_text, "multi_modal_data": {"image": imgs}})

        outputs = self.llm.generate(prompts, self.sampling_params)

        all_articles: list[dict[str, tp.Any]] = []
        for (window_start, imgs), output in zip(windows, outputs):
            window_end = window_start + len(imgs) - 1
            text = output.outputs[0].text.strip()
            articles = self._parse_articles(text)
            for art in articles:
                raw_span = art.get("page_span", [])
                if not isinstance(raw_span, list) or not raw_span:
                    continue
                mapped = [min(window_start + (p - 1), window_end) for p in raw_span if isinstance(p, int)]
                if not mapped:
                    continue
                mapped = sorted(set(mapped))
                art["page_span"] = mapped
                all_articles.append(art)

        result = dict(batch)
        result["result_json"] = [json.dumps({"articles": all_articles})]
        return result

    def _parse_articles(self, raw: str) -> list[dict[str, tp.Any]]:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data.get("articles", [])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            for closer in ['"}]}', '"}]', ']', '}']:
                try:
                    data = json.loads(text + closer)
                    if isinstance(data, dict):
                        return data.get("articles", [])
                except json.JSONDecodeError:
                    continue
        return []

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        result = dict(batch)
        result["result_json"] = [json.dumps({"articles": []})]
        return result
