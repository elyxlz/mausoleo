from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class WholeIssueVlm(BaseOperatorConfig):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 16384
    temperature: float = 0.0
    gpu_fraction: float = 1.0
    max_model_len: int = 65536
    max_images_per_prompt: int = 12
    backend: tp.Literal["vllm", "transformers"] = "transformers"


@register_operator(WholeIssueVlm, operation=OperatorType.MAP_BATCHES)
class WholeIssueVlmOperator(StatefulOperator[WholeIssueVlm]):
    def __init__(self, config: WholeIssueVlm) -> None:
        self.config = config
        if config.mock:
            return
        if config.backend == "vllm":
            self._init_vllm()
        else:
            self._init_transformers()

    def _init_vllm(self) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.config.model,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=self.config.max_model_len,
            limit_mm_per_prompt={"image": self.config.max_images_per_prompt},
        )
        self.sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=self.config.max_tokens)

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        self.hf_model = AutoModelForVision2Seq.from_pretrained(
            self.config.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)
        if self.config.backend == "vllm":
            return self._vllm_call(batch)
        return self._transformers_call(batch)

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        page_count = len(images_b64.split("|"))
        articles = [
            {
                "unit_type": "article",
                "headline": f"Articolo pagina {i + 1}",
                "paragraphs": [{"text": f"Testo dell'articolo dalla pagina {i + 1}. Contenuto di esempio."}],
                "page_span": [i + 1],
            }
            for i in range(page_count)
        ]
        result = dict(batch)
        result["result_json"] = [json.dumps({"articles": articles})]
        return result

    def _vllm_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        from PIL import Image
        from transformers import AutoProcessor

        if not hasattr(self, "_vllm_processor"):
            self._vllm_processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]
        pil_images = [Image.open(io.BytesIO(img)) for img in raw_images]

        content: list[dict[str, tp.Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": self.config.prompt})
        messages = [{"role": "user", "content": content}]
        prompt_text: str = self._vllm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = self.llm.generate([{"prompt": prompt_text, "multi_modal_data": {"image": pil_images}}], self.sampling_params)
        result = dict(batch)
        result["result_json"] = [outputs[0].outputs[0].text.strip()]
        return result

    def _transformers_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        import torch
        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]
        pil_images = [Image.open(io.BytesIO(img)) for img in raw_images]

        content: list[dict[str, tp.Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": self.config.prompt})
        messages: list[dict[str, tp.Any]] = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil_images, return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.config.max_tokens)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        result_text: str = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        result = dict(batch)
        result["result_json"] = [result_text]
        return result
