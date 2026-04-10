from __future__ import annotations

import dataclasses as dc
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class LlmCleanup(BaseOperatorConfig):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 8192
    temperature: float = 0.0
    gpu_fraction: float = 1.0
    max_model_len: int = 32768
    backend: tp.Literal["vllm", "transformers"] = "transformers"
    load_in_4bit: bool = False


@register_operator(LlmCleanup, operation=OperatorType.MAP_BATCHES)
class LlmCleanupOperator(StatefulOperator[LlmCleanup]):
    def __init__(self, config: LlmCleanup) -> None:
        self.config = config
        if config.mock:
            return
        if config.backend == "vllm":
            self._init_vllm()
        else:
            self._init_transformers()

    def _init_vllm(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        self.llm = LLM(
            model=self.config.model,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=self.config.max_model_len,
        )
        self.sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=self.config.max_tokens)

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        load_kwargs: dict[str, tp.Any] = {"device_map": "auto", "trust_remote_code": True}
        if self.config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            load_kwargs["torch_dtype"] = torch.float16
        self.hf_model = AutoModelForCausalLM.from_pretrained(self.config.model, **load_kwargs)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)
        if self.config.backend == "vllm":
            return self._vllm_call(batch)
        return self._transformers_call(batch)

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        page_texts: list[str] = json.loads(batch["page_texts"][0])
        articles = [
            {
                "unit_type": "article",
                "headline": f"Articolo {i + 1}",
                "paragraphs": [{"text": text}],
                "page_span": [i + 1],
            }
            for i, text in enumerate(page_texts)
        ]
        result = dict(batch)
        result["result_json"] = [json.dumps({"articles": articles})]
        return result

    def _build_prompt(self, batch: dict[str, tp.Any]) -> str:
        page_texts: list[str] = json.loads(batch["page_texts"][0])
        page_count = int(batch["page_count"][0])
        combined = "\n\n".join(f"--- Page {i + 1} ---\n{text}" for i, text in enumerate(page_texts))
        return self.config.prompt.format(text=combined, page_count=page_count)

    def _vllm_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        user_content = self._build_prompt(batch)
        messages: list[dict[str, str]] = [{"role": "user", "content": user_content}]
        formatted: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[assignment]
        outputs = self.llm.generate([formatted], self.sampling_params)
        result_text = outputs[0].outputs[0].text.strip()
        result = dict(batch)
        result["result_json"] = [result_text]
        return result

    def _transformers_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        import torch

        user_content = self._build_prompt(batch)
        messages: list[dict[str, str]] = [{"role": "user", "content": user_content}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(inputs, max_new_tokens=self.config.max_tokens)
        generated = output_ids[:, inputs.shape[1] :]
        result_text: str = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        result = dict(batch)
        result["result_json"] = [result_text]
        return result
