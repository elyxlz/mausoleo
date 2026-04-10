from __future__ import annotations

import dataclasses as dc
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class LlmPostCorrect(BaseOperatorConfig):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 8192
    gpu_fraction: float = 1.0
    max_model_len: int = 32768
    backend: tp.Literal["vllm", "transformers"] = "vllm"


@register_operator(LlmPostCorrect, operation=OperatorType.MAP_BATCHES)
class LlmPostCorrectOperator(StatefulOperator[LlmPostCorrect]):
    def __init__(self, config: LlmPostCorrect) -> None:
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
        self.llm = LLM(model=self.config.model, trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=self.config.max_model_len)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=self.config.max_tokens)

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(self.config.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return batch

        page_texts: list[str] = json.loads(batch["page_texts"][0])
        corrected: list[str] = []
        for text in page_texts:
            user_content = self.config.prompt.format(text=text)
            messages: list[dict[str, str]] = [{"role": "user", "content": user_content}]

            if self.config.backend == "vllm":
                formatted: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[assignment]
                outputs = self.llm.generate([formatted], self.sampling_params)
                corrected.append(outputs[0].outputs[0].text.strip())
            else:
                import torch

                inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
                inputs = inputs.to(self.hf_model.device)
                with torch.no_grad():
                    output_ids = self.hf_model.generate(inputs, max_new_tokens=self.config.max_tokens)
                generated = output_ids[:, inputs.shape[1]:]
                corrected.append(self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())

        result = dict(batch)
        result["page_texts"] = [json.dumps(corrected)]
        return result
