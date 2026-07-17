from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator

@dc.dataclass(frozen=True, kw_only=True)
class VlmOcr(BaseOperatorConfig):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    gpu_fraction: float = 1.0
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = True
    max_pixels: int | None = None
    max_model_len: int = 32768
    backend: tp.Literal["vllm", "transformers"] = "transformers"
    load_in_4bit: bool = False
    vllm_strict: bool = False


@register_operator(VlmOcr, operation=OperatorType.MAP_BATCHES)
class VlmOcrOperator(StatefulOperator[VlmOcr]):
    def __init__(self, config: VlmOcr) -> None:
        self.config = config
        if config.mock:
            return
        self._prime_cuda()
        if config.backend == "vllm":
            self._init_vllm()
        else:
            self._init_transformers()

    @staticmethod
    def _prime_cuda() -> None:
        import os
        import torch

        os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
        torch.backends.cudnn.enabled = False

        try:
            from transformers import image_transforms

            _orig_normalize = image_transforms.normalize

            def _patched_normalize(image: tp.Any, mean: tp.Any, std: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                import numpy as np

                m = np.asarray(mean)
                s = np.asarray(std)
                if m.ndim > 1:
                    m = m.flatten()[: image.shape[-1]] if image.ndim >= 3 else m.flatten()
                if s.ndim > 1:
                    s = s.flatten()[: image.shape[-1]] if image.ndim >= 3 else s.flatten()
                return _orig_normalize(image, m, s, *args, **kwargs)

            image_transforms.normalize = _patched_normalize
        except Exception:
            pass

    def _init_vllm(self) -> None:
        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, tp.Any] = dict(
            model=self.config.model,
            trust_remote_code=True,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=self.config.enforce_eager,
        )
        if self.config.max_pixels is not None:
            llm_kwargs["mm_processor_kwargs"] = {"max_pixels": self.config.max_pixels}
        if self.config.vllm_strict:
            llm_kwargs["dtype"] = "bfloat16"
            llm_kwargs["enable_prefix_caching"] = False
            llm_kwargs["seed"] = 0
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=self.config.max_tokens)

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        load_kwargs: dict[str, tp.Any] = {"device_map": "auto", "trust_remote_code": True}
        if self.config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        self.hf_model = AutoModelForImageTextToText.from_pretrained(self.config.model, **load_kwargs)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)
        if self.config.backend == "vllm":
            return self._vllm_call(batch)
        return self._transformers_call(batch)

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        page_count = len(images_b64.split("|"))
        page_texts = [f"Mock OCR output for page {i + 1}. Titolo principale dell'articolo." for i in range(page_count)]
        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _vllm_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        prompts: list[dict[str, tp.Any]] = []
        for img_bytes in raw_images:
            pil_img = Image.open(io.BytesIO(img_bytes))
            prompts.append({"prompt": self._format_prompt_vllm(pil_img), "multi_modal_data": {"image": pil_img}})

        outputs = self.llm.generate(prompts, self.sampling_params)  # type: ignore[arg-type]
        page_texts = [out.outputs[0].text for out in outputs]

        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _transformers_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        page_texts: list[str] = []
        for img_bytes in raw_images:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            page_texts.append(self._chat_template_call(pil_img))

        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _chat_template_call(self, pil_img: tp.Any) -> str:
        import torch

        messages: list[dict[str, tp.Any]] = [
            {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": self.config.prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.config.max_tokens, do_sample=False)  # type: ignore[attr-defined]
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]  # type: ignore[no-any-return]

    def _format_prompt_vllm(self, image: tp.Any) -> str:
        from transformers import AutoProcessor, AutoTokenizer

        if not hasattr(self, "_vllm_processor"):
            self._vllm_processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": self.config.prompt}]}]

        proc = self._vllm_processor
        if hasattr(proc, "chat_template") and proc.chat_template:
            return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[no-any-return]

        if hasattr(proc, "tokenizer") and hasattr(proc.tokenizer, "chat_template") and proc.tokenizer.chat_template:
            return proc.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[no-any-return]

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
            return tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                [{"role": "user", "content": f"<image>\n{self.config.prompt}"}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"<|user|>\n<image>\n{self.config.prompt}<|end|>\n<|assistant|>\n"
