from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator

ModelType = tp.Literal["default", "florence", "got_ocr", "minicpm", "phi3", "internvl", "hunyuan"]


def _detect_model_type(model_name: str) -> ModelType:
    lower = model_name.lower()
    if "florence" in lower:
        return "florence"
    if "got-ocr" in lower or "got_ocr" in lower:
        return "got_ocr"
    if "minicpm" in lower:
        return "minicpm"
    if "phi-3" in lower or "phi3" in lower:
        return "phi3"
    if "internvl" in lower:
        return "internvl"
    if "hunyuan" in lower:
        return "hunyuan"
    return "default"


@dc.dataclass(frozen=True, kw_only=True)
class VlmOcr(BaseOperatorConfig):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    gpu_fraction: float = 1.0
    max_model_len: int = 32768
    backend: tp.Literal["vllm", "transformers"] = "transformers"
    load_in_4bit: bool = False


@register_operator(VlmOcr, operation=OperatorType.MAP_BATCHES)
class VlmOcrOperator(StatefulOperator[VlmOcr]):
    def __init__(self, config: VlmOcr) -> None:
        self.config = config
        self.model_type: ModelType = _detect_model_type(config.model)
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
            limit_mm_per_prompt={"image": 1},
        )
        self.sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=self.config.max_tokens)

    def _init_transformers(self) -> None:
        if self.model_type == "florence":
            self._init_florence()
        elif self.model_type == "got_ocr":
            self._init_got_ocr()
        elif self.model_type == "phi3":
            self._init_phi3()
        elif self.model_type == "hunyuan":
            self._init_hunyuan()
        else:
            self._init_transformers_generic()

    def _init_transformers_generic(self) -> None:
        import torch
        from transformers import AutoModel, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        load_kwargs: dict[str, tp.Any] = {"device_map": "auto", "trust_remote_code": True}
        if self.config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        try:
            self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        except Exception:
            self.processor = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)

        for auto_cls in [AutoModelForVision2Seq, AutoModel]:
            try:
                self.hf_model = auto_cls.from_pretrained(self.config.model, **load_kwargs)
                break
            except (ValueError, ImportError):
                continue
        else:
            from transformers import AutoModelForCausalLM

            self.hf_model = AutoModelForCausalLM.from_pretrained(self.config.model, **load_kwargs)

    def _init_florence(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model, trust_remote_code=True, torch_dtype=torch.float32
        ).to("cuda")

    def _init_got_ocr(self) -> None:
        import torch
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)

        try:
            from transformers import AutoModelForImageTextToText

            self.hf_model = AutoModelForImageTextToText.from_pretrained(
                self.config.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        except ImportError:
            from transformers import AutoModel

            self.hf_model = AutoModel.from_pretrained(
                self.config.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
            )

    def _init_phi3(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        load_kwargs: dict[str, tp.Any] = {"device_map": "auto", "trust_remote_code": True, "_attn_implementation": "eager"}
        if self.config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True, num_crops=16)
        self.hf_model = AutoModelForCausalLM.from_pretrained(self.config.model, **load_kwargs)

    def _init_hunyuan(self) -> None:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)

        load_kwargs: dict[str, tp.Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if self.config.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        for auto_cls in [AutoModelForVision2Seq, AutoModel, AutoModelForCausalLM]:
            try:
                self.hf_model = auto_cls.from_pretrained(self.config.model, **load_kwargs)
                break
            except (ValueError, ImportError):
                continue

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

        outputs = self.llm.generate(prompts, self.sampling_params)
        page_texts = [out.outputs[0].text for out in outputs]

        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _transformers_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        dispatch: dict[ModelType, tp.Callable[[tp.Any], str]] = {
            "florence": self._florence_call,
            "got_ocr": self._got_ocr_call,
            "minicpm": self._minicpm_call,
            "phi3": self._phi3_call,
            "internvl": self._internvl_call,
            "hunyuan": self._hunyuan_call,
            "default": self._generate_api_call,
        }
        call_fn = dispatch.get(self.model_type, self._generate_api_call)

        page_texts: list[str] = []
        for img_bytes in raw_images:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            page_texts.append(call_fn(pil_img))

        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _generate_api_call(self, pil_img: tp.Any) -> str:
        import torch

        messages: list[dict[str, tp.Any]] = [
            {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": self.config.prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.config.max_tokens)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]  # type: ignore[no-any-return]

    def _hunyuan_call(self, pil_img: tp.Any) -> str:
        import torch

        messages: list[dict[str, tp.Any]] = [
            {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": self.config.prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.config.max_tokens, do_sample=False)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]  # type: ignore[no-any-return]

    def _florence_call(self, pil_img: tp.Any) -> str:
        import torch

        task_prompt = "<OCR>"
        inputs = self.processor(text=task_prompt, images=pil_img, return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.config.max_tokens, num_beams=3)
        generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=pil_img.size)
        return parsed.get(task_prompt, generated_text)  # type: ignore[no-any-return]

    def _got_ocr_call(self, pil_img: tp.Any) -> str:
        import torch

        inputs = self.processor(pil_img, return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=self.config.max_tokens,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated[0], skip_special_tokens=True)  # type: ignore[no-any-return]

    def _minicpm_call(self, pil_img: tp.Any) -> str:
        msgs = [{"role": "user", "content": [pil_img, self.config.prompt]}]
        response = self.hf_model.chat(image=None, msgs=msgs, tokenizer=self.processor.tokenizer)
        if isinstance(response, tuple):
            return response[0]  # type: ignore[no-any-return]
        return response  # type: ignore[no-any-return]

    def _phi3_call(self, pil_img: tp.Any) -> str:
        import torch

        messages = [{"role": "user", "content": f"<|image_1|>\n{self.config.prompt}"}]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [pil_img], return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.config.max_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]  # type: ignore[no-any-return]

    def _internvl_call(self, pil_img: tp.Any) -> str:
        pixel_values = self._load_internvl_image(pil_img)
        question = f"<image>\n{self.config.prompt}"
        response: str = self.hf_model.chat(self.processor, pixel_values, question, generation_config={"max_new_tokens": self.config.max_tokens})
        return response

    def _load_internvl_image(self, pil_img: tp.Any) -> tp.Any:
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        input_size = 448

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = transform(pil_img).unsqueeze(0).to(torch.bfloat16).cuda()
        return pixel_values

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
