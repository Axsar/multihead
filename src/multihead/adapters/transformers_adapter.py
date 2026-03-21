"""HuggingFace Transformers adapter for direct GPU inference.

This is the adapter for the existing setup:
- AutoModelForCausalLM / AutoModelForVision2Seq
- AutoProcessor / AutoTokenizer
- BitsAndBytesConfig for 4bit/8bit quantization
- Direct model.generate() calls (no inference server)
"""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..models import HeadManifest
from .base import HeadAdapter

logger = logging.getLogger(__name__)


class TransformersAdapter(HeadAdapter):
    """Direct HuggingFace Transformers inference on local GPU.

    Loads models via AutoModel + AutoProcessor, supports quantization
    via BitsAndBytesConfig. One request at a time (no batching).
    Model stays in GPU memory until explicitly unloaded.
    """

    def __init__(self, manifest: HeadManifest) -> None:
        super().__init__(manifest)
        self.model_name = manifest.model
        self.quantization = manifest.quantization
        self._model = None
        self._processor = None
        self._tokenizer = None

    async def load(self) -> None:
        """Load model + processor into GPU memory."""
        import torch

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        # Apply quantization if specified
        if self.quantization:
            from transformers import BitsAndBytesConfig

            if self.quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        # Determine model class based on head kind
        if self.manifest.kind == "vlm":
            from transformers import AutoModelForVision2Seq, AutoProcessor

            logger.info("Loading VLM: %s (quantization=%s)", self.model_name, self.quantization)
            self._model = AutoModelForVision2Seq.from_pretrained(self.model_name, **model_kwargs)
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_name)
            except Exception:
                logger.error("Processor load failed for %s, cleaning up model", self.model_name)
                await self.unload()
                raise
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading LLM: %s (quantization=%s)", self.model_name, self.quantization)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                logger.error("Tokenizer load failed for %s, cleaning up model", self.model_name)
                await self.unload()
                raise

        logger.info("Model loaded: %s", self.model_name)

    async def unload(self) -> None:
        """Unload model from GPU, free VRAM."""
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Model unloaded: %s", self.model_name)

    @staticmethod
    def _decode_images(images: list) -> list:
        """Decode base64 strings to PIL Images. Pass through PIL Images as-is."""
        import base64
        import io
        from PIL import Image

        decoded = []
        for img in images:
            if isinstance(img, str):
                # base64 string — strip data URI prefix if present
                if "," in img and img.index(",") < 100:
                    img = img.split(",", 1)[1]
                raw = base64.b64decode(img)
                decoded.append(Image.open(io.BytesIO(raw)))
            else:
                decoded.append(img)  # Already a PIL Image
        return decoded

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate text via model.generate()."""
        import torch

        if self._model is None:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        max_new_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)
        thinking = kwargs.get("thinking", True)
        skip_template = kwargs.get("_skip_chat_template", False)

        # Use chat template when available (required for Qwen3, etc.)
        tokenizer = self._tokenizer or self._processor
        if not skip_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if not thinking:
                template_kwargs["enable_thinking"] = False
            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Fallback if template doesn't support enable_thinking
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

        # Use tokenizer for LLM, processor for VLM
        if self._tokenizer is not None:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            input_len = inputs["input_ids"].shape[1]
        elif self._processor is not None:
            # For VLM: images should be passed via kwargs["images"]
            # Accept base64 strings or PIL Images
            images = kwargs.get("images")
            if images:
                images = self._decode_images(images)
                inputs = self._processor(text=prompt, images=images, return_tensors="pt").to(self._model.device)
            else:
                inputs = self._processor(text=prompt, return_tensors="pt").to(self._model.device)
            input_len = inputs["input_ids"].shape[1]
        else:
            raise RuntimeError("No tokenizer or processor available")

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        new_tokens = output_ids[0][input_len:]
        decoder = self._tokenizer or self._processor
        text = decoder.decode(new_tokens, skip_special_tokens=True)

        return {
            "text": text,
            "tokens_in": input_len,
            "tokens_out": len(new_tokens),
        }

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens using TextIteratorStreamer in a background thread."""
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        if self._model is None:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        max_new_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)

        tokenizer = self._tokenizer or self._processor
        if tokenizer is None:
            raise RuntimeError("No tokenizer or processor available")

        inputs = tokenizer(prompt, return_tensors="pt").to(self._model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(
            target=lambda: self._model.generate(**inputs, **gen_kwargs),
            daemon=True,
        )
        thread.start()

        loop = asyncio.get_event_loop()
        for token in streamer:
            if token:
                yield token
            await asyncio.sleep(0)  # Yield control to event loop

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat-style inference. Applies chat template if available.

        Note: generate() already applies chat template for single prompts.
        This method handles multi-turn messages directly.
        """
        tokenizer = self._tokenizer or self._processor
        thinking = kwargs.get("thinking", True)

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if not thinking:
                template_kwargs["enable_thinking"] = False
            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        # Skip chat template in generate() since we already applied it
        kwargs["_skip_chat_template"] = True
        return await self.generate(prompt, **kwargs)

    async def healthcheck(self) -> bool:
        return self._model is not None

    async def sleep(self, level: int = 1) -> None:
        """For transformers: sleep = unload (no intermediate state)."""
        await self.unload()

    async def wake(self) -> None:
        await self.load()
