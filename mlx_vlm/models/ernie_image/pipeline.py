from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.models.flux2.tiling import TilingConfig

from .config import ErnieImageVariant, get_variant, validate_dimensions
from .download import download_model, validate_model_layout
from .scheduler import LinearFlowScheduler
from .tokenizer import ErnieTokenizer
from .weights import load_text_encoder, load_transformer, load_vae

TiledVAE = Literal["auto", "on", "off"]


@dataclass(frozen=True, slots=True)
class ErnieImageRuntimeConfig:
    """Memory and sequence settings for ERNIE Image inference."""

    evict_text_encoder: bool = True
    evict_transformer: bool = False
    tiled_vae: TiledVAE = "auto"
    max_sequence_length: int = 2048


def _image_array(decoded: mx.array) -> mx.array:
    images = mx.clip(decoded / 2 + 0.5, 0, 1)
    images = images.transpose(0, 2, 3, 1).astype(mx.float32)
    image = (images[0] * 255).round().astype(mx.uint8)
    mx.eval(image)
    return image


class ErnieImagePipeline:
    """Text-to-image inference pipeline for ERNIE Image and Turbo."""

    def __init__(
        self,
        *,
        variant: str | ErnieImageVariant,
        model_path: str | Path,
        runtime_config: ErnieImageRuntimeConfig | None = None,
    ) -> None:
        self.variant = get_variant(variant)
        self.model_path = validate_model_layout(model_path)
        self.runtime_config = runtime_config or ErnieImageRuntimeConfig()
        self.tokenizer = ErnieTokenizer(
            self.model_path,
            max_length=self.runtime_config.max_sequence_length,
        )
        self.text_encoder = load_text_encoder(self.model_path)
        self.quantization_config = getattr(
            self.text_encoder, "quantization_config", None
        )
        self.transformer = None
        self.vae = None
        self.prompt_cache: dict[
            tuple[str, str | None, float, int],
            tuple[mx.array, mx.array],
        ] = {}

    @classmethod
    def from_pretrained(
        cls,
        variant: str | ErnieImageVariant = "ernie-image-turbo",
        *,
        model_path: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        evict_text_encoder: bool = True,
        evict_transformer: bool = False,
        tiled_vae: TiledVAE = "auto",
        max_sequence_length: int = 2048,
    ) -> ErnieImagePipeline:
        """Load an ERNIE Image pipeline from a local path or Hugging Face."""

        spec = get_variant(variant)
        if model_path is None:
            if not download:
                raise FileNotFoundError(
                    f"No local model_path was provided for {spec.repo_id}"
                )
            model_path = download_model(
                spec,
                token=token,
                revision=revision,
                force_download=force_download,
            )
        return cls(
            variant=spec,
            model_path=model_path,
            runtime_config=ErnieImageRuntimeConfig(
                evict_text_encoder=evict_text_encoder,
                evict_transformer=evict_transformer,
                tiled_vae=tiled_vae,
                max_sequence_length=max_sequence_length,
            ),
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        """Count the untruncated tokens in a prompt."""

        return self.tokenizer.count_tokens(prompt)

    def _ensure_text_encoder(self):
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder(self.model_path)
        return self.text_encoder

    def _encode_prompts(
        self,
        prompt: str,
        *,
        negative_prompt: str | None,
        guidance: float,
        max_sequence_length: int,
    ) -> tuple[mx.array, mx.array]:
        cache_key = (prompt, negative_prompt, guidance, max_sequence_length)
        cached = self.prompt_cache.get(cache_key)
        if cached is not None:
            return cached
        prompts = (
            [prompt]
            if guidance <= 1.0
            else [
                negative_prompt if negative_prompt and negative_prompt.strip() else " ",
                prompt,
            ]
        )
        tokens = self.tokenizer.tokenize(
            prompts,
            max_length=max_sequence_length,
        )
        embeddings = self._ensure_text_encoder()(
            tokens.input_ids,
            tokens.attention_mask,
        )
        lengths = mx.sum(tokens.attention_mask, axis=1).astype(mx.int32)
        mx.eval(embeddings, lengths)
        self.prompt_cache[cache_key] = (embeddings, lengths)
        if self.runtime_config.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()
        return embeddings, lengths

    def _ensure_components(self) -> None:
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path)
            transformer_quantization = getattr(
                self.transformer, "quantization_config", None
            )
            if transformer_quantization is not None:
                self.quantization_config = transformer_quantization
        if self.vae is None:
            self.vae = load_vae(self.model_path)

    def _predictor(
        self,
        text_bth: mx.array,
        text_lens: mx.array,
        latents: mx.array,
    ):
        _, _, height, width = latents.shape
        cos, sin, attention_mask = self.transformer.get_pos_encoding(
            text_bth.shape[0],
            height,
            width,
            text_bth.shape[1],
            text_lens,
        )
        mx.eval(cos, sin, attention_mask)

        def predict(latent_input: mx.array, sigma: mx.array, guidance: float):
            timestep = sigma.reshape((1,)) * 1000.0
            if text_bth.shape[0] == 1:
                return self.transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    cos=cos,
                    sin=sin,
                    attn_mask=attention_mask,
                )
            prediction = self.transformer(
                hidden_states=mx.concatenate([latent_input, latent_input], axis=0),
                timestep=mx.broadcast_to(timestep, (2,)),
                text_bth=text_bth,
                text_lens=text_lens,
                cos=cos,
                sin=sin,
                attn_mask=attention_mask,
            )
            unconditional, conditional = prediction[:1], prediction[1:]
            return unconditional + guidance * (conditional - unconditional)

        return predict

    def _tiling_config(
        self,
        *,
        width: int,
        height: int,
        override: bool | None,
    ) -> TilingConfig | None:
        if override is True:
            return TilingConfig()
        if override is False:
            return None
        if self.runtime_config.tiled_vae == "on":
            return TilingConfig()
        if self.runtime_config.tiled_vae == "off":
            return None
        return TilingConfig() if max(width, height) >= 512 else None

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 42,
        steps: int = 8,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        max_sequence_length: int | None = None,
        tiled_vae: bool | None = None,
    ) -> mx.array:
        """Generate one RGB uint8 image array."""

        validate_dimensions(width=width, height=height)
        if not prompt:
            raise ValueError("prompt must not be empty")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        text_bth, text_lens = self._encode_prompts(
            prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            max_sequence_length=(
                max_sequence_length or self.runtime_config.max_sequence_length
            ),
        )
        self._ensure_components()
        latents = mx.random.normal(
            (1, 128, height // 16, width // 16),
            key=mx.random.key(seed),
        ).astype(mx.bfloat16)
        scheduler = LinearFlowScheduler(num_inference_steps=steps)
        predict = self._predictor(text_bth, text_lens, latents)
        for index in range(steps):
            velocity = predict(latents, scheduler.sigmas[index], guidance)
            latents = scheduler.step(
                velocity=velocity,
                step_index=index,
                latents=latents,
            )
            mx.eval(latents)
        decoded = self.vae.decode_packed_latents(
            latents,
            tiling_config=self._tiling_config(
                width=width,
                height=height,
                override=tiled_vae,
            ),
        )
        mx.eval(decoded)
        if self.runtime_config.evict_transformer:
            self.transformer = None
            self.vae = None
            gc.collect()
            mx.clear_cache()
        return _image_array(decoded)

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        """Generate one PIL image."""

        return Image.fromarray(np.array(self.generate_array(prompt, **kwargs)))


__all__ = ["ErnieImagePipeline", "ErnieImageRuntimeConfig", "TiledVAE"]
