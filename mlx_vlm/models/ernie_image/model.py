from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx

from mlx_vlm.generate.image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)

from .config import ErnieImageVariant, get_variant, variant_from_local_path
from .download import validate_model_layout
from .pipeline import ErnieImagePipeline


def resolve_variant(
    model: str | ErnieImageVariant | None,
) -> ErnieImageVariant:
    """Resolve a model ID or local path to an ERNIE Image variant."""

    if isinstance(model, ErnieImageVariant):
        return model
    if model is None:
        return get_variant()
    path = Path(model).expanduser()
    if path.exists():
        return variant_from_local_path(path)
    return get_variant(model)


def can_load(model: str) -> bool:
    """Return whether a model ID or path is a loadable ERNIE Image checkpoint."""

    path = Path(model).expanduser()
    try:
        if path.exists():
            validate_model_layout(path)
        resolve_variant(model)
        return True
    except (FileNotFoundError, ValueError):
        return False


@dataclass(slots=True)
class ErnieImageGenerationModel(ImageGenerationModel):
    """Image-generation protocol adapter for ERNIE Image."""

    is_image_generation_model: ClassVar[bool] = True
    model_type: ClassVar[str] = "ernie_image"
    pipeline: ErnieImagePipeline
    model_id: str
    family: str = "ernie_image"

    @property
    def variant(self) -> str:
        return self.pipeline.variant.name

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        seed = 0 if request.seed is None else request.seed
        array = self.pipeline.generate_array(
            request.prompt,
            seed=seed,
            steps=request.steps,
            width=request.width,
            height=request.height,
            guidance=request.guidance,
            negative_prompt=request.extra.get("negative_prompt"),
            max_sequence_length=request.extra.get("max_sequence_length"),
            tiled_vae=request.extra.get("tiled_vae"),
        )
        quantization = getattr(self.pipeline, "quantization_config", None)
        metadata = {
            "model_path": str(self.pipeline.model_path),
            "architecture": "quantized" if quantization else "dense",
            "default_steps": self.pipeline.variant.default_steps,
            "default_guidance": self.pipeline.variant.default_guidance,
            "vae_variant": "full",
        }
        if quantization is not None:
            metadata["quantization"] = dict(quantization)
        return ImageGenerationResult(
            array=array,
            seed=seed,
            width=request.width,
            height=request.height,
            steps=request.steps,
            model=self.model_id,
            family=self.family,
            variant=self.variant,
            guidance=request.guidance,
            prompt_tokens=self.pipeline.count_prompt_tokens(request.prompt),
            peak_memory=mx.get_peak_memory() / 1e9,
            metadata=metadata,
        )

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return can_load(model)

    @classmethod
    def from_model_id(
        cls,
        model: str = "ernie-image-turbo",
        **kwargs: Any,
    ) -> ErnieImageGenerationModel:
        model_path_arg = kwargs.pop("model_path", None)
        model_path = (
            Path(model).expanduser()
            if model_path_arg is None and Path(model).expanduser().exists()
            else model_path_arg
        )
        variant = resolve_variant(model_path if model_path is not None else model)
        pipeline = ErnieImagePipeline.from_pretrained(
            variant,
            model_path=model_path,
            download=kwargs.pop("download", True),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            force_download=kwargs.pop("force_download", False),
            evict_text_encoder=kwargs.pop("evict_text_encoder", True),
            evict_transformer=kwargs.pop("evict_transformer", False),
            tiled_vae=kwargs.pop("tiled_vae", "auto"),
            max_sequence_length=kwargs.pop("max_sequence_length", 2048),
        )
        return cls(pipeline=pipeline, model_id=str(model))


def load(
    model: str = "ernie-image-turbo",
    **kwargs: Any,
) -> ErnieImageGenerationModel:
    """Load an ERNIE Image model through the shared generation adapter."""

    return ErnieImageGenerationModel.from_model_id(model, **kwargs)


__all__ = [
    "ErnieImageGenerationModel",
    "can_load",
    "load",
    "resolve_variant",
]
