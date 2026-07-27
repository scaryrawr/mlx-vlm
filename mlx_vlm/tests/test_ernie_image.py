import importlib
import json
from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.ernie_image.config import (
    get_variant,
    validate_dimensions,
    variant_from_local_path,
)
from mlx_vlm.models.ernie_image.download import (
    DOWNLOAD_PATTERNS,
    validate_model_layout,
)
from mlx_vlm.models.ernie_image.model import ErnieImageGenerationModel
from mlx_vlm.models.ernie_image.pipeline import ErnieImagePipeline
from mlx_vlm.models.ernie_image.scheduler import LinearFlowScheduler
from mlx_vlm.models.ernie_image.transformer import ErnieTransformer
from mlx_vlm.models.ernie_image.weights import apply_quantized_weights

image_module = importlib.import_module("mlx_vlm.generate.image")


def _write_layout(root: Path) -> None:
    for relative in (
        "transformer/0.safetensors",
        "transformer/model.safetensors.index.json",
        "text_encoder/0.safetensors",
        "vae/0.safetensors",
        "tokenizer/tokenizer.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


@pytest.mark.parametrize(
    "alias,variant",
    [
        ("ernie-image", "ernie-image"),
        ("baidu/ERNIE-Image", "ernie-image"),
        ("ernie-image-turbo", "ernie-image-turbo"),
        ("baidu/ERNIE-Image-Turbo", "ernie-image-turbo"),
    ],
)
def test_ernie_variant_aliases(alias: str, variant: str) -> None:
    assert get_variant(alias).name == variant


def test_ernie_local_layout_and_variant(tmp_path: Path) -> None:
    model_path = tmp_path / "ERNIE-Image-Turbo-mxfp8"
    _write_layout(model_path)

    assert validate_model_layout(model_path) == model_path
    assert variant_from_local_path(model_path).name == "ernie-image-turbo"

    base_path = tmp_path / "ERNIE-Image-mxfp8"
    _write_layout(base_path)
    assert variant_from_local_path(base_path).name == "ernie-image"


def test_ernie_component_index_detection(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    index = {
        "weight_map": {
            "x_embedder.proj.weight": "0.safetensors",
            "text_proj.weight": "0.safetensors",
            "layers.0.self_attention.to_q.weight": "0.safetensors",
        }
    }
    (tmp_path / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps(index)
    )

    assert (
        image_generation_model_class(tmp_path.as_posix()) is ErnieImageGenerationModel
    )
    assert is_image_generation_model(tmp_path.as_posix())


def test_ernie_remote_alias_detection_does_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_module,
        "get_model_path",
        lambda *args, **kwargs: pytest.fail("remote metadata should not be needed"),
    )

    assert (
        image_generation_model_class("baidu/ERNIE-Image-Turbo")
        is ErnieImageGenerationModel
    )
    assert is_image_generation_model("ernie-image-turbo")


def test_ernie_download_patterns_include_all_components() -> None:
    assert "transformer/*.safetensors" in DOWNLOAD_PATTERNS
    assert "text_encoder/*.safetensors" in DOWNLOAD_PATTERNS
    assert "vae/*.safetensors" in DOWNLOAD_PATTERNS
    assert "tokenizer/**" in DOWNLOAD_PATTERNS


@pytest.mark.parametrize("width,height", [(255, 512), (512, 2050), (513, 512)])
def test_ernie_validate_dimensions_rejects_bad_sizes(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(width=width, height=height)


def test_ernie_scheduler_uses_static_log_shift() -> None:
    scheduler = LinearFlowScheduler(num_inference_steps=2, shift=1.3863)

    expected_first = mx.exp(mx.array(1.3863)) / (
        mx.exp(mx.array(1.3863)) + (1.0 / mx.array(1.0) - 1.0)
    )
    expected_second = mx.exp(mx.array(1.3863)) / (
        mx.exp(mx.array(1.3863)) + (1.0 / mx.array(0.5) - 1.0)
    )
    assert mx.allclose(
        scheduler.sigmas[:2], mx.stack([expected_first, expected_second])
    )
    assert scheduler.sigmas[-1].item() == 0.0

    latents = mx.ones((1, 2), dtype=mx.bfloat16)
    velocity = mx.ones_like(latents)
    stepped = scheduler.step(velocity=velocity, step_index=0, latents=latents)
    assert mx.allclose(
        stepped,
        latents + (scheduler.sigmas[1] - scheduler.sigmas[0]).astype(latents.dtype),
    )


def test_ernie_transformer_preserves_latent_shape() -> None:
    model = ErnieTransformer(
        hidden_size=32,
        num_attention_heads=4,
        num_layers=1,
        ffn_hidden_size=64,
        in_channels=8,
        out_channels=8,
        text_in_dim=12,
        rope_axes_dim=(2, 2, 4),
    )
    hidden_states = mx.zeros((1, 8, 2, 3), dtype=mx.bfloat16)
    output = model(
        hidden_states=hidden_states,
        timestep=mx.array([1000.0]),
        text_bth=mx.zeros((1, 4, 12), dtype=mx.bfloat16),
        text_lens=mx.array([3], dtype=mx.int32),
    )
    mx.eval(output)

    assert output.shape == hidden_states.shape


def test_ernie_mxfp8_weights_load_without_affine_biases() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(32, 16, bias=False)

    dense = mx.arange(16 * 32, dtype=mx.float32).reshape(16, 32) / 100
    weight, scales = mx.quantize(dense, group_size=32, bits=8, mode="mxfp8")
    model = TinyModel()
    quantization = apply_quantized_weights(
        model,
        {"linear.weight": weight, "linear.scales": scales},
        {
            "quantization_level": "8",
            "quantization_mode": "mxfp8",
            "quantization_group_size": "32",
        },
    )

    assert isinstance(model.linear, nn.QuantizedLinear)
    assert model.linear.biases is None
    assert quantization == {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    assert model.linear(mx.ones((1, 32))).shape == (1, 16)


def test_ernie_model_adapter_returns_generation_result() -> None:
    pipeline = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipeline.variant = get_variant("ernie-image-turbo")
    pipeline.model_path = Path("/tmp/ernie")
    pipeline.quantization_config = {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    pipeline.count_prompt_tokens = lambda prompt: 3
    pipeline.generate_array = lambda *args, **kwargs: mx.zeros(
        (32, 48, 3), dtype=mx.uint8
    )
    model = ErnieImageGenerationModel(
        pipeline=pipeline,
        model_id="baidu/ERNIE-Image-Turbo",
    )

    result = model.generate(
        ImageGenerationRequest(
            prompt="a small test",
            seed=7,
            steps=1,
            width=48,
            height=32,
            guidance=1.0,
        )
    )

    assert result.family == "ernie_image"
    assert result.variant == "ernie-image-turbo"
    assert result.array.shape == (32, 48, 3)
    assert result.metadata["quantization"]["mode"] == "mxfp8"
