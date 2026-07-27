from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.vae import Flux2VAE

from .text_encoder import ErnieMistralTextEncoder
from .transformer import ErnieTransformer

FULL_DECODER_CHANNELS = (128, 256, 512, 512)


def _load_safetensors(
    directory: Path,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing weight directory: {directory}")
    weights: dict[str, mx.array] = {}
    metadata: dict[str, Any] = {}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensor index: {index_path}") from exc
        index_metadata = index.get("metadata", {})
        if isinstance(index_metadata, dict):
            metadata.update(index_metadata)
    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    for path in files:
        shard, shard_metadata = mx.load(str(path), return_metadata=True)
        weights.update(shard)
        if isinstance(shard_metadata, dict):
            metadata.update(shard_metadata)
    return weights, metadata


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    candidates = [metadata]
    for container_key in ("quantization", "quantization_config"):
        value = metadata.get(container_key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        for key in keys:
            if key in candidate:
                return candidate[key]
    return None


def _metadata_int(metadata: dict[str, Any], *keys: str) -> int | None:
    value = _metadata_value(metadata, *keys)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid quantization metadata value: {value!r}") from exc


def _quantization_config(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
) -> tuple[dict[str, int | str], set[str]] | None:
    paths = {key.removesuffix(".scales") for key in weights if key.endswith(".scales")}
    if not paths:
        return None
    modules = dict(model.named_modules())
    inferred_bits: set[int] = set()
    inferred_group_sizes: set[int] = set()
    for path in paths:
        module = modules.get(path)
        packed = weights.get(f"{path}.weight")
        scales = weights.get(f"{path}.scales")
        if module is None or not hasattr(module, "to_quantized"):
            raise ValueError(
                f"Quantized checkpoint path has no matching module: {path}"
            )
        if packed is None or scales is None:
            raise ValueError(f"Incomplete quantized weights for module: {path}")
        if packed.dtype != mx.uint32:
            raise ValueError(
                f"Expected packed uint32 weights for {path}, got {packed.dtype}"
            )
        if not hasattr(module, "weight") or module.weight.ndim != 2:
            raise ValueError(f"Unsupported quantized module shape for: {path}")
        output_dims, input_dims = module.weight.shape
        if packed.shape[0] != output_dims or scales.shape[0] != output_dims:
            raise ValueError(
                f"Quantized output shape mismatch for {path}: "
                f"weight={tuple(packed.shape)}, scales={tuple(scales.shape)}, "
                f"expected output={output_dims}"
            )
        packed_bits = packed.shape[-1] * 32
        if packed_bits % input_dims or input_dims % scales.shape[-1]:
            raise ValueError(f"Could not infer quantization shape for {path}")
        inferred_bits.add(packed_bits // input_dims)
        inferred_group_sizes.add(input_dims // scales.shape[-1])
    if len(inferred_bits) != 1 or len(inferred_group_sizes) != 1:
        raise ValueError("Mixed quantization parameters are not supported")
    inferred_bits_value = inferred_bits.pop()
    inferred_group_size = inferred_group_sizes.pop()
    bits = _metadata_int(metadata, "bits", "num_bits", "quantization_level")
    group_size = _metadata_int(
        metadata,
        "group_size",
        "quantization_group_size",
    )
    if bits is not None and bits != inferred_bits_value:
        raise ValueError(
            "Quantization bits disagree with checkpoint tensor shapes: "
            f"configured={bits}, inferred={inferred_bits_value}"
        )
    if group_size is not None and group_size != inferred_group_size:
        raise ValueError(
            "Quantization group size disagrees with checkpoint tensor shapes: "
            f"configured={group_size}, inferred={inferred_group_size}"
        )
    mode = str(_metadata_value(metadata, "mode", "quantization_mode") or "affine")
    if mode not in {"affine", "mxfp4", "nvfp4", "mxfp8"}:
        raise ValueError(f"Unsupported ERNIE Image quantization mode: {mode}")
    if mode == "affine":
        missing_biases = [path for path in paths if f"{path}.biases" not in weights]
        if missing_biases:
            raise ValueError(
                f"Affine quantized module is missing bias tensors: {missing_biases[0]}"
            )
    return (
        {
            "bits": bits or inferred_bits_value,
            "group_size": group_size or inferred_group_size,
            "mode": mode,
        },
        paths,
    )


def apply_quantized_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
) -> dict[str, int | str] | None:
    """Apply dense or pre-quantized MLX weights and return quantization metadata."""

    quantized = _quantization_config(model, weights, metadata)
    if quantized is not None:
        config, paths = quantized
        nn.quantize(
            model,
            group_size=int(config["group_size"]),
            bits=int(config["bits"]),
            mode=str(config["mode"]),
            class_predicate=lambda path, module: path in paths,
        )
    else:
        config = None
    model.update(tree_unflatten(list(weights.items())), strict=True)
    model.quantization_config = config
    return config


def _cast_float(value: mx.array) -> mx.array:
    return (
        value.astype(ModelConfig.precision)
        if mx.issubdtype(value.dtype, mx.floating)
        else value
    )


def _match_conv_layout(
    value: mx.array,
    *,
    target_shape: tuple[int, ...] | None,
    key: str,
) -> mx.array:
    if target_shape is None or tuple(value.shape) == target_shape:
        return value
    transposed = value.transpose(0, 2, 3, 1)
    if tuple(transposed.shape) == target_shape:
        return transposed
    raise ValueError(
        f"Unsupported convolution weight shape for {key}: "
        f"checkpoint={tuple(value.shape)}, expected={target_shape}"
    )


def load_text_encoder(model_path: str | Path) -> ErnieMistralTextEncoder:
    """Load ERNIE's text encoder from dense or mflux-quantized weights."""

    raw, metadata = _load_safetensors(Path(model_path).expanduser() / "text_encoder")
    model = ErnieMistralTextEncoder()
    weights = {
        key: _cast_float(value)
        for key, value in raw.items()
        if key.startswith("language_model.model.")
    }
    apply_quantized_weights(model, weights, metadata)
    return model


def load_transformer(model_path: str | Path) -> ErnieTransformer:
    """Load ERNIE's joint flow transformer."""

    raw, metadata = _load_safetensors(Path(model_path).expanduser() / "transformer")
    model = ErnieTransformer()
    target_shapes = {
        key: tuple(value.shape) for key, value in tree_flatten(model.parameters())
    }
    weights = {}
    for raw_key, value in raw.items():
        key = raw_key.replace("adaLN_modulation.1.", "adaln_modulation.")
        tensor = _cast_float(value)
        if tensor.ndim == 4:
            tensor = _match_conv_layout(
                tensor,
                target_shape=target_shapes.get(key),
                key=key,
            )
        weights[key] = tensor
    apply_quantized_weights(model, weights, metadata)
    return model


def load_vae(model_path: str | Path) -> Flux2VAE:
    """Load the full-channel FLUX.2 decoder used by ERNIE Image."""

    raw, metadata = _load_safetensors(Path(model_path).expanduser() / "vae")
    model = Flux2VAE(decoder_block_out_channels=FULL_DECODER_CHANNELS)
    target_shapes = {
        key: tuple(value.shape) for key, value in tree_flatten(model.parameters())
    }
    prefixes = ("decoder.", "post_quant_conv.", "bn.")
    weights = {}
    for raw_key, value in raw.items():
        if not raw_key.startswith(prefixes) or raw_key.endswith(".num_batches_tracked"):
            continue
        key = raw_key.replace(".to_out.0.", ".to_out.")
        tensor = _cast_float(value)
        if tensor.ndim == 4:
            tensor = _match_conv_layout(
                tensor,
                target_shape=target_shapes.get(key),
                key=key,
            )
        weights[key] = tensor
    apply_quantized_weights(model, weights, metadata)
    return model


__all__ = [
    "apply_quantized_weights",
    "load_text_encoder",
    "load_transformer",
    "load_vae",
]
