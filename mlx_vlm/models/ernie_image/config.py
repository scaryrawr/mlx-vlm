from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ErnieImageVariant:
    """Configuration for an ERNIE Image checkpoint family."""

    name: str
    aliases: tuple[str, ...]
    repo_id: str
    default_steps: int
    default_guidance: float


def _variant(
    name: str,
    repo_id: str,
    *,
    steps: int,
    guidance: float,
) -> ErnieImageVariant:
    return ErnieImageVariant(
        name=name,
        aliases=(name, repo_id, repo_id.rsplit("/", 1)[-1]),
        repo_id=repo_id,
        default_steps=steps,
        default_guidance=guidance,
    )


VARIANTS = {
    "ernie-image": _variant(
        "ernie-image",
        "baidu/ERNIE-Image",
        steps=50,
        guidance=4.0,
    ),
    "ernie-image-turbo": _variant(
        "ernie-image-turbo",
        "baidu/ERNIE-Image-Turbo",
        steps=8,
        guidance=1.0,
    ),
}

_ALIASES = {
    alias.lower(): variant for variant in VARIANTS.values() for alias in variant.aliases
}


def get_variant(
    name: str | ErnieImageVariant = "ernie-image-turbo",
) -> ErnieImageVariant:
    """Resolve a supported ERNIE Image alias to its variant."""

    if isinstance(name, ErnieImageVariant):
        return name
    key = name.strip().lower().rstrip("/")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIANTS))
        raise ValueError(
            f"Unknown ERNIE Image variant {name!r}. Supported: {supported}"
        ) from exc


def variant_from_local_path(model_path: str | Path) -> ErnieImageVariant:
    """Infer the ERNIE Image variant represented by a local checkpoint."""

    root = Path(model_path).expanduser()
    name = str(root).lower().replace("_", "-")
    if "turbo" in name:
        return VARIANTS["ernie-image-turbo"]
    if "ernie-image" in name:
        return VARIANTS["ernie-image"]

    model_index = root / "model_index.json"
    if model_index.exists():
        metadata = json.loads(model_index.read_text())
        source = str(metadata.get("_name_or_path") or metadata.get("name") or "")
        if "turbo" in source.lower():
            return VARIANTS["ernie-image-turbo"]
        if "ernie" in str(metadata.get("_class_name") or "").lower():
            return VARIANTS["ernie-image"]

    transformer_index = root / "transformer" / "model.safetensors.index.json"
    if transformer_index.exists():
        index = json.loads(transformer_index.read_text())
        metadata = index.get("metadata", {})
        variant = metadata.get("variant") if isinstance(metadata, dict) else None
        if variant:
            return get_variant(str(variant))
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict):
            markers = {
                "x_embedder.proj.weight",
                "text_proj.weight",
                "layers.0.self_attention.to_q.weight",
            }
            if markers <= set(weight_map):
                # Base and Turbo share an architecture; saved checkpoints without
                # variant metadata use the more common Turbo runtime defaults.
                return VARIANTS["ernie-image-turbo"]

    if root.name.lower() in {"ernie-image", "ernie_image"}:
        return VARIANTS["ernie-image"]
    raise ValueError(
        f"Could not infer an ERNIE Image variant from local model path: {root}. "
        "Use a directory name containing ERNIE-Image or ERNIE-Image-Turbo."
    )


def validate_dimensions(*, width: int, height: int) -> None:
    """Validate dimensions supported by the ERNIE Image latent grid."""

    for label, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{label} must be in [256, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")


def list_variants() -> tuple[str, ...]:
    """Return canonical ERNIE Image variant names."""

    return tuple(VARIANTS)


__all__ = [
    "VARIANTS",
    "ErnieImageVariant",
    "get_variant",
    "list_variants",
    "validate_dimensions",
    "variant_from_local_path",
]
