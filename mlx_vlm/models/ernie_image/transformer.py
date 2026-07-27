from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mlx_vlm.models.flux2.constants import ModelConfig


def _rope(position: mx.array, dim: int, theta: int) -> mx.array:
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    return position[..., None].astype(mx.float32) / (theta**scale)


def _apply_rotary(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    half = q.shape[-1] // 2
    q_rotated = mx.concatenate([-q[..., half:], q[..., :half]], axis=-1)
    k_rotated = mx.concatenate([-k[..., half:], k[..., :half]], axis=-1)
    return q * cos + q_rotated * sin, k * cos + k_rotated * sin


class ErnieRopeEmbedder(nn.Module):
    """Build ERNIE's text-height-width rotary frequencies."""

    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, int, int]) -> None:
        super().__init__()
        if sum(axes_dim) != dim:
            raise ValueError(
                f"rope_axes_dim must sum to head dim {dim}, got {axes_dim}"
            )
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> mx.array:
        parts = [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        embedding = mx.concatenate(parts, axis=-1)[:, :, None, :]
        embedding = mx.stack([embedding, embedding], axis=-1)
        return embedding.reshape(*embedding.shape[:-2], -1)


def get_timestep_embedding(timesteps: mx.array, dim: int) -> mx.array:
    """Create sinusoidal flow-timestep embeddings."""

    half_dim = dim // 2
    frequency = math.log(10000) / half_dim
    frequencies = mx.exp(-frequency * mx.arange(half_dim, dtype=mx.float32))
    arguments = timesteps[:, None].astype(mx.float32) * frequencies[None, :]
    return mx.concatenate([mx.sin(arguments), mx.cos(arguments)], axis=-1)


class ErnieTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, embedding: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(embedding)))


class ErnieAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float,
        qk_layernorm: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = [nn.Linear(hidden_size, hidden_size, bias=False)]
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps) if qk_layernorm else None
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps) if qk_layernorm else None

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        batch, length, _ = x.shape
        q = self.to_q(x).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.to_k(x).reshape(batch, length, self.num_heads, self.head_dim)
        v = self.to_v(x).reshape(batch, length, self.num_heads, self.head_dim)
        if self.norm_q is not None and self.norm_k is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        q, k = _apply_rotary(q, k, cos, sin)
        output = scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out[0](output)


class ErnieFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.up_proj(x) * nn.gelu(self.gate_proj(x)))


class ErnieTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        eps: float,
        qk_layernorm: bool,
    ) -> None:
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieAttention(hidden_size, num_heads, eps, qk_layernorm)
        self.adaLN_mlp_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieFeedForward(hidden_size, ffn_hidden_size)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        modulation: tuple[mx.array, ...],
        mask: mx.array | None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        residual = x
        x = self.adaLN_sa_ln(x)
        x = residual + gate_msa * self.self_attention(
            x * (1 + scale_msa) + shift_msa,
            cos,
            sin,
            mask,
        )
        residual = x
        x = self.adaLN_mlp_ln(x)
        return residual + gate_mlp * self.mlp(x * (1 + scale_mlp) + shift_mlp)


class ErniePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        output = self.proj(x)
        batch, height, width, channels = output.shape
        return output.reshape(batch, height * width, channels)


class ErnieAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, affine=False)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(conditioning), 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class ErnieTransformer(nn.Module):
    """Joint image-text flow transformer used by ERNIE Image."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_embedder = ErniePatchEmbed(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False)
        self.time_embedding = ErnieTimestepEmbedder(hidden_size)
        self.adaln_modulation = nn.Linear(hidden_size, 6 * hidden_size)
        self.pos_embed = ErnieRopeEmbedder(
            dim=self.head_dim,
            theta=rope_theta,
            axes_dim=rope_axes_dim,
        )
        self.layers = [
            ErnieTransformerBlock(
                hidden_size,
                num_attention_heads,
                ffn_hidden_size,
                eps,
                qk_layernorm,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = ErnieAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        self._pos_cache: dict[
            tuple[int, int, int, tuple[int, ...]],
            tuple[mx.array, mx.array, mx.array],
        ] = {}

    def get_pos_encoding(
        self,
        batch: int,
        height: int,
        width: int,
        text_length: int,
        text_lens: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return cached rotary frequencies and the padded-text attention mask."""

        cache_key = (height, width, text_length, tuple(text_lens.tolist()))
        cached = self._pos_cache.get(cache_key)
        if cached is not None:
            return cached
        image_length = height * width
        grid_y, grid_x = mx.meshgrid(
            mx.arange(height, dtype=mx.float32),
            mx.arange(width, dtype=mx.float32),
            indexing="ij",
        )
        grid = mx.stack([grid_y.reshape(-1), grid_x.reshape(-1)], axis=-1)
        text_coordinate = text_lens.astype(mx.float32)[:, None, None]
        image_ids = mx.concatenate(
            [
                mx.broadcast_to(text_coordinate, (batch, image_length, 1)),
                mx.broadcast_to(grid[None, :, :], (batch, image_length, 2)),
            ],
            axis=-1,
        )
        text_positions = mx.arange(text_length, dtype=mx.float32)[None, :, None]
        text_ids = mx.concatenate(
            [
                mx.broadcast_to(text_positions, (batch, text_length, 1)),
                mx.zeros((batch, text_length, 2), dtype=mx.float32),
            ],
            axis=-1,
        )
        frequencies = self.pos_embed(mx.concatenate([image_ids, text_ids], axis=1))
        cos = mx.cos(frequencies).astype(ModelConfig.precision)
        sin = mx.sin(frequencies).astype(ModelConfig.precision)
        valid_text = mx.arange(text_length)[None, :] < text_lens[:, None]
        valid = mx.concatenate(
            [mx.ones((batch, image_length), dtype=mx.bool_), valid_text],
            axis=1,
        )
        mask = mx.where(valid, 0.0, -float("inf")).astype(mx.bfloat16)
        result = (cos, sin, mask[:, None, None, :])
        if len(self._pos_cache) >= 64:
            self._pos_cache.pop(next(iter(self._pos_cache)))
        self._pos_cache[cache_key] = result
        return result

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        text_bth: mx.array,
        text_lens: mx.array,
        *,
        cos: mx.array | None = None,
        sin: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        batch, _, height, width = hidden_states.shape
        image_length = height * width
        text_length = text_bth.shape[1]
        image = self.x_embedder(hidden_states.transpose(0, 2, 3, 1))
        text = self.text_proj(text_bth)
        x = mx.concatenate([image, text], axis=1)
        if cos is None or sin is None or attn_mask is None:
            cos, sin, attn_mask = self.get_pos_encoding(
                batch,
                height,
                width,
                text_length,
                text_lens,
            )
        timestep_embedding = get_timestep_embedding(
            timestep.astype(mx.float32), self.hidden_size
        )
        conditioning = self.time_embedding(timestep_embedding)
        modulation = tuple(
            item[:, None, :]
            for item in mx.split(
                self.adaln_modulation(nn.silu(conditioning)),
                6,
                axis=-1,
            )
        )
        for layer in self.layers:
            x = layer(x, cos, sin, modulation, attn_mask)
        image = self.final_norm(x[:, :image_length, :], conditioning).astype(
            hidden_states.dtype
        )
        patches = self.final_linear(image)
        output = patches.reshape(
            batch,
            height,
            width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        output = output.transpose(0, 5, 1, 3, 2, 4)
        return output.reshape(
            batch,
            self.out_channels,
            height * self.patch_size,
            width * self.patch_size,
        )


__all__ = ["ErnieTransformer", "get_timestep_embedding"]
