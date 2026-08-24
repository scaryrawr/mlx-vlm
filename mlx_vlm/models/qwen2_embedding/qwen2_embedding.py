import mlx.core as mx
from mlx import nn

from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from ..qwen2.language import Qwen2Model
from .config import ModelConfig


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        **kwargs,
    ) -> EmbeddingOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_length), dtype=mx.int32)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} does not match "
                f"input_ids shape {input_ids.shape}"
            )

        hidden_states = self.model.embed_tokens(input_ids)
        token_mask = attention_mask[:, :, None].astype(hidden_states.dtype)
        hidden_states = mx.where(token_mask == 1, hidden_states, 0.0)

        positions = mx.arange(seq_length)
        causal = positions[None, :] <= positions[:, None]
        valid_keys = attention_mask[:, None, None, :].astype(mx.bool_)
        allowed = causal[None, None, :, :] & valid_keys
        mask = mx.where(allowed, 0.0, mx.finfo(hidden_states.dtype).min).astype(
            hidden_states.dtype
        )

        for layer in self.model.layers:
            hidden_states = layer(hidden_states, mask, None)
            hidden_states = mx.where(token_mask == 1, hidden_states, 0.0)
        hidden_states = self.model.norm(hidden_states)

        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "lasttoken"
        }
        text_embeds = normalize_embeddings(
            pool_by_config(hidden_states, attention_mask, pooling_config)
        )
        return EmbeddingOutput(
            last_hidden_state=hidden_states,
            text_embeds=text_embeds,
        )

    def sanitize(self, weights):
        out = {}
        for key, value in weights.items():
            if "lm_head.weight" in key or "rotary_emb.inv_freq" in key:
                continue
            if not key.startswith("model."):
                key = f"model.{key}"
            out[key] = value
        return out

    @property
    def layers(self):
        return self.model.layers
