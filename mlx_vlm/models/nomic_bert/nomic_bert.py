import mlx.core as mx
from mlx import nn

from ..pooling import EmbeddingOutput, normalize_embeddings, pool_by_config
from .config import ModelConfig


class NomicBertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = (
            config.max_position_embeddings if config.rotary_emb_fraction <= 0 else 0
        )
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                self.type_vocab_size, config.hidden_size
            )
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                self.max_position_embeddings, config.hidden_size
            )

    def __call__(self, input_ids, token_type_ids=None, position_ids=None):
        embeddings = self.word_embeddings(input_ids)
        batch_size, seq_length, _ = embeddings.shape
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = mx.arange(seq_length, dtype=mx.int32)[None, :]
            embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings


class NomicBertGatedMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.activation_function = config.activation_function
        self.fc11 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_fc1_bias
        )
        self.fc12 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_fc1_bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_fc2_bias
        )

    def _activation(self, x):
        if self.activation_function == "swiglu":
            return nn.silu(x)
        if self.activation_function == "geglu":
            return nn.gelu(x)
        if self.activation_function == "glu":
            return mx.sigmoid(x)
        return nn.gelu(x)

    def __call__(self, hidden_states):
        return self.fc2(
            self.fc11(hidden_states) * self._activation(self.fc12(hidden_states))
        )


class NomicBertMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_fc1_bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_fc2_bias
        )

    def __call__(self, hidden_states):
        return self.fc2(nn.gelu(self.fc1(hidden_states)))


class NomicBertAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.rotary_emb_dim = int(self.head_dim * config.rotary_emb_fraction)
        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.qkv_proj_bias
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.qkv_proj_bias
        )
        self.drop = nn.Dropout(p=config.attn_pdrop)
        self.rotary_emb = (
            nn.RoPE(
                self.rotary_emb_dim,
                traditional=config.rotary_emb_interleaved,
                base=config.rotary_emb_base,
            )
            if self.rotary_emb_dim > 0
            else None
        )

    def __call__(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            batch_size, seq_length, 3, self.num_heads, self.head_dim
        )
        query = qkv[:, :, 0].transpose(0, 2, 1, 3)
        key = qkv[:, :, 1].transpose(0, 2, 1, 3)
        value = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if self.rotary_emb is not None:
            query = self.rotary_emb(query)
            key = self.rotary_emb(key)
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=attention_mask,
        )
        output = self.drop(output)
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )
        return self.out_proj(output)


class NomicBertBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.prenorm = config.prenorm
        self.attn = NomicBertAttention(config)
        self.mlp = (
            NomicBertGatedMLP(config)
            if config.activation_function in ("glu", "swiglu", "geglu")
            else NomicBertMLP(config)
        )
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, bias=True
        )
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, bias=True
        )

    def __call__(self, hidden_states, attention_mask=None):
        if self.prenorm:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            hidden_states = residual + self.dropout1(
                self.attn(hidden_states, attention_mask=attention_mask)
            )
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            return residual + self.dropout2(self.mlp(hidden_states))

        attention_output = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm1(self.dropout1(attention_output) + hidden_states)
        mlp_output = self.mlp(hidden_states)
        return self.norm2(self.dropout2(mlp_output) + hidden_states)


class NomicBertEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = [NomicBertBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class NomicBertPooler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, hidden_states):
        return mx.tanh(self.dense(hidden_states[:, 0]))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = NomicBertEmbeddings(config)
        self.emb_drop = nn.Dropout(config.embd_pdrop)
        self.emb_ln = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, bias=True
        )
        self.encoder = NomicBertEncoder(config)
        if config.add_pooling_layer:
            self.pooler = NomicBertPooler(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
        **kwargs,
    ) -> EmbeddingOutput:
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_length), dtype=mx.int32)
        hidden_states = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        hidden_states = self.emb_drop(self.emb_ln(hidden_states))
        mask = 1.0 - attention_mask[:, None, None, :].astype(hidden_states.dtype)
        mask = mask * -10000.0
        hidden_states = self.encoder(hidden_states, attention_mask=mask)

        pooling_config = getattr(self, "pooling_config", None) or {
            "pooling_mode": "mean"
        }
        text_embeds = normalize_embeddings(
            pool_by_config(hidden_states, attention_mask, pooling_config)
        )
        return EmbeddingOutput(
            last_hidden_state=hidden_states,
            text_embeds=text_embeds,
        )

    def sanitize(self, weights):
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "position_ids" not in key
        }

    @property
    def layers(self):
        return self.encoder.layers
