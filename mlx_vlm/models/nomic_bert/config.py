from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "nomic_bert"
    n_embd: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    n_inner: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    intermediate_size: int | None = None
    vocab_size: int = 30528
    type_vocab_size: int = 2
    max_position_embeddings: int | None = None
    n_positions: int | None = None
    activation_function: str = "swiglu"
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    layer_norm_epsilon: float | None = None
    layer_norm_eps: float | None = None
    rotary_emb_base: float = 1000.0
    rotary_emb_fraction: float = 1.0
    rotary_emb_interleaved: bool = False
    max_trained_positions: int = 2048
    qkv_proj_bias: bool = False
    mlp_fc1_bias: bool = False
    mlp_fc2_bias: bool = False
    prenorm: bool = False
    causal: bool = False
    add_pooling_layer: bool = False
    pad_token_id: int = 0

    def __post_init__(self):
        self.n_embd = self.n_embd if self.n_embd is not None else self.hidden_size
        self.n_layer = (
            self.n_layer if self.n_layer is not None else self.num_hidden_layers
        )
        self.n_head = (
            self.n_head if self.n_head is not None else self.num_attention_heads
        )
        self.n_inner = (
            self.n_inner if self.n_inner is not None else self.intermediate_size
        )
        if None in (self.n_embd, self.n_layer, self.n_head, self.n_inner):
            raise ValueError(
                "NomicBERT config must include hidden/layer/head/intermediate sizes."
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"hidden size ({self.n_embd}) must be divisible by "
                f"attention heads ({self.n_head})."
            )

        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.intermediate_size = self.n_inner
        if self.layer_norm_epsilon is None:
            self.layer_norm_epsilon = (
                self.layer_norm_eps if self.layer_norm_eps is not None else 1e-12
            )
        self.layer_norm_eps = self.layer_norm_epsilon
        if self.max_position_embeddings is None:
            self.max_position_embeddings = (
                self.n_positions or self.max_trained_positions
            )
        if self.causal:
            raise NotImplementedError("Causal NomicBERT variants are not supported.")
