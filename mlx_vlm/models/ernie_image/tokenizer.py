from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer


@dataclass(frozen=True, slots=True)
class TokenizerOutput:
    """Token IDs and validity mask for an ERNIE prompt batch."""

    input_ids: mx.array
    attention_mask: mx.array


class ErnieTokenizer:
    """Raw-language tokenizer used by ERNIE Image's Mistral encoder."""

    def __init__(self, model_path: str | Path, max_length: int = 2048) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).expanduser() / "tokenizer"),
            local_files_only=True,
            use_fast=True,
        )

    def count_tokens(self, prompt: str) -> int:
        """Count prompt tokens without truncating the input."""

        tokens = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_tensors=None,
        )
        return len(tokens["input_ids"])

    def tokenize(
        self,
        prompt: str | list[str],
        *,
        max_length: int | None = None,
    ) -> TokenizerOutput:
        """Tokenize one or more raw prompts with longest-sequence padding."""

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [item if item is not None else "" for item in prompts]
        if all(item == "" for item in prompts):
            batch = len(prompts)
            return TokenizerOutput(
                input_ids=mx.array(np.empty((batch, 0), dtype=np.int32)),
                attention_mask=mx.array(np.empty((batch, 0), dtype=np.int32)),
            )
        tokens = self.tokenizer(
            prompts,
            padding="longest",
            max_length=max_length or self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        return TokenizerOutput(
            input_ids=mx.array(tokens["input_ids"]),
            attention_mask=mx.array(tokens["attention_mask"]),
        )


__all__ = ["ErnieTokenizer", "TokenizerOutput"]
