from __future__ import annotations

import mlx.core as mx


class LinearFlowScheduler:
    """Linear flow-matching scheduler with ERNIE's static logarithmic shift."""

    def __init__(
        self,
        *,
        num_inference_steps: int,
        shift: float = 1.3863,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )
        base = mx.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            dtype=mx.float32,
        )
        shift_scale = mx.exp(mx.array(shift, dtype=mx.float32))
        shifted = shift_scale / (shift_scale + (1.0 / base - 1.0))
        self.sigmas = mx.concatenate(
            [shifted, mx.zeros((1,), dtype=mx.float32)],
            axis=0,
        )

    def step(
        self,
        *,
        velocity: mx.array,
        step_index: int,
        latents: mx.array,
    ) -> mx.array:
        """Advance one Euler step through the flow trajectory."""

        delta = (self.sigmas[step_index + 1] - self.sigmas[step_index]).astype(
            latents.dtype
        )
        return latents + delta * velocity.astype(latents.dtype)


__all__ = ["LinearFlowScheduler"]
