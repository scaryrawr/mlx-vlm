# ERNIE Image

`mlx-vlm` supports text-to-image generation with `baidu/ERNIE-Image` and
`baidu/ERNIE-Image-Turbo`, including MLX-native checkpoints saved by mflux in
affine, MXFP4, NVFP4, or MXFP8 form.

```bash
mlx_vlm.generate \
  --output-modality image \
  --model ~/.models/baidu/ERNIE-Image-Turbo-mxfp8 \
  --prompt "A watercolor lighthouse above a stormy sea" \
  --size 1024x1024 \
  --steps 8 \
  --guidance 1 \
  --output outputs/ernie.png
```

The unquantized Hugging Face checkpoints can be loaded by model ID:

```bash
mlx_vlm.generate \
  --output-modality image \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A studio photograph of a glass sculpture" \
  --size 1024x1024 \
  --steps 8
```

Python usage:

```python
from mlx_vlm.generate.image import ImageGenerationRequest
from mlx_vlm.models.ernie_image import load

model = load("~/.models/baidu/ERNIE-Image-Turbo-mxfp8")
result = model.generate(
    ImageGenerationRequest(
        prompt="A tiny robot tending a rooftop garden",
        seed=42,
        steps=8,
        width=1024,
        height=1024,
        guidance=1.0,
    )
)
result.save("outputs/ernie.png")
```
