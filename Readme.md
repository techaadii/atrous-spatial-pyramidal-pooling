# Atrous Spatial Pyramid Pooling — Segmentation Head

A clean PyTorch implementation of an **ASPP-based segmentation head** designed to sit on top of any ViT backbone that produces patch token embeddings.

---

## What is ASPP?

Atrous (dilated) convolutions apply filters with gaps between kernel elements. The dilation rate `r` controls how far apart the sampled neighbours are, giving a larger effective receptive field without increasing parameters.

```
Effective kernel size = k + (k-1)(r-1)

kernel=3, dilation=1  →  effective = 3   (standard conv)
kernel=3, dilation=6  →  effective = 13
kernel=3, dilation=12 →  effective = 25
kernel=3, dilation=18 →  effective = 37
```

ASPP runs **multiple dilated convolutions in parallel** at different rates, then fuses them — so the model sees the image at multiple scales simultaneously.

---

## Architecture

```
ViT Tokens  (B, N, D)
      │
      ▼
Reshape → (B, D, 16, 16)
      │
      ├── Conv 1×1          ─────────────────────────┐
      ├── Conv 3×3 dilation=6  (effective 13×13)      │
      ├── Conv 3×3 dilation=12 (effective 25×25)      ├── cat → (B, 256×5, 16, 16)
      ├── Conv 3×3 dilation=18 (effective 37×37)      │
      └── Global Avg Pool → upsample                 ─┘
                    │
                    ▼
            Bottleneck Conv 1×1
            → (B, 256, 16, 16)
                    │
                    ▼
            Classifier Conv 1×1
            → (B, num_classes, 16, 16)
                    │
                    ▼
            Bilinear upsample ×8
            → (B, num_classes, 128, 128)
```

---

## Usage

```python
from ASPPModule import AtrousConvolutionSegHead

# tokens from any ViT backbone
# shape: (B, N, D)  e.g. (8, 256, 384) for ViT-Small with patch 16
tokens = torch.randn(8, 256, 384)

head = AtrousConvolutionSegHead(
    embed_dim  = 384,   # must match ViT output dim
    num_classes = 2,    # binary: landslide / background
)

logits = head(tokens)   # (8, 2, 128, 128)
```

---

## Config

| Parameter | Value |
|-----------|-------|
| Dilation rates | 6, 12, 18 |
| ASPP output dim | 256 |
| Activation | GELU |
| Normalisation | LayerNorm |
| Dropout | 0.1 (bottleneck) |
| Upsample factor | ×8 |
| Input | `(B, N, D)` ViT tokens |
| Output | `(B, num_classes, H, W)` |

---

## Requirements

```
torch >= 2.0
```

---

## Reference

> Chen et al. — *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*
> [IEEE Xplore](https://ieeexplore.ieee.org/document/10116882)

---

## License

MIT
