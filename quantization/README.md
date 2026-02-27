# INT8 Quantization Pipeline

Converts FP32 model weights to INT8 for onchain inference via `sol_matmul_i8`.

## Quick Start

```bash
# Quantize both models
python quantization/quantize.py \
  --checkpoint checkpoints/world-model.pt \
  --output-dir quantization/output/world-model

python quantization/quantize.py \
  --checkpoint checkpoints/policy.pt \
  --output-dir quantization/output/policy

# Generate activation LUTs
python quantization/generate_luts.py -o quantization/output/luts
```

## Pipeline

### `quantize.py`

Unified quantization for any model type (Mamba2, MLP, PolicyMLP). Auto-detects architecture from checkpoint weight keys.

**Quantization strategy:**
- 2D weights (matmul): per-channel symmetric — one scale per output row
- 1D weights (bias/norm): per-tensor symmetric — one scale per tensor
- 3D weights (conv1d): reshape to 2D, then per-channel symmetric

**What it reads from the checkpoint:**
- `model_state_dict` — all weight tensors
- `encoding_config` — normalization scales and vocab sizes (stored in manifest)
- `context_len` — number of context frames (stored in manifest)

### `generate_luts.py`

Generates 256-entry INT8 lookup tables for activation functions:
- **SiLU** — gate activation in Mamba2 blocks
- **softplus** — dt computation in selective scan
- **rsqrt** — RMSNorm normalization
- **exp_neg** — A_bar decay in selective scan

## Output Files

### World Model (~4.1 MB from 16.6 MB FP32)

```
output/world-model/
├── weights_int8.bin    # 4,347,748 bytes — packed INT8 weights
└── manifest.json       # Architecture, per-weight scales, shard map
```

### Policy (~1.5 MB from 6.0 MB FP32)

```
output/policy/
├── weights_int8.bin    # 1,573,621 bytes
└── manifest.json
```

### Activation LUTs (1 KB)

```
output/luts/
├── luts.bin    # 1024 bytes — 4 LUTs x 256 entries
└── luts.json   # Scale metadata
```

## Manifest Structure

```json
{
  "format": "mamba2_int8_v1",
  "architecture": { "model_type": "mamba2", "d_model": 384, ... },
  "total_weight_bytes": 4347748,
  "context_len": 10,
  "encoding_config": { "xy_scale": 0.05, "action_vocab": 400, ... },
  "shard_map": {
    "num_shards": 2,
    "shards": [
      { "index": 0, "offset": 0, "size": 2174976 },
      { "index": 1, "offset": 2174976, "size": 2172772 }
    ]
  },
  "weights": {
    "layer_weights": { "layers.0.mamba.in_proj.weight": { ... }, ... },
    "embeddings": { "action_embed.weight": { ... }, ... },
    "projections": { "frame_proj.weight": { ... }, ... },
    "heads": { "p0_action_head.weight": { ... }, ... }
  },
  "error_summary": {
    "mean_snr_db": 44.6,
    "min_snr_db": 10.2,
    "max_snr_db": 53.8
  }
}
```

Each weight entry contains:
- `offset` / `size` — byte position within `weights_int8.bin`
- `shape` — original tensor shape
- `quantization` — `"per_channel_symmetric"` or `"per_tensor_symmetric"`
- `scales` — dequantization scales (array for per-channel, scalar for per-tensor)
- `snr_db` — signal-to-noise ratio in decibels

## Mapping to Solana Accounts

| Output | Solana Account | Notes |
|--------|---------------|-------|
| `weights_int8.bin` shard 0 | `WeightShard` #0 | Upload via CLI chunked writes |
| `weights_int8.bin` shard 1 | `WeightShard` #1 | 4096-byte aligned boundary |
| `manifest.json` | `ModelManifest` | Architecture + scales + shard map |
| `luts.bin` | `ModelManifest` LUT field | 1024 bytes, embedded in manifest account |

## Quality Assessment

SNR thresholds for INT8 quantization:
- **> 40 dB**: Excellent — imperceptible quality loss
- **30-40 dB**: Good — minor numerical differences, no behavioral impact
- **20-30 dB**: Acceptable — small weights may show larger relative error
- **< 20 dB**: Investigate — may indicate very small or near-zero weights

Current results:
- World model mean SNR: **44.6 dB** (39 matrix weights)
- Policy mean SNR: **45.0 dB** (13 matrix weights)

Low-SNR outliers are typically small embedding tables (e.g., 3x2 hurtbox_embed) where the absolute errors are tiny but relative errors are larger due to small signal power.
