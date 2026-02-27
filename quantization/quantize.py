#!/usr/bin/env python3
"""Quantize model weights to INT8 for onchain inference via sol_matmul_i8.

Unified script handling both model types:
  - FrameStackMamba2 (world model) — detected by `layers.0.mamba` keys
  - PolicyMLP (policy) — detected by `trunk.0` keys

Quantization strategy:
  - 2D weights (matmul): per-channel symmetric (scale per output row)
  - 1D weights (bias/norm/params): per-tensor symmetric (single scale)
  - 3D weights (conv1d): reshape to 2D then per-channel symmetric

Iterates ALL state_dict keys generically — no hardcoded weight name patterns.

Output:
  - weights_int8.bin: Packed INT8 weights, 4096-byte aligned shard boundaries
  - manifest.json: Architecture, per-weight scales, shard map, encoding_config

Usage:
    python quantization/quantize.py --checkpoint checkpoints/world-model.pt \
        --output-dir quantization/output/world-model

    python quantization/quantize.py --checkpoint checkpoints/policy.pt \
        --output-dir quantization/output/policy
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm


# ── Quantization core ────────────────────────────────────────────────────────


def quantize_per_channel_symmetric(
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric quantization for 2D weight matrices.

    For W of shape (out_features, in_features):
        scale[c] = max(|W[c, :]|) / 127
        W_int8[c, :] = round(W[c, :] / scale[c])

    Returns (w_int8, scales) with shapes (out, in) and (out,).
    """
    assert weight.ndim == 2, f"Expected 2D, got {weight.shape}"
    abs_max = np.abs(weight).max(axis=1)
    abs_max = np.maximum(abs_max, 1e-8)
    scales = abs_max / 127.0
    w_scaled = weight / scales[:, np.newaxis]
    w_int8 = np.round(w_scaled).clip(-128, 127).astype(np.int8)
    return w_int8, scales.astype(np.float32)


def quantize_per_tensor_symmetric(
    tensor: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric quantization for 1D vectors."""
    abs_max = max(np.abs(tensor).max(), 1e-8)
    scale = abs_max / 127.0
    t_scaled = tensor / scale
    t_int8 = np.round(t_scaled).clip(-128, 127).astype(np.int8)
    return t_int8, float(scale)


def measure_quantization_error(
    original: np.ndarray, quantized: np.ndarray, scale: np.ndarray | float
) -> dict:
    """Measure quantization error (MSE, max abs error, SNR, relative error)."""
    if isinstance(scale, np.ndarray) and scale.ndim == 1:
        reconstructed = quantized.astype(np.float32) * scale[:, np.newaxis]
    else:
        reconstructed = quantized.astype(np.float32) * scale

    error = original - reconstructed
    signal_power = np.mean(original**2)
    noise_power = np.mean(error**2)
    return {
        "mse": float(noise_power),
        "max_abs_error": float(np.abs(error).max()),
        "snr_db": float(10 * np.log10(signal_power / max(noise_power, 1e-10))),
        "relative_error_pct": float(
            100 * np.sqrt(noise_power) / max(np.sqrt(signal_power), 1e-10)
        ),
    }


# ── Checkpoint loading ────────────────────────────────────────────────────────


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint file and return the full checkpoint dict."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Expected 'model_state_dict' key in checkpoint. "
            f"Found keys: {list(ckpt.keys())}"
        )
    return ckpt


def detect_model_type(state_dict: dict[str, torch.Tensor]) -> str:
    """Detect model type from state_dict keys.

    Returns 'mamba2', 'mlp', or 'policy'.
    """
    has_mamba = any("layers.0.mamba" in k for k in state_dict)
    has_trunk = any(k.startswith("trunk.") for k in state_dict)
    has_analog_head = any("analog_head" in k for k in state_dict)

    if has_mamba:
        return "mamba2"
    elif has_trunk and has_analog_head:
        return "policy"
    elif has_trunk:
        return "mlp"
    else:
        raise ValueError(
            f"Cannot detect model type. Keys start with: "
            f"{sorted(set(k.split('.')[0] for k in state_dict))}"
        )


def classify_weight(key: str) -> str:
    """Classify a weight key into a manifest group."""
    if key.startswith("layers."):
        return "layer_weights"
    elif key.endswith("_embed.weight"):
        return "embeddings"
    elif key.endswith("_head.weight") or key.endswith("_head.bias"):
        return "heads"
    else:
        return "projections"


# ── Architecture detection ────────────────────────────────────────────────────


def detect_mamba2_arch(state_dict: dict[str, np.ndarray]) -> dict:
    """Extract Mamba2 architecture params from weight shapes."""
    d_model = state_dict["frame_proj.weight"].shape[0]

    layer_indices = set()
    for k in state_dict:
        if k.startswith("layers."):
            layer_indices.add(int(k.split(".")[1]))
    n_layers = len(layer_indices)

    d_inner = 2 * d_model
    nheads = state_dict["layers.0.mamba.A_log"].shape[0]
    headdim = d_inner // nheads
    conv_dim = state_dict["layers.0.mamba.conv1d.weight"].shape[0]
    d_state = (conv_dim - d_inner) // 2

    return {
        "model_type": "mamba2",
        "d_model": int(d_model),
        "d_inner": int(d_inner),
        "d_state": int(d_state),
        "n_layers": int(n_layers),
        "nheads": int(nheads),
        "headdim": int(headdim),
    }


def detect_mlp_arch(state_dict: dict[str, np.ndarray]) -> dict:
    """Extract MLP architecture params from weight shapes."""
    hidden_dim = state_dict["trunk.0.weight"].shape[0]
    trunk_dim = state_dict["trunk.3.weight"].shape[0]
    return {
        "model_type": "mlp",
        "hidden_dim": int(hidden_dim),
        "trunk_dim": int(trunk_dim),
    }


def detect_policy_arch(state_dict: dict[str, np.ndarray]) -> dict:
    """Extract PolicyMLP architecture params from weight shapes."""
    hidden_dim = state_dict["trunk.0.weight"].shape[0]
    trunk_dim = state_dict["trunk.3.weight"].shape[0]
    return {
        "model_type": "policy",
        "hidden_dim": int(hidden_dim),
        "trunk_dim": int(trunk_dim),
    }


# ── Main quantization pipeline ──────────────────────────────────────────────


def quantize_model(
    state_dict: dict[str, np.ndarray],
) -> tuple[bytes, dict[str, dict]]:
    """Quantize all weights generically by ndim.

    Returns:
        weight_bytes: Packed INT8 weights
        weight_info: Per-key quantization metadata (offset, size, shape, scales, errors)
    """
    weight_buf = bytearray()
    weight_info = {}

    sorted_keys = sorted(state_dict.keys())

    for key in tqdm(sorted_keys, desc="Quantizing weights"):
        w = state_dict[key]
        offset = len(weight_buf)

        if w.ndim == 2:
            w_int8, scales = quantize_per_channel_symmetric(w)
            weight_buf.extend(w_int8.tobytes())
            err = measure_quantization_error(w, w_int8, scales)
            weight_info[key] = {
                "offset": offset,
                "size": w_int8.nbytes,
                "shape": list(w.shape),
                "quantization": "per_channel_symmetric",
                "scales": scales.tolist(),
                "snr_db": err["snr_db"],
                "error": err,
            }

        elif w.ndim == 1:
            w_int8, scale = quantize_per_tensor_symmetric(w)
            weight_buf.extend(w_int8.tobytes())
            err = measure_quantization_error(w, w_int8, scale)
            weight_info[key] = {
                "offset": offset,
                "size": w_int8.nbytes,
                "shape": list(w.shape),
                "quantization": "per_tensor_symmetric",
                "scale": scale,
                "snr_db": err["snr_db"],
                "error": err,
            }

        elif w.ndim == 3:
            # Conv1d: (out_channels, in_channels, kernel_size) → reshape to 2D
            out_ch, in_ch, k = w.shape
            w_2d = w.reshape(out_ch, in_ch * k)
            w_int8, scales = quantize_per_channel_symmetric(w_2d)
            weight_buf.extend(w_int8.tobytes())
            err = measure_quantization_error(w_2d, w_int8, scales)
            weight_info[key] = {
                "offset": offset,
                "size": w_int8.nbytes,
                "shape": list(w.shape),
                "original_shape_3d": list(w.shape),
                "reshaped_2d": [out_ch, in_ch * k],
                "quantization": "per_channel_symmetric",
                "scales": scales.tolist(),
                "snr_db": err["snr_db"],
                "error": err,
            }

        else:
            print(f"  Warning: skipping {key} with ndim={w.ndim}, shape={w.shape}")

    return bytes(weight_buf), weight_info


def build_shard_map(total_bytes: int) -> dict:
    """Split weight bytes into 4096-aligned shards (~10MB Solana account limit)."""
    if total_bytes == 0:
        return {"num_shards": 0, "shards": []}

    # Target ~4MB per shard (safe margin under 10MB account limit)
    target_shard_size = 4 * 1024 * 1024
    num_shards = max(1, (total_bytes + target_shard_size - 1) // target_shard_size)

    shards = []
    for i in range(num_shards):
        if i < num_shards - 1:
            boundary = ((i + 1) * total_bytes // num_shards + 4095) & ~4095
        else:
            boundary = total_bytes

        start = shards[-1]["offset"] + shards[-1]["size"] if shards else 0
        shards.append({"index": i, "offset": start, "size": boundary - start})

    return {"num_shards": num_shards, "shards": shards}


def build_manifest(
    arch: dict,
    weight_info: dict[str, dict],
    total_bytes: int,
    encoding_config: dict | None,
    context_len: int | None,
) -> dict:
    """Build the manifest.json structure."""
    # Group weights by category
    groups = {"layer_weights": {}, "embeddings": {}, "projections": {}, "heads": {}}
    for key, info in weight_info.items():
        group = classify_weight(key)
        groups[group][key] = info

    # Collect SNR stats from 2D weights only (meaningful for matmul quality)
    snrs = [
        info["snr_db"]
        for info in weight_info.values()
        if "snr_db" in info and len(info.get("shape", [])) >= 2
    ]

    shard_map = build_shard_map(total_bytes)

    manifest = {
        "format": f"{arch['model_type']}_int8_v1",
        "architecture": arch,
        "total_weight_bytes": total_bytes,
        "shard_map": shard_map,
        "weights": groups,
        "error_summary": {
            "num_2d_weights": len(snrs),
            "mean_snr_db": float(np.mean(snrs)) if snrs else 0,
            "min_snr_db": float(np.min(snrs)) if snrs else 0,
            "max_snr_db": float(np.max(snrs)) if snrs else 0,
        },
    }

    if encoding_config is not None:
        manifest["encoding_config"] = encoding_config
    if context_len is not None:
        manifest["context_len"] = context_len

    return manifest


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Quantize model weights to INT8 for onchain inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantization/output"),
        help="Output directory (default: quantization/output)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    state_dict_tensors = ckpt["model_state_dict"]

    # Convert to numpy
    state_dict = {k: v.numpy() for k, v in state_dict_tensors.items()}

    # ── Detect model type and architecture ───────────────────────────
    model_type = detect_model_type(state_dict_tensors)
    print(f"Detected model type: {model_type}")

    if model_type == "mamba2":
        arch = detect_mamba2_arch(state_dict)
    elif model_type == "policy":
        arch = detect_policy_arch(state_dict)
    else:
        arch = detect_mlp_arch(state_dict)

    print(f"Architecture: {json.dumps(arch, indent=2)}")

    # ── Extract encoding_config and context_len ──────────────────────
    encoding_config = None
    context_len = None

    if "encoding_config" in ckpt and ckpt["encoding_config"]:
        encoding_config = ckpt["encoding_config"]
        # Ensure it's a plain dict (not a dataclass)
        if hasattr(encoding_config, "__dataclass_fields__"):
            encoding_config = asdict(encoding_config)
        context_len = ckpt.get("context_len")
        print(f"Encoding config found, context_len={context_len}")
    elif "config" in ckpt and ckpt["config"]:
        old_cfg = dict(ckpt["config"])
        context_len = old_cfg.pop("context_len", None)
        encoding_config = old_cfg
        print(f"Legacy config found, context_len={context_len}")

    # ── Print weight summary ─────────────────────────────────────────
    total_params = sum(v.size for v in state_dict.values())
    total_fp32 = sum(v.nbytes for v in state_dict.values())
    print(f"\nWeight summary:")
    print(f"  Parameters: {total_params:,}")
    print(f"  FP32 size:  {total_fp32:,} bytes ({total_fp32 / 1024 / 1024:.1f} MB)")
    print(f"  Keys:       {len(state_dict)}")

    # List all weights
    print(f"\n  Weight keys ({len(state_dict)}):")
    for key in sorted(state_dict.keys()):
        w = state_dict[key]
        print(f"    {key:50s} {str(w.shape):20s} {w.dtype}")

    # ── Quantize ─────────────────────────────────────────────────────
    print()
    weight_bytes, weight_info = quantize_model(state_dict)

    # ── Build manifest ───────────────────────────────────────────────
    manifest = build_manifest(
        arch, weight_info, len(weight_bytes), encoding_config, context_len
    )

    # ── Write outputs ────────────────────────────────────────────────
    weights_path = args.output_dir / "weights_int8.bin"
    with open(weights_path, "wb") as f:
        f.write(weight_bytes)

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    int8_bytes = len(weight_bytes)
    compression = total_fp32 / int8_bytes if int8_bytes > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Quantization complete!")
    print(f"  Model:       {model_type}")
    print(f"  INT8 output: {weights_path} ({int8_bytes:,} bytes, {int8_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Manifest:    {manifest_path}")
    print(f"  Compression: {compression:.1f}x ({total_fp32 / 1024 / 1024:.1f} MB → {int8_bytes / 1024 / 1024:.1f} MB)")

    # Shard layout
    shard_map = manifest["shard_map"]
    print(f"\n  Shard layout ({shard_map['num_shards']} shards):")
    for shard in shard_map["shards"]:
        print(
            f"    Shard {shard['index']}: "
            f"offset={shard['offset']:,}, "
            f"size={shard['size']:,} bytes "
            f"({shard['size'] / 1024 / 1024:.2f} MB)"
        )

    # Error summary
    es = manifest["error_summary"]
    if es["num_2d_weights"] > 0:
        print(f"\n  Quantization quality ({es['num_2d_weights']} matrix weights):")
        print(f"    Mean SNR: {es['mean_snr_db']:.1f} dB")
        print(f"    Min SNR:  {es['min_snr_db']:.1f} dB")
        print(f"    Max SNR:  {es['max_snr_db']:.1f} dB")

    # Verify shard map covers all bytes with no gaps
    if shard_map["shards"]:
        total_shard_bytes = sum(s["size"] for s in shard_map["shards"])
        assert total_shard_bytes == int8_bytes, (
            f"Shard map gap: shards cover {total_shard_bytes} but total is {int8_bytes}"
        )
        for i, s in enumerate(shard_map["shards"]):
            if i > 0:
                prev = shard_map["shards"][i - 1]
                assert s["offset"] == prev["offset"] + prev["size"], (
                    f"Gap between shard {i-1} and {i}"
                )
        print(f"\n  Shard map verified: no gaps, covers all {int8_bytes:,} bytes")

    # Solana rent estimate
    rent_per_byte = 0.00000696
    rent_sol = int8_bytes * rent_per_byte
    print(f"  Estimated Solana rent: {rent_sol:.2f} SOL")


if __name__ == "__main__":
    main()
