#!/usr/bin/env python3
"""Quantize Mamba2 FP32 weights to INT8 for onchain inference.

Takes trained Mamba2 weights (float32, from nojohns-training) and produces:
  - weights_int8.bin: Packed INT8 weights for all layers, ready for upload
  - manifest.json: Architecture params, per-layer scale/zero-point, shard map

Quantization strategy:
  - Per-channel symmetric quantization for weight matrices
  - Per-tensor quantization for biases and small vectors
  - INT8 × INT8 → accumulate INT32 → requantize per-layer

The manifest.json contains everything the onchain ModelManifest account needs:
  - Architecture: d_model, d_inner, d_state, num_layers
  - Per-layer: scale factors, zero points (for requantization between layers)
  - Shard map: byte offsets for each weight matrix within shards
  - LUT references

Usage:
    python quantize_mamba2.py --weights path/to/model.safetensors --output-dir quantization/output
    python quantize_mamba2.py --weights path/to/model.pt --output-dir quantization/output
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

from tqdm import tqdm


# ── Quantization core ────────────────────────────────────────────────────────


def quantize_per_channel_symmetric(
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric quantization.

    For a weight matrix W of shape (out_features, in_features),
    compute per-output-channel scale factors such that:
        W_int8[c, :] = round(W[c, :] / scale[c])
        W_approx[c, :] = W_int8[c, :] * scale[c]

    Returns:
        w_int8: INT8 quantized weights, shape (out_features, in_features)
        scales: Per-channel scale factors, shape (out_features,)
    """
    assert weight.ndim == 2, f"Expected 2D weight, got shape {weight.shape}"

    # Per-channel: find max absolute value per output channel
    abs_max = np.abs(weight).max(axis=1)  # shape: (out_features,)
    abs_max = np.maximum(abs_max, 1e-8)  # Avoid division by zero

    # Scale = abs_max / 127 (symmetric: range is [-127, 127])
    scales = abs_max / 127.0

    # Quantize
    w_scaled = weight / scales[:, np.newaxis]
    w_int8 = np.round(w_scaled).clip(-128, 127).astype(np.int8)

    return w_int8, scales.astype(np.float32)


def quantize_per_tensor_symmetric(
    tensor: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric quantization for vectors and small tensors."""
    abs_max = max(np.abs(tensor).max(), 1e-8)
    scale = abs_max / 127.0
    t_scaled = tensor / scale
    t_int8 = np.round(t_scaled).clip(-128, 127).astype(np.int8)
    return t_int8, float(scale)


def measure_quantization_error(
    original: np.ndarray, quantized: np.ndarray, scale: np.ndarray | float
) -> dict:
    """Measure quantization error metrics."""
    if isinstance(scale, np.ndarray) and scale.ndim == 1:
        # Per-channel: reconstruct
        reconstructed = quantized.astype(np.float32) * scale[:, np.newaxis]
    else:
        reconstructed = quantized.astype(np.float32) * scale

    error = original - reconstructed
    return {
        "mse": float(np.mean(error**2)),
        "max_abs_error": float(np.abs(error).max()),
        "snr_db": float(
            10 * np.log10(np.mean(original**2) / max(np.mean(error**2), 1e-10))
        ),
        "relative_error_pct": float(
            100 * np.sqrt(np.mean(error**2)) / max(np.sqrt(np.mean(original**2)), 1e-10)
        ),
    }


# ── Mamba2 weight extraction ────────────────────────────────────────────────


# Expected Mamba2 weight names (from mamba_ssm / nojohns model)
MAMBA2_WEIGHT_PATTERNS = {
    "in_proj": "layers.{layer}.mixer.in_proj.weight",
    "out_proj": "layers.{layer}.mixer.out_proj.weight",
    "A_log": "layers.{layer}.mixer.A_log",
    "D": "layers.{layer}.mixer.D",
    "dt_bias": "layers.{layer}.mixer.dt_bias",
    "norm": "layers.{layer}.norm.weight",
    # Conv1d for Mamba (not Mamba2 recurrent-only mode)
    "conv1d_weight": "layers.{layer}.mixer.conv1d.weight",
    "conv1d_bias": "layers.{layer}.mixer.conv1d.bias",
}

# Embedding and output head
MAMBA2_GLOBAL_WEIGHTS = {
    "embedding": "backbone.embedding.weight",
    "norm_f": "backbone.norm_f.weight",
    # Output heads for world model
    "head_continuous": "head_continuous.weight",
    "head_continuous_bias": "head_continuous.bias",
    "head_action_state": "head_action_state.weight",
    "head_action_state_bias": "head_action_state.bias",
    "head_binary": "head_binary.weight",
    "head_binary_bias": "head_binary.bias",
}


def load_weights(path: Path) -> dict[str, np.ndarray]:
    """Load model weights from safetensors or PyTorch checkpoint."""
    if path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("safetensors package required for .safetensors files")
        state_dict = load_safetensors(str(path))
        return {k: v.numpy() for k, v in state_dict.items()}
    elif path.suffix in (".pt", ".pth", ".bin"):
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return {k: v.numpy() for k, v in state_dict.items()}
    else:
        raise ValueError(f"Unsupported weight format: {path.suffix}")


def detect_architecture(state_dict: dict[str, np.ndarray]) -> dict:
    """Auto-detect Mamba2 architecture from weight shapes."""
    # Count layers
    layer_indices = set()
    for key in state_dict:
        if "layers." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    num_layers = len(layer_indices) if layer_indices else 0

    # Detect d_model from norm weight
    d_model = None
    for key, val in state_dict.items():
        if "norm" in key and "weight" in key and val.ndim == 1:
            d_model = val.shape[0]
            break

    # Detect d_inner from in_proj
    d_inner = None
    for key, val in state_dict.items():
        if "in_proj" in key and "weight" in key and val.ndim == 2:
            # in_proj maps d_model → d_inner*2 + extras
            # The exact split depends on Mamba2 config
            d_inner = val.shape[0] // 2  # Approximate
            break

    # Detect d_state from A_log
    d_state = None
    for key, val in state_dict.items():
        if "A_log" in key:
            if val.ndim == 1:
                d_state = 1  # Scalar per head
            elif val.ndim == 2:
                d_state = val.shape[1]
            break

    return {
        "num_layers": num_layers,
        "d_model": d_model or 512,
        "d_inner": d_inner or 1024,
        "d_state": d_state or 16,
    }


# ── Main quantization pipeline ──────────────────────────────────────────────


def quantize_model(
    state_dict: dict[str, np.ndarray],
    arch: dict,
) -> tuple[bytes, dict]:
    """Quantize all model weights and produce binary + manifest.

    Returns:
        weight_bytes: Packed INT8 weights (ready for upload to WeightShard accounts)
        manifest: Dict containing architecture, per-layer quant params, shard map
    """
    num_layers = arch["num_layers"]
    d_model = arch["d_model"]
    d_inner = arch["d_inner"]

    weight_buf = bytearray()
    layer_params = []
    global_params = {}
    errors = {}

    print(f"\nQuantizing {num_layers} layers (d_model={d_model}, d_inner={d_inner})...")

    # ── Per-layer weights ────────────────────────────────────────────────

    for layer_idx in tqdm(range(num_layers), desc="Quantizing layers"):
        layer_info = {"layer": layer_idx, "weights": {}}

        for weight_name, pattern in MAMBA2_WEIGHT_PATTERNS.items():
            key = pattern.format(layer=layer_idx)
            if key not in state_dict:
                continue

            w = state_dict[key]
            offset = len(weight_buf)

            if w.ndim == 2:
                # Matrix: per-channel quantization
                w_int8, scales = quantize_per_channel_symmetric(w)
                weight_buf.extend(w_int8.tobytes())

                err = measure_quantization_error(w, w_int8, scales)
                errors[f"layer{layer_idx}.{weight_name}"] = err

                layer_info["weights"][weight_name] = {
                    "offset": offset,
                    "size": w_int8.nbytes,
                    "shape": list(w.shape),
                    "quantization": "per_channel_symmetric",
                    "scales": scales.tolist(),
                    "snr_db": err["snr_db"],
                }
            elif w.ndim == 1:
                # Vector: per-tensor quantization
                w_int8, scale = quantize_per_tensor_symmetric(w)
                weight_buf.extend(w_int8.tobytes())

                layer_info["weights"][weight_name] = {
                    "offset": offset,
                    "size": w_int8.nbytes,
                    "shape": list(w.shape),
                    "quantization": "per_tensor_symmetric",
                    "scale": scale,
                }
            elif w.ndim == 3:
                # Conv1d: (out_channels, in_channels, kernel_size)
                # Reshape to 2D for quantization, then pack
                out_ch, in_ch, k = w.shape
                w_2d = w.reshape(out_ch, in_ch * k)
                w_int8, scales = quantize_per_channel_symmetric(w_2d)
                weight_buf.extend(w_int8.tobytes())

                layer_info["weights"][weight_name] = {
                    "offset": offset,
                    "size": w_int8.nbytes,
                    "shape": list(w.shape),
                    "quantization": "per_channel_symmetric",
                    "scales": scales.tolist(),
                }

        layer_params.append(layer_info)

    # ── Global weights (embedding, heads) ────────────────────────────────

    print("Quantizing global weights...")
    for weight_name, key in MAMBA2_GLOBAL_WEIGHTS.items():
        if key not in state_dict:
            # Try common alternatives
            alternatives = [
                key.replace("backbone.", ""),
                key.replace("backbone.", "model."),
            ]
            found = False
            for alt in alternatives:
                if alt in state_dict:
                    key = alt
                    found = True
                    break
            if not found:
                print(f"  Skipping {weight_name} (not found: {key})")
                continue

        w = state_dict[key]
        offset = len(weight_buf)

        if w.ndim == 2:
            w_int8, scales = quantize_per_channel_symmetric(w)
            weight_buf.extend(w_int8.tobytes())

            err = measure_quantization_error(w, w_int8, scales)
            errors[weight_name] = err

            global_params[weight_name] = {
                "offset": offset,
                "size": w_int8.nbytes,
                "shape": list(w.shape),
                "quantization": "per_channel_symmetric",
                "scales": scales.tolist(),
                "snr_db": err["snr_db"],
            }
        elif w.ndim == 1:
            w_int8, scale = quantize_per_tensor_symmetric(w)
            weight_buf.extend(w_int8.tobytes())

            global_params[weight_name] = {
                "offset": offset,
                "size": w_int8.nbytes,
                "shape": list(w.shape),
                "quantization": "per_tensor_symmetric",
                "scale": scale,
            }

    # ── Shard map ────────────────────────────────────────────────────────

    total_bytes = len(weight_buf)
    # Split into 2 shards (Solana account size limit ~10MB with realloc)
    shard_boundary = total_bytes // 2
    # Align to 4096 bytes
    shard_boundary = (shard_boundary + 4095) & ~4095

    shard_map = {
        "num_shards": 2,
        "shards": [
            {"index": 0, "offset": 0, "size": shard_boundary},
            {"index": 1, "offset": shard_boundary, "size": total_bytes - shard_boundary},
        ],
    }

    # ── Manifest ─────────────────────────────────────────────────────────

    manifest = {
        "format": "mamba2_int8_v1",
        "architecture": arch,
        "total_weight_bytes": total_bytes,
        "shard_map": shard_map,
        "layers": layer_params,
        "global_weights": global_params,
        "quantization_errors": errors,
    }

    return bytes(weight_buf), manifest


# ── Generate dummy weights for testing ───────────────────────────────────────


def generate_dummy_weights(
    d_model: int = 512,
    d_inner: int = 1024,
    d_state: int = 16,
    num_layers: int = 12,
) -> dict[str, np.ndarray]:
    """Generate random FP32 weights matching Mamba2 architecture for testing."""
    rng = np.random.default_rng(42)
    state_dict = {}

    for layer in range(num_layers):
        prefix = f"layers.{layer}"
        # in_proj: d_model → 2*d_inner (z + x_ssm)
        state_dict[f"{prefix}.mixer.in_proj.weight"] = rng.standard_normal(
            (2 * d_inner, d_model)
        ).astype(np.float32) * 0.02

        # out_proj: d_inner → d_model
        state_dict[f"{prefix}.mixer.out_proj.weight"] = rng.standard_normal(
            (d_model, d_inner)
        ).astype(np.float32) * 0.02

        # A_log: d_inner
        state_dict[f"{prefix}.mixer.A_log"] = rng.standard_normal(
            d_inner
        ).astype(np.float32)

        # D: d_inner
        state_dict[f"{prefix}.mixer.D"] = rng.standard_normal(
            d_inner
        ).astype(np.float32) * 0.1

        # dt_bias: d_inner
        state_dict[f"{prefix}.mixer.dt_bias"] = rng.standard_normal(
            d_inner
        ).astype(np.float32) * 0.1

        # norm: d_model
        state_dict[f"{prefix}.norm.weight"] = np.ones(
            d_model, dtype=np.float32
        ) + rng.standard_normal(d_model).astype(np.float32) * 0.01

    # Global weights
    vocab_size = 64  # Small for world model (encoded game state, not text)
    state_dict["backbone.embedding.weight"] = rng.standard_normal(
        (vocab_size, d_model)
    ).astype(np.float32) * 0.02
    state_dict["backbone.norm_f.weight"] = np.ones(
        d_model, dtype=np.float32
    )

    # Output heads
    num_continuous = 12  # x, y, percent, shield, speeds, state_age, hitlag, stocks
    num_action_states = 400
    num_binary = 2  # facing, on_ground

    state_dict["head_continuous.weight"] = rng.standard_normal(
        (num_continuous * 2, d_model)  # *2 for both players
    ).astype(np.float32) * 0.02
    state_dict["head_continuous.bias"] = np.zeros(
        num_continuous * 2, dtype=np.float32
    )
    state_dict["head_action_state.weight"] = rng.standard_normal(
        (num_action_states * 2, d_model)
    ).astype(np.float32) * 0.02
    state_dict["head_action_state.bias"] = np.zeros(
        num_action_states * 2, dtype=np.float32
    )
    state_dict["head_binary.weight"] = rng.standard_normal(
        (num_binary * 2, d_model)
    ).astype(np.float32) * 0.02
    state_dict["head_binary.bias"] = np.zeros(
        num_binary * 2, dtype=np.float32
    )

    return state_dict


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Mamba2 weights to INT8 for onchain inference"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to FP32 weights (.safetensors, .pt, .pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantization/output"),
        help="Output directory",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use randomly generated dummy weights for testing",
    )
    parser.add_argument(
        "--d-model", type=int, default=512,
        help="Model dimension (default: 512)",
    )
    parser.add_argument(
        "--d-inner", type=int, default=1024,
        help="Inner dimension (default: 1024, typically 2×d_model)",
    )
    parser.add_argument(
        "--d-state", type=int, default=16,
        help="SSM state dimension (default: 16)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=12,
        help="Number of Mamba2 layers (default: 12)",
    )
    args = parser.parse_args()

    if not args.weights and not args.dummy:
        print("No weights specified. Use --weights <path> or --dummy for testing.")
        print("Generating dummy weights for demonstration...\n")
        args.dummy = True

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load weights
    if args.dummy:
        print(f"Generating dummy Mamba2 weights:")
        print(f"  d_model={args.d_model}, d_inner={args.d_inner}")
        print(f"  d_state={args.d_state}, num_layers={args.num_layers}")
        state_dict = generate_dummy_weights(
            d_model=args.d_model,
            d_inner=args.d_inner,
            d_state=args.d_state,
            num_layers=args.num_layers,
        )
        arch = {
            "num_layers": args.num_layers,
            "d_model": args.d_model,
            "d_inner": args.d_inner,
            "d_state": args.d_state,
        }
    else:
        print(f"Loading weights from {args.weights}...")
        state_dict = load_weights(args.weights)
        arch = detect_architecture(state_dict)
        print(f"Detected architecture: {arch}")

    # Print weight summary
    total_fp32 = sum(v.nbytes for v in state_dict.values())
    total_params = sum(v.size for v in state_dict.values())
    print(f"\nWeight summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  FP32 size: {total_fp32:,} bytes ({total_fp32/1024/1024:.1f} MB)")
    print(f"  Expected INT8 size: ~{total_params:,} bytes ({total_params/1024/1024:.1f} MB)")

    # Quantize
    weight_bytes, manifest = quantize_model(state_dict, arch)

    # Write outputs
    weights_path = args.output_dir / "weights_int8.bin"
    with open(weights_path, "wb") as f:
        f.write(weight_bytes)

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Quantization complete!")
    print(f"  INT8 weights: {weights_path} ({len(weight_bytes):,} bytes, {len(weight_bytes)/1024/1024:.1f} MB)")
    print(f"  Manifest: {manifest_path}")
    print(f"  Compression ratio: {total_fp32/len(weight_bytes):.1f}x")

    shard_map = manifest["shard_map"]
    print(f"\n  Shard layout ({shard_map['num_shards']} shards):")
    for shard in shard_map["shards"]:
        print(f"    Shard {shard['index']}: offset={shard['offset']:,}, size={shard['size']:,} bytes ({shard['size']/1024/1024:.1f} MB)")

    # Print quantization error summary
    if manifest.get("quantization_errors"):
        print(f"\n  Quantization error summary:")
        snrs = []
        for name, err in manifest["quantization_errors"].items():
            snrs.append(err["snr_db"])
        print(f"    Mean SNR: {np.mean(snrs):.1f} dB")
        print(f"    Min SNR:  {np.min(snrs):.1f} dB")
        print(f"    Max SNR:  {np.max(snrs):.1f} dB")

    # Solana cost estimate
    rent_per_byte = 0.00000696  # ~6.96 lamports per byte for rent exemption
    sol_per_lamport = 1e-9
    rent_sol = len(weight_bytes) * rent_per_byte
    print(f"\n  Estimated Solana rent deposit: {rent_sol:.1f} SOL")


if __name__ == "__main__":
    main()
