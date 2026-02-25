#!/usr/bin/env python3
"""Generate INT8 lookup tables for activation functions used in Mamba2 inference.

Produces 256-entry LUTs for:
  - SiLU(x) = x * sigmoid(x)   — gate activation
  - softplus(x) = ln(1 + exp(x)) — dt computation in selective scan
  - rsqrt(x) = 1/sqrt(x)       — RMSNorm
  - exp_neg(x) = exp(-x)        — A_bar in selective scan

Each LUT maps an INT8 input (-128..127) to an INT8 output, using configurable
input/output scales. The LUTs are stored as 256-byte arrays.

Output: luts.bin (binary, 4 × 256 = 1024 bytes) + luts.json (metadata)
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def compute_silu_lut(input_scale: float, output_scale: float) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))"""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        # Map table index to signed INT8 value
        int8_val = np.int8(i if i < 128 else i - 256)
        # Dequantize to float
        x = float(int8_val) * input_scale
        # Compute SiLU
        sigmoid_x = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        silu_x = x * sigmoid_x
        # Quantize output
        quantized = int(np.round(silu_x / output_scale))
        lut[i] = np.clip(quantized, -128, 127).astype(np.int8)
    return lut


def compute_softplus_lut(input_scale: float, output_scale: float) -> np.ndarray:
    """softplus(x) = ln(1 + exp(x))"""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        int8_val = np.int8(i if i < 128 else i - 256)
        x = float(int8_val) * input_scale
        # Numerically stable softplus
        if x > 20:
            sp = x
        elif x < -20:
            sp = 0.0
        else:
            sp = np.log(1.0 + np.exp(x))
        quantized = int(np.round(sp / output_scale))
        lut[i] = np.clip(quantized, -128, 127).astype(np.int8)
    return lut


def compute_rsqrt_lut(input_scale: float, output_scale: float) -> np.ndarray:
    """rsqrt(x) = 1/sqrt(x) for RMSNorm.

    Input is the mean squared value (always positive), so we map
    0..255 to positive range only, using unsigned interpretation.
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Unsigned: map 0..255 to small positive range
        # Avoid division by zero: minimum input maps to max output
        x = max(float(i) * input_scale, 1e-6)
        rsqrt = 1.0 / np.sqrt(x)
        quantized = int(np.round(rsqrt / output_scale))
        lut[i] = np.clip(quantized, 0, 255).astype(np.uint8)
    return lut


def compute_exp_neg_lut(input_scale: float, output_scale: float) -> np.ndarray:
    """exp(-x) for computing A_bar = exp(-dt * A) in selective scan.

    Input is dt*A product (positive), output is decay factor (0..1).
    Both unsigned interpretation.
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        x = float(i) * input_scale  # Unsigned: 0..255
        exp_neg = np.exp(-x)
        quantized = int(np.round(exp_neg / output_scale))
        lut[i] = np.clip(quantized, 0, 255).astype(np.uint8)
    return lut


def main():
    parser = argparse.ArgumentParser(description="Generate INT8 activation LUTs for Mamba2")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("quantization/output"),
        help="Output directory for LUT files",
    )
    parser.add_argument(
        "--silu-input-scale", type=float, default=0.0625,
        help="Input dequantization scale for SiLU (default: 1/16)",
    )
    parser.add_argument(
        "--silu-output-scale", type=float, default=0.0625,
        help="Output quantization scale for SiLU",
    )
    parser.add_argument(
        "--softplus-input-scale", type=float, default=0.0625,
        help="Input dequantization scale for softplus",
    )
    parser.add_argument(
        "--softplus-output-scale", type=float, default=0.03125,
        help="Output quantization scale for softplus (1/32, always positive)",
    )
    parser.add_argument(
        "--rsqrt-input-scale", type=float, default=0.01,
        help="Input scale for rsqrt (maps 0..255 to 0..2.55)",
    )
    parser.add_argument(
        "--rsqrt-output-scale", type=float, default=0.05,
        help="Output scale for rsqrt",
    )
    parser.add_argument(
        "--exp-input-scale", type=float, default=0.03125,
        help="Input scale for exp(-x) (maps 0..255 to 0..8)",
    )
    parser.add_argument(
        "--exp-output-scale", type=float, default=0.00392157,
        help="Output scale for exp(-x) (maps 0..1 to 0..255, ~1/255)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating activation LUTs for Mamba2 INT8 inference...")

    # Generate all LUTs
    silu_lut = compute_silu_lut(args.silu_input_scale, args.silu_output_scale)
    softplus_lut = compute_softplus_lut(args.softplus_input_scale, args.softplus_output_scale)
    rsqrt_lut = compute_rsqrt_lut(args.rsqrt_input_scale, args.rsqrt_output_scale)
    exp_neg_lut = compute_exp_neg_lut(args.exp_input_scale, args.exp_output_scale)

    # Print sample values for verification
    print("\nSiLU LUT samples (index → output):")
    for idx in [0, 64, 128, 192, 255]:
        int8_in = np.int8(idx if idx < 128 else idx - 256)
        x = float(int8_in) * args.silu_input_scale
        out = np.int8(silu_lut[idx] if silu_lut[idx] < 128 else silu_lut[idx] - 256)
        print(f"  [{idx:3d}] int8={int8_in:+4d} → x={x:+7.3f} → SiLU={x/(1+np.exp(-x)):+7.3f} → int8={out:+4d}")

    print("\nSoftplus LUT samples:")
    for idx in [0, 64, 128, 192, 255]:
        int8_in = np.int8(idx if idx < 128 else idx - 256)
        x = float(int8_in) * args.softplus_input_scale
        sp = np.log(1 + np.exp(np.clip(x, -20, 20)))
        out = np.int8(softplus_lut[idx] if softplus_lut[idx] < 128 else softplus_lut[idx] - 256)
        print(f"  [{idx:3d}] int8={int8_in:+4d} → x={x:+7.3f} → softplus={sp:+7.3f} → int8={out:+4d}")

    # Write binary file: 4 LUTs × 256 bytes = 1024 bytes
    bin_path = args.output_dir / "luts.bin"
    with open(bin_path, "wb") as f:
        f.write(silu_lut.tobytes())
        f.write(softplus_lut.tobytes())
        f.write(rsqrt_lut.view(np.uint8).tobytes())
        f.write(exp_neg_lut.view(np.uint8).tobytes())

    print(f"\nBinary LUTs written to {bin_path} ({bin_path.stat().st_size} bytes)")

    # Write metadata JSON
    metadata = {
        "format": "int8_luts_v1",
        "num_luts": 4,
        "lut_size": 256,
        "total_bytes": 1024,
        "luts": [
            {
                "name": "silu",
                "offset": 0,
                "size": 256,
                "input_signed": True,
                "output_signed": True,
                "input_scale": args.silu_input_scale,
                "output_scale": args.silu_output_scale,
                "description": "SiLU(x) = x * sigmoid(x), gate activation",
            },
            {
                "name": "softplus",
                "offset": 256,
                "size": 256,
                "input_signed": True,
                "output_signed": True,
                "input_scale": args.softplus_input_scale,
                "output_scale": args.softplus_output_scale,
                "description": "softplus(x) = ln(1 + exp(x)), dt computation",
            },
            {
                "name": "rsqrt",
                "offset": 512,
                "size": 256,
                "input_signed": False,
                "output_signed": False,
                "input_scale": args.rsqrt_input_scale,
                "output_scale": args.rsqrt_output_scale,
                "description": "1/sqrt(x), RMSNorm normalization",
            },
            {
                "name": "exp_neg",
                "offset": 768,
                "size": 256,
                "input_signed": False,
                "output_signed": False,
                "input_scale": args.exp_input_scale,
                "output_scale": args.exp_output_scale,
                "description": "exp(-x), A_bar decay in selective scan",
            },
        ],
    }

    json_path = args.output_dir / "luts.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"LUT metadata written to {json_path}")

    # Verification: check monotonicity and range
    print("\nVerification:")
    print(f"  SiLU range: [{silu_lut.min()}, {silu_lut.max()}]")
    print(f"  Softplus range: [{softplus_lut.min()}, {softplus_lut.max()}]")
    print(f"  rsqrt range: [{rsqrt_lut.min()}, {rsqrt_lut.max()}]")
    print(f"  exp_neg range: [{exp_neg_lut.min()}, {exp_neg_lut.max()}]")

    # Check softplus is monotonically non-decreasing (signed interpretation)
    sp_signed = softplus_lut.view(np.int8)
    diffs = np.diff(sp_signed.astype(np.int16))
    if np.all(diffs >= 0):
        print("  ✓ Softplus is monotonically non-decreasing")
    else:
        num_violations = np.sum(diffs < 0)
        print(f"  ✗ Softplus monotonicity violations: {num_violations}")

    # Check exp_neg is monotonically non-increasing
    diffs = np.diff(exp_neg_lut.astype(np.int16))
    if np.all(diffs <= 0):
        print("  ✓ exp(-x) is monotonically non-increasing")
    else:
        num_violations = np.sum(diffs > 0)
        print(f"  ✗ exp(-x) monotonicity violations: {num_violations}")


if __name__ == "__main__":
    main()
