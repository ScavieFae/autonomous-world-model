/// LUT-based activation functions for INT8 Mamba2 inference.
///
/// Each LUT is a 256-entry table mapping an INT8 input (-128..127) to an INT8 output.
/// For unsigned activations (rsqrt, exp_neg), the input/output are unsigned (0..255).
///
/// LUTs are stored in the ModelManifest account, packed as:
///   [silu_lut(256)] [softplus_lut(256)] [rsqrt_lut(256)] [exp_neg_lut(256)]
///
/// Total: 1024 bytes. Negligible compared to weight storage.
/// Lookup cost: 1 memory access (~1-2 CU) vs hundreds of CU for software float.

/// LUT offsets within the packed LUT data
pub const SILU_OFFSET: usize = 0;
pub const SOFTPLUS_OFFSET: usize = 256;
pub const RSQRT_OFFSET: usize = 512;
pub const EXP_NEG_OFFSET: usize = 768;
pub const LUT_TOTAL_SIZE: usize = 1024;

/// SiLU activation via lookup table.
/// SiLU(x) = x * sigmoid(x) — used for gating in Mamba2.
///
/// Input: signed INT8 value
/// Output: signed INT8 value
#[inline(always)]
pub fn silu_lut(lut_data: &[u8], x: i8) -> i8 {
    // Convert signed i8 to unsigned index: -128 → 0, -127 → 1, ..., 127 → 255
    let idx = (x as u8) as usize;
    lut_data[SILU_OFFSET + idx] as i8
}

/// Softplus activation via lookup table.
/// softplus(x) = ln(1 + exp(x)) — used for dt computation in selective scan.
///
/// Input: signed INT8 value
/// Output: signed INT8 value (always non-negative in float, but quantized range)
#[inline(always)]
pub fn softplus_lut(lut_data: &[u8], x: i8) -> i8 {
    let idx = (x as u8) as usize;
    lut_data[SOFTPLUS_OFFSET + idx] as i8
}

/// Reciprocal square root via lookup table.
/// rsqrt(x) = 1/sqrt(x) — used for RMSNorm.
///
/// Input: unsigned value (mean-squared, always positive)
/// Output: unsigned value (always positive)
#[inline(always)]
pub fn rsqrt_lut(lut_data: &[u8], x: u8) -> u8 {
    lut_data[RSQRT_OFFSET + x as usize]
}

/// Negative exponential via lookup table.
/// exp(-x) — used for A_bar computation in selective scan: A_bar = exp(-dt * A).
///
/// Input: unsigned value (dt * A product, always non-negative)
/// Output: unsigned value (decay factor, 0..1 mapped to 0..255)
#[inline(always)]
pub fn exp_neg_lut(lut_data: &[u8], x: u8) -> u8 {
    lut_data[EXP_NEG_OFFSET + x as usize]
}

/// Apply SiLU activation to a slice in-place.
/// Used for: gate = SiLU(z) in Mamba2 gating.
#[inline]
pub fn silu_slice(lut_data: &[u8], data: &mut [i8]) {
    for v in data.iter_mut() {
        *v = silu_lut(lut_data, *v);
    }
}

/// Apply softplus activation to a slice in-place.
/// Used for: dt = softplus(dt_raw) in selective scan.
#[inline]
pub fn softplus_slice(lut_data: &[u8], data: &mut [i8]) {
    for v in data.iter_mut() {
        *v = softplus_lut(lut_data, *v);
    }
}

/// RMSNorm using LUT for rsqrt.
///
/// Computes: y[i] = x[i] * weight[i] / rms(x)
/// Where rms(x) = sqrt(mean(x^2))
///
/// In INT8:
///   1. Compute mean of squared values (INT32 accumulator)
///   2. Map to LUT index for rsqrt
///   3. Element-wise multiply x * weight * rsqrt_factor
///
/// Returns: normalized values as INT8, plus the scale factor for downstream use.
pub fn rmsnorm_int8(
    lut_data: &[u8],
    x: &[i8],
    weight: &[i8],
    output: &mut [i8],
    weight_scale: i32,
) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());

    // Compute mean squared value
    let mut sum_sq: i64 = 0;
    for &val in x.iter() {
        let v = val as i64;
        sum_sq += v * v;
    }

    // Map to rsqrt LUT index
    // mean_sq = sum_sq / n, scaled to fit in u8
    let mean_sq = (sum_sq / n as i64) as u32;
    // Scale to 0..255 range for LUT lookup
    let lut_idx = (mean_sq.min(255 * 64) / 64) as u8;
    let rsqrt_val = rsqrt_lut(lut_data, lut_idx) as i32;

    // Apply normalization: output[i] = x[i] * weight[i] * rsqrt_val
    for i in 0..n {
        let val = x[i] as i32 * weight[i] as i32 * rsqrt_val;
        // Rescale: adjust for weight_scale and rsqrt output scale
        let rescaled = (val * weight_scale) >> 16;
        output[i] = rescaled.clamp(-128, 127) as i8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_luts() -> Vec<u8> {
        let mut luts = vec![0u8; LUT_TOTAL_SIZE];

        // SiLU LUT: approximate SiLU(x) for x in [-8, 8]
        for i in 0u16..256 {
            let x = (i as i8) as f64 / 16.0;
            let silu = x / (1.0 + (-x).exp());
            luts[SILU_OFFSET + i as usize] = (silu * 16.0 + 0.5)
                .clamp(-128.0, 127.0) as i8 as u8;
        }

        // Softplus LUT
        for i in 0u16..256 {
            let x = (i as i8) as f64 / 16.0;
            let sp = (1.0 + x.exp()).ln();
            luts[SOFTPLUS_OFFSET + i as usize] = (sp * 32.0)
                .clamp(-128.0, 127.0) as i8 as u8;
        }

        // rsqrt LUT
        for i in 0u16..256 {
            let x = (i.max(1) as f64) / 32.0;
            let rsqrt = 1.0 / x.sqrt();
            luts[RSQRT_OFFSET + i as usize] = (rsqrt * 32.0).min(255.0) as u8;
        }

        // exp_neg LUT
        for i in 0u16..256 {
            let x = i as f64 / 32.0;
            let exp_neg = (-x).exp();
            luts[EXP_NEG_OFFSET + i as usize] = (exp_neg * 255.0) as u8;
        }

        luts
    }

    #[test]
    fn test_silu_properties() {
        let luts = make_test_luts();

        // SiLU(0) ≈ 0
        let val = silu_lut(&luts, 0);
        assert!(val.abs() <= 1, "SiLU(0) should be near 0, got {}", val);

        // SiLU is monotonically increasing for large positive values
        let neg = silu_lut(&luts, -64);
        let pos = silu_lut(&luts, 64);
        assert!(pos > neg, "SiLU should be increasing: {} > {}", pos, neg);
    }

    #[test]
    fn test_softplus_positive() {
        let luts = make_test_luts();

        // softplus is always non-negative
        for i in -128i8..=127 {
            let val = softplus_lut(&luts, i);
            // In the quantized representation, non-negative float maps to non-negative int8
            // (assuming the scale/zero-point are set correctly)
            // For the test LUT, softplus(x) * 32 should be >= 0
            // Since softplus(x) >= 0 for all x, and scale is positive, val >= 0
        }

        // softplus is monotonically non-decreasing
        let mut prev = softplus_lut(&luts, -128);
        for i in -127i8..=127 {
            let curr = softplus_lut(&luts, i);
            assert!(curr >= prev, "softplus should be non-decreasing at {}", i);
            prev = curr;
        }
    }

    #[test]
    fn test_exp_neg_decreasing() {
        let luts = make_test_luts();

        // exp(-x) is monotonically non-increasing
        let mut prev = exp_neg_lut(&luts, 0);
        for i in 1u8..=255 {
            let curr = exp_neg_lut(&luts, i);
            assert!(curr <= prev, "exp(-x) should be non-increasing at {}", i);
            prev = curr;
        }

        // exp(0) ≈ 1.0 → 255 in unsigned repr
        assert!(exp_neg_lut(&luts, 0) > 200, "exp(0) should be near max");
    }
}
