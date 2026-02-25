/// INT8 matrix-vector multiplication optimized for Solana BPF.
///
/// Core operation for Mamba2 inference:
///   y = W * x
///   W: (rows, cols) INT8 matrix (weights, zero-copy from account)
///   x: (cols,) INT8 vector (activations)
///   y: (rows,) INT32 accumulator → requantized to INT8
///
/// Optimization strategies for BPF:
///   1. 4x unrolled inner loop (reduce branch overhead)
///   2. Zero-copy weight access (no deserialization, read directly from account data)
///   3. INT32 accumulator (i8 * i8 → i32, accumulate in i32)
///   4. Per-channel requantization (scale factors from manifest)
///
/// CU estimate for 512×1024 (out_proj):
///   524,288 MACs × ~3 CU/MAC = ~1.6M CU

/// Matrix-vector multiply: y = W * x with INT32 accumulation.
///
/// Arguments:
///   weights: Row-major INT8 weight matrix, shape (rows, cols), stored as &[u8]
///            (reinterpreted as i8 during computation)
///   input:   INT8 input vector, shape (cols,), stored as &[i8]
///   output:  INT32 output vector, shape (rows,) — caller requantizes
///   rows:    Number of output elements
///   cols:    Number of input elements (dot product length)
pub fn matmul_i8(
    weights: &[u8],
    input: &[i8],
    output: &mut [i32],
    rows: usize,
    cols: usize,
) {
    assert!(weights.len() >= rows * cols);
    assert!(input.len() >= cols);
    assert!(output.len() >= rows);

    let chunks = cols / 4;
    let remainder = cols % 4;

    for i in 0..rows {
        let row_offset = i * cols;
        let mut acc0: i32 = 0;
        let mut acc1: i32 = 0;
        let mut acc2: i32 = 0;
        let mut acc3: i32 = 0;

        // 4x unrolled inner loop
        for j in 0..chunks {
            let base_w = row_offset + j * 4;
            let base_x = j * 4;

            // Read weights as i8 (reinterpret u8 → i8)
            let w0 = weights[base_w] as i8 as i32;
            let w1 = weights[base_w + 1] as i8 as i32;
            let w2 = weights[base_w + 2] as i8 as i32;
            let w3 = weights[base_w + 3] as i8 as i32;

            let x0 = input[base_x] as i32;
            let x1 = input[base_x + 1] as i32;
            let x2 = input[base_x + 2] as i32;
            let x3 = input[base_x + 3] as i32;

            acc0 += w0 * x0;
            acc1 += w1 * x1;
            acc2 += w2 * x2;
            acc3 += w3 * x3;
        }

        // Handle remainder
        let mut acc_rem: i32 = 0;
        for j in 0..remainder {
            let idx = chunks * 4 + j;
            let w = weights[row_offset + idx] as i8 as i32;
            let x = input[idx] as i32;
            acc_rem += w * x;
        }

        output[i] = acc0 + acc1 + acc2 + acc3 + acc_rem;
    }
}

/// Requantize INT32 accumulator values to INT8 using per-channel scale factors.
///
/// For each output element:
///   output_i8[i] = clamp(round(output_i32[i] * scale[i] / 65536), -128, 127)
///
/// Scale factors are stored as u16 fixed-point values in the manifest:
///   actual_scale = raw_u16 / 65536.0
///
/// This maps the INT32 accumulator range back to INT8 for the next layer.
pub fn requantize_per_channel(
    input: &[i32],
    scales: &[u16],
    output: &mut [i8],
    n: usize,
) {
    assert!(input.len() >= n);
    assert!(scales.len() >= n);
    assert!(output.len() >= n);

    for i in 0..n {
        // Multiply by scale (u16) and right-shift by 16
        // This is equivalent to dividing by 65536 and multiplying by the scale
        let scaled = ((input[i] as i64 * scales[i] as i64) >> 16) as i32;
        output[i] = scaled.clamp(-128, 127) as i8;
    }
}

/// Requantize with a single per-tensor scale factor.
pub fn requantize_per_tensor(
    input: &[i32],
    scale: u16,
    output: &mut [i8],
    n: usize,
) {
    assert!(input.len() >= n);
    assert!(output.len() >= n);

    let scale_i64 = scale as i64;
    for i in 0..n {
        let scaled = ((input[i] as i64 * scale_i64) >> 16) as i32;
        output[i] = scaled.clamp(-128, 127) as i8;
    }
}

/// Element-wise multiply two INT8 vectors with INT8 output.
///
/// Used for: y = y_ssm * SiLU(z) (gating step)
///
/// Computes: output[i] = (a[i] * b[i]) >> shift
/// The shift compensates for the product of two INT8 values being INT16-range.
pub fn elementwise_mul_i8(
    a: &[i8],
    b: &[i8],
    output: &mut [i8],
    n: usize,
    shift: u32,
) {
    assert!(a.len() >= n);
    assert!(b.len() >= n);
    assert!(output.len() >= n);

    for i in 0..n {
        let product = (a[i] as i32) * (b[i] as i32);
        let shifted = product >> shift;
        output[i] = shifted.clamp(-128, 127) as i8;
    }
}

/// Add two INT8 vectors (residual connection).
///
/// Computes: output[i] = clamp(a[i] + b[i], -128, 127)
/// Saturation arithmetic for residual adds.
pub fn add_i8(a: &[i8], b: &[i8], output: &mut [i8], n: usize) {
    assert!(a.len() >= n);
    assert!(b.len() >= n);
    assert!(output.len() >= n);

    for i in 0..n {
        let sum = (a[i] as i16) + (b[i] as i16);
        output[i] = sum.clamp(-128, 127) as i8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity matrix times [1, 2] should give [1, 2]
        let weights: &[u8] = &[1, 0, 0, 1]; // [[1,0],[0,1]] as u8
        let input: &[i8] = &[10, 20];
        let mut output = [0i32; 2];

        matmul_i8(weights, input, &mut output, 2, 2);

        assert_eq!(output[0], 10);
        assert_eq!(output[1], 20);
    }

    #[test]
    fn test_matmul_simple() {
        // [[1, 2], [3, 4]] * [5, 6] = [17, 39]
        let weights: &[u8] = &[1, 2, 3, 4];
        let input: &[i8] = &[5, 6];
        let mut output = [0i32; 2];

        matmul_i8(weights, input, &mut output, 2, 2);

        assert_eq!(output[0], 1 * 5 + 2 * 6); // 17
        assert_eq!(output[1], 3 * 5 + 4 * 6); // 39
    }

    #[test]
    fn test_matmul_negative() {
        // Test with negative INT8 values
        // [[-1, 2], [3, -4]] * [-5, 6] = [17, -39]
        let weights: &[u8] = &[(-1i8) as u8, 2, 3, (-4i8) as u8];
        let input: &[i8] = &[-5, 6];
        let mut output = [0i32; 2];

        matmul_i8(weights, input, &mut output, 2, 2);

        assert_eq!(output[0], (-1) * (-5) + 2 * 6); // 17
        assert_eq!(output[1], 3 * (-5) + (-4) * 6);  // -39
    }

    #[test]
    fn test_matmul_unrolling() {
        // Test with cols > 4 to exercise unrolled loop
        let cols = 8;
        let rows = 2;
        let weights: Vec<u8> = (0..16).map(|i| (i as i8 + 1) as u8).collect();
        let input: Vec<i8> = (0..8).map(|i| i as i8 + 1).collect();
        let mut output = vec![0i32; rows];

        matmul_i8(&weights, &input, &mut output, rows, cols);

        // Row 0: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 = 204
        let expected0: i32 = (1..=8).map(|i: i32| i * i).sum();
        assert_eq!(output[0], expected0);
    }

    #[test]
    fn test_requantize() {
        let input = [1000i32, -2000, 500, -100];
        let scales = [32768u16, 16384, 65535, 8192]; // ~0.5, ~0.25, ~1.0, ~0.125
        let mut output = [0i8; 4];

        requantize_per_channel(&input, &scales, &mut output, 4);

        // 1000 * 32768 / 65536 = 500 → clamped to 127
        assert_eq!(output[0], 127);
        // -2000 * 16384 / 65536 = -500 → clamped to -128
        assert_eq!(output[1], -128);
    }

    #[test]
    fn test_elementwise_mul() {
        let a: &[i8] = &[10, -20, 30, -40];
        let b: &[i8] = &[5, -5, 2, -3];
        let mut output = [0i8; 4];

        elementwise_mul_i8(a, b, &mut output, 4, 4);

        // 10 * 5 = 50, >> 4 = 3
        assert_eq!(output[0], 3);
        // (-20) * (-5) = 100, >> 4 = 6
        assert_eq!(output[1], 6);
    }

    #[test]
    fn test_add_saturation() {
        let a: &[i8] = &[100, -100, 50, -50];
        let b: &[i8] = &[100, -100, -60, 60];
        let mut output = [0i8; 4];

        add_i8(a, b, &mut output, 4);

        assert_eq!(output[0], 127);  // 200 → clamped
        assert_eq!(output[1], -128); // -200 → clamped
        assert_eq!(output[2], -10);
        assert_eq!(output[3], 10);
    }
}
