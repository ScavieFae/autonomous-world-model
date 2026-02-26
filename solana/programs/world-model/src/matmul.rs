/// INT8 matrix-vector multiplication optimized for Solana BPF.
///
/// Core operation for Mamba2 inference:
///   y = W * x
///   W: (rows, cols) INT8 matrix (weights, zero-copy from account)
///   x: (cols,) INT8 vector (activations)
///   y: (rows,) INT32 accumulator → requantized to INT8
///
/// Uses packed u32 loads for ~16 CU/MAC (proven in cu-benchmark).

/// Matrix-vector multiply: y = W * x with INT32 accumulation.
///
/// Inner loop uses packed u32 `read_unaligned` to load 4 bytes at once,
/// reducing memory load count by 4x vs individual byte loads.
///
/// Arguments:
///   weights: Row-major INT8 weight matrix, shape (rows, cols), stored as &[u8]
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

    // SAFETY: bounds checked above via asserts. Packed loads read 4 bytes
    // at a time from within the validated slice range.
    unsafe {
        let w_ptr = weights.as_ptr();
        let x_ptr = input.as_ptr() as *const u8;

        for i in 0..rows {
            let mut acc: i32 = 0;
            let row_offset = i * cols;

            // Packed 4-byte loads — the key optimization (~16 CU/MAC)
            for j in 0..chunks {
                let w_base = row_offset + j * 4;
                let x_base = j * 4;

                // Load 4 weight bytes via pointer cast
                let w4 = (w_ptr.add(w_base) as *const u32).read_unaligned();
                // Load 4 input bytes
                let x4 = (x_ptr.add(x_base) as *const u32).read_unaligned();

                // Extract individual bytes as signed i8 -> i32
                let w0 = (w4 as u8) as i8 as i32;
                let w1 = ((w4 >> 8) as u8) as i8 as i32;
                let w2 = ((w4 >> 16) as u8) as i8 as i32;
                let w3 = ((w4 >> 24) as u8) as i8 as i32;

                let x0 = (x4 as u8) as i8 as i32;
                let x1 = ((x4 >> 8) as u8) as i8 as i32;
                let x2 = ((x4 >> 16) as u8) as i8 as i32;
                let x3 = ((x4 >> 24) as u8) as i8 as i32;

                acc += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
            }

            // Handle remainder (cols not divisible by 4)
            for j in 0..remainder {
                let idx = chunks * 4 + j;
                let w = *weights.get_unchecked(row_offset + idx) as i8 as i32;
                let x = *input.get_unchecked(idx) as i32;
                acc += w * x;
            }

            output[i] = acc;
        }
    }
}

/// Requantize INT32 accumulator values to INT8 using per-channel scale factors.
///
/// For each output element:
///   output_i8[i] = clamp(round(output_i32[i] * scale[i] / 65536), -128, 127)
///
/// Scale factors are stored as u16 fixed-point values in the manifest:
///   actual_scale = raw_u16 / 65536.0
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
        let weights: &[u8] = &[(-1i8) as u8, 2, 3, (-4i8) as u8];
        let input: &[i8] = &[-5, 6];
        let mut output = [0i32; 2];

        matmul_i8(weights, input, &mut output, 2, 2);

        assert_eq!(output[0], (-1) * (-5) + 2 * 6); // 17
        assert_eq!(output[1], 3 * (-5) + (-4) * 6);  // -39
    }

    #[test]
    fn test_matmul_unrolling() {
        // Test with cols=8 to exercise packed loop
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
    fn test_matmul_large_packed() {
        // Test with a larger matrix that fully exercises the packed loop
        let rows = 4;
        let cols = 16;
        let weights: Vec<u8> = (0..64).map(|i| ((i % 5) as i8 - 2) as u8).collect();
        let input: Vec<i8> = (0..16).map(|i| i as i8 - 8).collect();
        let mut output = vec![0i32; rows];

        matmul_i8(&weights, &input, &mut output, rows, cols);

        // Verify against naive computation
        for i in 0..rows {
            let expected: i32 = (0..cols).map(|j| {
                let w = weights[i * cols + j] as i8 as i32;
                let x = input[j] as i32;
                w * x
            }).sum();
            assert_eq!(output[i], expected, "row {} mismatch", i);
        }
    }

    #[test]
    fn test_requantize() {
        let input = [1000i32, -2000, 500, -100];
        let scales = [32768u16, 16384, 65535, 8192];
        let mut output = [0i8; 4];

        requantize_per_channel(&input, &scales, &mut output, 4);

        assert_eq!(output[0], 127);  // 1000 * 32768 / 65536 = 500 → clamped
        assert_eq!(output[1], -128); // -2000 * 16384 / 65536 = -500 → clamped
    }

    #[test]
    fn test_elementwise_mul() {
        let a: &[i8] = &[10, -20, 30, -40];
        let b: &[i8] = &[5, -5, 2, -3];
        let mut output = [0i8; 4];

        elementwise_mul_i8(a, b, &mut output, 4, 4);

        assert_eq!(output[0], 3);  // 10 * 5 = 50, >> 4 = 3
        assert_eq!(output[1], 6);  // (-20) * (-5) = 100, >> 4 = 6
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
