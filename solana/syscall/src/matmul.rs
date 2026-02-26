/// INT8 matrix-vector multiply: y[i] = sum(W[i][j] * x[j]) for j in 0..cols
///
/// All types: i8 x i8 -> i32 accumulate. No floating point.
/// Runs natively on the validator — auto-vectorizes on ARM NEON / x86 AVX.
pub fn matmul_i8(
    weights: &[i8],
    input: &[i8],
    output: &mut [i32],
    rows: usize,
    cols: usize,
) {
    assert!(weights.len() >= rows * cols);
    assert!(input.len() >= cols);
    assert!(output.len() >= rows);

    for i in 0..rows {
        let mut acc: i32 = 0;
        let row_start = i * cols;
        for j in 0..cols {
            acc += weights[row_start + j] as i32 * input[j] as i32;
        }
        output[i] = acc;
    }
}
