use awm_syscall::matmul::matmul_i8;

#[test]
fn identity_matrix() {
    let n = 4;
    let mut weights = vec![0i8; n * n];
    for i in 0..n {
        weights[i * n + i] = 1;
    }
    let input = vec![10i8, 20, 30, 40];
    let mut output = vec![0i32; n];

    matmul_i8(&weights, &input, &mut output, n, n);

    assert_eq!(output, vec![10, 20, 30, 40]);
}

#[test]
fn known_values() {
    // [[1,2],[3,4]] x [5,6] = [17, 39]
    let weights = vec![1i8, 2, 3, 4];
    let input = vec![5i8, 6];
    let mut output = vec![0i32; 2];

    matmul_i8(&weights, &input, &mut output, 2, 2);

    assert_eq!(output[0], 17);
    assert_eq!(output[1], 39);
}

#[test]
fn negative_values() {
    let weights = vec![-1i8, 2, 3, -4];
    let input = vec![-5i8, 6];
    let mut output = vec![0i32; 2];

    matmul_i8(&weights, &input, &mut output, 2, 2);

    assert_eq!(output[0], (-1) * (-5) + 2 * 6); // 17
    assert_eq!(output[1], 3 * (-5) + (-4) * 6); // -39
}

#[test]
fn saturating_i8_range() {
    // Max positive * max positive
    let weights = vec![127i8, 127];
    let input = vec![127i8, 127];
    let mut output = vec![0i32; 1];

    matmul_i8(&weights, &input, &mut output, 1, 2);

    assert_eq!(output[0], 127 * 127 + 127 * 127); // 32258

    // Max negative * max negative
    let weights = vec![-128i8, -128];
    let input = vec![-128i8, -128];
    let mut output = vec![0i32; 1];

    matmul_i8(&weights, &input, &mut output, 1, 2);

    assert_eq!(output[0], (-128) * (-128) + (-128) * (-128)); // 32768
}

#[test]
fn production_dimensions_2048x512() {
    let rows = 2048;
    let cols = 512;
    let weights: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 7 + 13) % 256) as i8)
        .collect();
    let input: Vec<i8> = (0..cols).map(|i| ((i * 3 + 5) % 256) as i8).collect();
    let mut output = vec![0i32; rows];

    matmul_i8(&weights, &input, &mut output, rows, cols);

    // Verify select rows against naive computation
    for &i in &[0, 1, 100, 1000, 2047] {
        let expected: i32 = (0..cols)
            .map(|j| weights[i * cols + j] as i32 * input[j] as i32)
            .sum();
        assert_eq!(output[i], expected, "row {} mismatch", i);
    }
}

#[test]
fn production_dimensions_512x1024() {
    let rows = 512;
    let cols = 1024;
    let weights: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 11 + 3) % 256) as i8)
        .collect();
    let input: Vec<i8> = (0..cols).map(|i| ((i * 7 + 1) % 256) as i8).collect();
    let mut output = vec![0i32; rows];

    matmul_i8(&weights, &input, &mut output, rows, cols);

    for &i in &[0, 1, 255, 511] {
        let expected: i32 = (0..cols)
            .map(|j| weights[i * cols + j] as i32 * input[j] as i32)
            .sum();
        assert_eq!(output[i], expected, "row {} mismatch", i);
    }
}

#[test]
fn cols_not_divisible_by_4() {
    for cols in [3, 5, 7, 1, 13] {
        let rows = 3;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i + 1) as i8).collect();
        let input: Vec<i8> = (0..cols).map(|i| (i + 1) as i8).collect();
        let mut output = vec![0i32; rows];

        matmul_i8(&weights, &input, &mut output, rows, cols);

        for i in 0..rows {
            let expected: i32 = (0..cols)
                .map(|j| weights[i * cols + j] as i32 * input[j] as i32)
                .sum();
            assert_eq!(output[i], expected, "cols={} row {} mismatch", cols, i);
        }
    }
}

#[test]
fn single_element() {
    let weights = vec![7i8];
    let input = vec![3i8];
    let mut output = vec![0i32; 1];

    matmul_i8(&weights, &input, &mut output, 1, 1);

    assert_eq!(output[0], 21);
}

#[test]
fn zero_weights() {
    let rows = 4;
    let cols = 4;
    let weights = vec![0i8; rows * cols];
    let input = vec![100i8, -50, 25, -12];
    let mut output = vec![999i32; rows];

    matmul_i8(&weights, &input, &mut output, rows, cols);

    assert_eq!(output, vec![0, 0, 0, 0]);
}
