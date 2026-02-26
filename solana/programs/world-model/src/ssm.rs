/// Selective scan step — the core SSM recurrence for Mamba2.
///
/// For each (i, j) in d_inner × d_state:
///   A_bar = exp(-dt[i] * A[i])                   (LUT)
///   h_new[i,j] = A_bar * h[i,j] + dt[i] * B[i,j] * x_ssm[i]   (INT8/INT32 MAC)
///   y[i] += C[i,j] * h_new[i,j]                 (INT8 dot product)
///
/// CU estimate for d_inner=1024, d_state=16: ~147K CU

use crate::lut;

/// Execute one selective scan step.
///
/// Arguments:
///   x_ssm:    SSM input vector, shape (d_inner,)
///   dt:       Timestep after softplus, shape (d_inner,)
///   h:        Hidden state, shape (d_inner * d_state,) — modified in place
///   a_log:    Log diagonal of SSM decay matrix, shape (d_inner,)
///   lut_data: Packed activation LUTs (1024 bytes)
///   y_ssm:    Output vector, shape (d_inner,) — written
///   d_inner:  Inner dimension
///   d_state:  State dimension
pub fn selective_scan_step(
    x_ssm: &[i8],
    dt: &[i8],
    h: &mut [i8],
    a_log: &[u8],
    lut_data: &[u8],
    y_ssm: &mut [i8],
    d_inner: usize,
    d_state: usize,
) {
    for i in 0..d_inner {
        let dt_val = dt[i] as i32;
        let a_val = a_log[i] as i8 as i32;
        let x_val = x_ssm[i] as i32;

        // A_bar = exp(-dt * A) via LUT
        let dt_a = ((dt_val.abs() * a_val.abs()) >> 4).min(255) as u8;
        let a_bar = lut::exp_neg_lut(lut_data, dt_a) as i32;

        let mut y_acc: i32 = 0;

        for j in 0..d_state {
            let h_idx = i * d_state + j;

            // Current hidden state
            let h_val = h[h_idx] as i32;

            // B and C derived from position (simplified)
            // In full Mamba2, these come from in_proj's B and C output heads
            let b_val = ((x_val * (j as i32 + 1)) >> 4).clamp(-128, 127);
            let c_val = ((x_val * (d_state as i32 - j as i32)) >> 4).clamp(-128, 127);

            // h_new = A_bar * h + dt * B * x_ssm
            let h_new = (a_bar * h_val + dt_val * b_val) >> 8;
            h[h_idx] = h_new.clamp(-128, 127) as i8;

            // y += C * h_new
            y_acc += c_val * h_new;
        }

        // Requantize SSM output
        y_ssm[i] = (y_acc >> 8).clamp(-128, 127) as i8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_luts() -> Vec<u8> {
        let mut luts = vec![0u8; lut::LUT_TOTAL_SIZE];

        // exp_neg LUT
        for i in 0u16..256 {
            let x = i as f64 / 32.0;
            let exp_neg = (-x).exp();
            luts[lut::EXP_NEG_OFFSET + i as usize] = (exp_neg * 255.0) as u8;
        }

        luts
    }

    #[test]
    fn test_ssm_step_zero_input() {
        let luts = make_test_luts();
        let d_inner = 4;
        let d_state = 2;

        let x_ssm = vec![0i8; d_inner];
        let dt = vec![10i8; d_inner];
        let mut h = vec![10i8; d_inner * d_state];
        let a_log = vec![16u8; d_inner];
        let mut y_ssm = vec![0i8; d_inner];

        selective_scan_step(&x_ssm, &dt, &mut h, &a_log, &luts, &mut y_ssm, d_inner, d_state);

        // With zero input, hidden state should decay toward zero
        // and output should be near zero (since C depends on x_val=0)
        for &y in &y_ssm {
            assert_eq!(y, 0, "zero input should produce zero output");
        }
    }

    #[test]
    fn test_ssm_step_nonzero() {
        let luts = make_test_luts();
        let d_inner = 4;
        let d_state = 2;

        let x_ssm = vec![32i8; d_inner];
        let dt = vec![16i8; d_inner];
        let mut h = vec![0i8; d_inner * d_state];
        let a_log = vec![8u8; d_inner];
        let mut y_ssm = vec![0i8; d_inner];

        selective_scan_step(&x_ssm, &dt, &mut h, &a_log, &luts, &mut y_ssm, d_inner, d_state);

        // With nonzero input and zero initial hidden state, we should get nonzero output
        let any_nonzero = y_ssm.iter().any(|&y| y != 0);
        assert!(any_nonzero, "nonzero input should produce nonzero output");
    }
}
