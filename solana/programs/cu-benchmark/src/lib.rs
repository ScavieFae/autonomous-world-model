use anchor_lang::prelude::*;

declare_id!("2ugkUeQwNdfFpQXKHja4LiFxFgvn1VNn7w1YLp6XeNEJ");

/// CU benchmark program for INT8 matmul and LUT-based activations.
///
/// Results determine whether we pursue single-tx (~60M CU) or multi-tx pipeline.

#[program]
pub mod cu_benchmark {
    use super::*;

    /// Benchmark INT8 matrix-vector multiply.
    /// y[i] = sum_j(W[i][j] * x[j]), accumulated in i32, requantized to i8.
    pub fn bench_matmul(ctx: Context<BenchMatmul>, rows: u32, cols: u32) -> Result<()> {
        let data = ctx.accounts.benchmark.try_borrow_data()?;

        let rows = rows as usize;
        let cols = cols as usize;
        let weight_size = rows * cols;
        let total_needed = weight_size + cols + rows;

        require!(data.len() >= total_needed, BenchError::InsufficientData);

        let weights = &data[..weight_size];
        let input = &data[weight_size..weight_size + cols];
        let scale: i32 = 128;

        msg!("matmul start: {}x{}", rows, cols);

        for i in 0..rows {
            let mut acc: i32 = 0;
            let row_offset = i * cols;
            for j in 0..cols {
                let w = weights[row_offset + j] as i8 as i32;
                let x = input[j] as i8 as i32;
                acc += w * x;
            }
            let scaled = (acc * scale) >> 8;
            let _output = scaled.clamp(-128, 127) as i8;
        }

        msg!("matmul done: {}x{}", rows, cols);
        Ok(())
    }

    /// Benchmark INT8 matmul with 4x unrolled inner loop.
    pub fn bench_matmul_tiled(ctx: Context<BenchMatmul>, rows: u32, cols: u32) -> Result<()> {
        let data = ctx.accounts.benchmark.try_borrow_data()?;

        let rows = rows as usize;
        let cols = cols as usize;
        let weight_size = rows * cols;
        let total_needed = weight_size + cols + rows;

        require!(data.len() >= total_needed, BenchError::InsufficientData);

        let weights = &data[..weight_size];
        let input = &data[weight_size..weight_size + cols];
        let scale: i32 = 128;

        msg!("matmul_tiled start: {}x{}", rows, cols);

        for i in 0..rows {
            let mut acc0: i32 = 0;
            let mut acc1: i32 = 0;
            let mut acc2: i32 = 0;
            let mut acc3: i32 = 0;
            let row_offset = i * cols;
            let chunks = cols / 4;
            let remainder = cols % 4;

            for j in 0..chunks {
                let base = row_offset + j * 4;
                let x_base = j * 4;
                acc0 += weights[base] as i8 as i32 * input[x_base] as i8 as i32;
                acc1 += weights[base + 1] as i8 as i32 * input[x_base + 1] as i8 as i32;
                acc2 += weights[base + 2] as i8 as i32 * input[x_base + 2] as i8 as i32;
                acc3 += weights[base + 3] as i8 as i32 * input[x_base + 3] as i8 as i32;
            }

            let mut acc_rem: i32 = 0;
            for j in 0..remainder {
                let idx = chunks * 4 + j;
                acc_rem += weights[row_offset + idx] as i8 as i32 * input[idx] as i8 as i32;
            }

            let acc = acc0 + acc1 + acc2 + acc3 + acc_rem;
            let scaled = (acc * scale) >> 8;
            let _output = scaled.clamp(-128, 127) as i8;
        }

        msg!("matmul_tiled done: {}x{}", rows, cols);
        Ok(())
    }

    /// Benchmark LUT-based activation (SiLU=0, softplus=1, rsqrt=2).
    pub fn bench_lut_activation(
        ctx: Context<BenchLut>,
        num_elements: u32,
        activation_type: u8,
    ) -> Result<()> {
        let data = ctx.accounts.lut.try_borrow_data()?;
        let num_elements = num_elements as usize;
        let lut_offset = (activation_type as usize) * 256;

        require!(data.len() >= lut_offset + 256, BenchError::InsufficientData);
        require!(data.len() >= 768 + num_elements, BenchError::InsufficientData);

        let lut = &data[lut_offset..lut_offset + 256];
        let input = &data[768..768 + num_elements];

        let name = match activation_type {
            0 => "SiLU", 1 => "softplus", 2 => "rsqrt", _ => "unknown",
        };
        msg!("lut_{} start: {} elements", name, num_elements);

        let mut checksum: u32 = 0;
        for i in 0..num_elements {
            let idx = input[i] as usize;
            checksum = checksum.wrapping_add(lut[idx] as u32);
        }

        msg!("lut_{} done: checksum={}", name, checksum);
        Ok(())
    }

    /// Benchmark Mamba2 selective scan step.
    pub fn bench_ssm_step(ctx: Context<BenchSsm>, d_inner: u32, d_state: u32) -> Result<()> {
        let data = ctx.accounts.ssm_data.try_borrow_data()?;

        let d_inner = d_inner as usize;
        let d_state = d_state as usize;
        let h_size = d_inner * d_state;
        let dt_raw_offset = 512usize;
        let x_offset = dt_raw_offset + d_inner;
        let b_offset = x_offset + d_inner;
        let c_offset = b_offset + h_size;
        let h_offset = c_offset + h_size;
        let a_offset = h_offset + h_size;
        let total_needed = a_offset + d_inner;

        require!(data.len() >= total_needed, BenchError::InsufficientData);

        let softplus_lut = &data[0..256];
        let exp_lut = &data[256..512];

        msg!("ssm_step start: d_inner={}, d_state={}", d_inner, d_state);

        for i in 0..d_inner {
            let dt_raw_idx = data[dt_raw_offset + i] as usize;
            let dt = softplus_lut[dt_raw_idx] as i32;
            let a_val = data[a_offset + i] as i8 as i32;
            let x_val = data[x_offset + i] as i8 as i32;

            for j in 0..d_state {
                let h_idx = i * d_state + j;
                let dt_a_product = ((dt * a_val) >> 4).clamp(0, 255) as usize;
                let a_bar = exp_lut[dt_a_product] as i32;
                let h_val = data[h_offset + h_idx] as i8 as i32;
                let b_val = data[b_offset + h_idx] as i8 as i32;
                let h_new = (a_bar * h_val + dt * b_val * x_val) >> 8;
                let _h_new_q = h_new.clamp(-128, 127) as i8;
                let c_val = data[c_offset + h_idx] as i8 as i32;
                let _y = c_val * h_new;
            }
        }

        msg!("ssm_step done: {}x{}", d_inner, d_state);
        Ok(())
    }

    /// Benchmark INT8 matmul with unsafe indexing (no bounds checks).
    pub fn bench_matmul_unsafe(ctx: Context<BenchMatmul>, rows: u32, cols: u32) -> Result<()> {
        let data = ctx.accounts.benchmark.try_borrow_data()?;

        let rows = rows as usize;
        let cols = cols as usize;
        let weight_size = rows * cols;
        let total_needed = weight_size + cols + rows;

        require!(data.len() >= total_needed, BenchError::InsufficientData);

        let weights = &data[..weight_size];
        let input = &data[weight_size..weight_size + cols];

        msg!("matmul_unsafe start: {}x{}", rows, cols);

        // SAFETY: bounds checked above via require!
        unsafe {
            for i in 0..rows {
                let mut acc: i32 = 0;
                let row_offset = i * cols;
                for j in 0..cols {
                    let w = *weights.get_unchecked(row_offset + j) as i8 as i32;
                    let x = *input.get_unchecked(j) as i8 as i32;
                    acc += w * x;
                }
                let _output = ((acc * 128) >> 8).clamp(-128, 127) as i8;
            }
        }

        msg!("matmul_unsafe done: {}x{}", rows, cols);
        Ok(())
    }

    /// Benchmark INT8 matmul with unsafe indexing + packed u32 loads.
    /// Loads 4 bytes at once, extracts individual i8 values, reduces load count by 4x.
    pub fn bench_matmul_packed(ctx: Context<BenchMatmul>, rows: u32, cols: u32) -> Result<()> {
        let data = ctx.accounts.benchmark.try_borrow_data()?;

        let rows = rows as usize;
        let cols = cols as usize;
        let weight_size = rows * cols;
        let total_needed = weight_size + cols + rows;

        require!(data.len() >= total_needed, BenchError::InsufficientData);
        require!(cols % 4 == 0, BenchError::InsufficientData); // cols must be multiple of 4

        let weights = &data[..weight_size];
        let input = &data[weight_size..weight_size + cols];

        msg!("matmul_packed start: {}x{}", rows, cols);

        let chunks = cols / 4;

        // SAFETY: bounds checked above, cols divisible by 4
        unsafe {
            for i in 0..rows {
                let mut acc: i32 = 0;
                let row_offset = i * cols;
                for j in 0..chunks {
                    let w_base = row_offset + j * 4;
                    let x_base = j * 4;

                    // Load 4 weight bytes via pointer cast
                    let w_ptr = weights.as_ptr().add(w_base) as *const u32;
                    let w4 = w_ptr.read_unaligned();

                    // Load 4 input bytes
                    let x_ptr = input.as_ptr().add(x_base) as *const u32;
                    let x4 = x_ptr.read_unaligned();

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
                let _output = ((acc * 128) >> 8).clamp(-128, 127) as i8;
            }
        }

        msg!("matmul_packed done: {}x{}", rows, cols);
        Ok(())
    }

    /// Benchmark full Mamba2 layer (in_proj + SSM + gate + out_proj).
    pub fn bench_full_layer(
        ctx: Context<BenchFullLayer>,
        d_model: u32,
        d_inner: u32,
        d_state: u32,
    ) -> Result<()> {
        let w_data = ctx.accounts.weights.try_borrow_data()?;
        let s_data = ctx.accounts.state.try_borrow_data()?;

        let d_model = d_model as usize;
        let d_inner = d_inner as usize;
        let d_state = d_state as usize;
        let w_len = w_data.len();
        let s_len = s_data.len();

        msg!("full_layer start: d_model={}, d_inner={}, d_state={}", d_model, d_inner, d_state);

        // Step 1: RMSNorm
        let mut norm_sum: i64 = 0;
        for i in 0..d_model.min(s_len) {
            let x = s_data[i] as i8 as i64;
            norm_sum += x * x;
        }

        // Step 2: in_proj matmul (d_model → 2*d_inner)
        let proj_out_dim = 2 * d_inner;
        let max_rows = proj_out_dim.min(w_len / d_model.max(1));
        let mut proj_checksum: i64 = 0;
        for i in 0..max_rows {
            let mut acc: i32 = 0;
            let row_offset = i * d_model;
            for j in 0..d_model {
                if row_offset + j < w_len && j < s_len {
                    acc += w_data[row_offset + j] as i8 as i32 * s_data[j] as i8 as i32;
                }
            }
            proj_checksum += acc as i64;
        }

        // Step 3: SSM step
        let mut ssm_checksum: i64 = 0;
        for i in 0..d_inner.min(256) {
            for j in 0..d_state {
                let idx = (i * d_state + j) % w_len.max(1);
                let h = w_data[idx] as i8 as i32;
                let b = w_data[(idx + 1) % w_len.max(1)] as i8 as i32;
                ssm_checksum += (h * b) as i64;
            }
        }

        // Step 4: Gate (SiLU + multiply)
        // Step 5: out_proj matmul (d_inner → d_model)
        let out_max_rows = d_model.min(w_len / d_inner.max(1));
        let mut out_checksum: i64 = 0;
        for i in 0..out_max_rows {
            let mut acc: i32 = 0;
            let row_offset = i * d_inner;
            for j in 0..d_inner {
                if row_offset + j < w_len {
                    acc += w_data[row_offset + j] as i8 as i32 * s_data[j % s_len] as i8 as i32;
                }
            }
            out_checksum += acc as i64;
        }

        msg!("full_layer done: norm={} proj={} ssm={} out={}", norm_sum, proj_checksum, ssm_checksum, out_checksum);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct BenchMatmul<'info> {
    /// CHECK: Benchmark data account — no ownership checks needed.
    pub benchmark: AccountInfo<'info>,
}

#[derive(Accounts)]
pub struct BenchLut<'info> {
    /// CHECK: LUT data account.
    pub lut: AccountInfo<'info>,
}

#[derive(Accounts)]
pub struct BenchSsm<'info> {
    /// CHECK: SSM data account.
    pub ssm_data: AccountInfo<'info>,
}

#[derive(Accounts)]
pub struct BenchFullLayer<'info> {
    /// CHECK: Weight data account.
    pub weights: AccountInfo<'info>,
    /// CHECK: State data account.
    pub state: AccountInfo<'info>,
}

#[error_code]
pub enum BenchError {
    #[msg("Account data too small for specified dimensions")]
    InsufficientData,
}
