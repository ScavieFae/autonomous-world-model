/// Mamba2 INT8 inference kernel for onchain execution.
///
/// Implements a single-step (autoregressive) Mamba2 forward pass:
///   (input_state, controller_inputs, hidden_state) → (output_state, new_hidden_state)
///
/// Architecture (per layer):
///   1. RMSNorm(x)
///   2. in_proj: x → [z, x_ssm, B, C, dt]    (INT8 matmul)
///   3. Selective scan step:
///      dt = softplus(dt)                       (LUT)
///      A_bar = exp(-dt * A)                    (LUT)
///      h_new = A_bar * h + dt * B * x_ssm     (INT8/INT32 MAC)
///      y = C * h_new                           (INT8 dot product)
///   4. Gate: y = y * SiLU(z)                  (LUT + multiply)
///   5. out_proj: y → residual                 (INT8 matmul)
///   6. Residual add                           (INT32 add, requantize)
///
/// Per-layer CU estimate (d_model=512, d_inner=1024, d_state=16):
///   in_proj:  ~3.1M CU
///   SSM step: ~147K CU
///   gate:     ~5K CU
///   out_proj: ~1.6M CU
///   total:    ~4.9M CU per layer, ~59M CU for 12 layers

use crate::lut;
use crate::matmul;

/// Configuration for a Mamba2 model, matching ModelManifest fields.
pub struct Mamba2Config {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}

/// Weight layout offsets within a shard.
/// These are computed from the manifest and used to index into weight account data.
pub struct LayerWeights<'a> {
    /// in_proj weight: (2*d_inner, d_model) — maps input to [z, x_ssm]
    pub in_proj: &'a [u8],
    /// out_proj weight: (d_model, d_inner) — maps gated output back to residual
    pub out_proj: &'a [u8],
    /// RMSNorm weight: (d_model,)
    pub norm: &'a [u8],
    /// A_log diagonal: (d_inner,) — log of SSM decay matrix
    pub a_log: &'a [u8],
    /// dt bias: (d_inner,) — timestep bias
    pub dt_bias: &'a [u8],
    /// Per-channel requantization scales for in_proj output
    pub in_proj_scales: &'a [u16],
    /// Per-channel requantization scales for out_proj output
    pub out_proj_scales: &'a [u16],
}

/// Scratch buffers for intermediate computations within a layer.
/// Allocated once and reused across layers to avoid per-layer allocation.
pub struct ScratchBuffers {
    /// Normalized input: (d_model,)
    pub x_norm: Vec<i8>,
    /// in_proj output before split: (2*d_inner,) as INT32
    pub proj_i32: Vec<i32>,
    /// z (gate input): (d_inner,)
    pub z: Vec<i8>,
    /// x_ssm (SSM input): (d_inner,)
    pub x_ssm: Vec<i8>,
    /// dt after softplus: (d_inner,)
    pub dt: Vec<i8>,
    /// SSM output: (d_inner,)
    pub y_ssm: Vec<i8>,
    /// Gate output (SiLU(z)): (d_inner,)
    pub gate: Vec<i8>,
    /// Gated output: (d_inner,)
    pub y_gated: Vec<i8>,
    /// out_proj output as INT32: (d_model,)
    pub out_i32: Vec<i32>,
    /// Layer output: (d_model,)
    pub y_out: Vec<i8>,
}

impl ScratchBuffers {
    pub fn new(d_model: usize, d_inner: usize) -> Self {
        Self {
            x_norm: vec![0i8; d_model],
            proj_i32: vec![0i32; 2 * d_inner],
            z: vec![0i8; d_inner],
            x_ssm: vec![0i8; d_inner],
            dt: vec![0i8; d_inner],
            y_ssm: vec![0i8; d_inner],
            gate: vec![0i8; d_inner],
            y_gated: vec![0i8; d_inner],
            out_i32: vec![0i32; d_model],
            y_out: vec![0i8; d_model],
        }
    }
}

/// Execute one Mamba2 layer (single timestep, single layer).
///
/// This is the core inner loop called num_layers times per frame.
///
/// Arguments:
///   x: Input activations, shape (d_model,) — modified in place (residual)
///   h: Hidden state for this layer, shape (d_inner * d_state,) — modified in place
///   weights: Weight data and scales for this layer
///   lut_data: Packed activation LUTs (1024 bytes)
///   config: Model configuration
///   scratch: Pre-allocated scratch buffers
pub fn mamba2_layer_step(
    x: &mut [i8],
    h: &mut [i8],
    weights: &LayerWeights,
    lut_data: &[u8],
    config: &Mamba2Config,
    scratch: &mut ScratchBuffers,
) {
    let d_model = config.d_model;
    let d_inner = config.d_inner;
    let d_state = config.d_state;

    // ── Step 1: RMSNorm ─────────────────────────────────────────────────
    lut::rmsnorm_int8(
        lut_data,
        x,
        // Reinterpret norm weights as i8
        unsafe { core::slice::from_raw_parts(weights.norm.as_ptr() as *const i8, d_model) },
        &mut scratch.x_norm,
        256, // weight_scale
    );

    // ── Step 2: in_proj matmul ──────────────────────────────────────────
    // x_norm → [z, x_ssm] via matmul with in_proj weights
    // in_proj shape: (2*d_inner, d_model)
    matmul::matmul_i8(
        weights.in_proj,
        &scratch.x_norm,
        &mut scratch.proj_i32,
        2 * d_inner,
        d_model,
    );

    // Requantize and split into z and x_ssm
    let mut proj_i8 = vec![0i8; 2 * d_inner];
    matmul::requantize_per_channel(
        &scratch.proj_i32,
        weights.in_proj_scales,
        &mut proj_i8,
        2 * d_inner,
    );

    scratch.z.copy_from_slice(&proj_i8[..d_inner]);
    scratch.x_ssm.copy_from_slice(&proj_i8[d_inner..2 * d_inner]);

    // ── Step 3: Selective scan step ─────────────────────────────────────
    // dt = softplus(x_ssm[..d_inner] + dt_bias)
    // For simplicity, dt is derived from x_ssm (in full Mamba2, it's a separate projection)
    for i in 0..d_inner {
        let dt_raw = (scratch.x_ssm[i] as i16 + weights.dt_bias[i] as i8 as i16)
            .clamp(-128, 127) as i8;
        scratch.dt[i] = lut::softplus_lut(lut_data, dt_raw);
    }

    // Selective scan: for each (i, j) in d_inner × d_state
    //   A_bar = exp(-dt[i] * A[i])
    //   h_new[i,j] = A_bar * h[i,j] + dt[i] * B[i,j] * x_ssm[i]
    //   y[i] += C[i,j] * h_new[i,j]
    //
    // Where B and C are derived from x_ssm (simplified; in full Mamba2,
    // they come from separate projections in in_proj)
    for i in 0..d_inner {
        let dt_val = scratch.dt[i] as i32;
        let a_val = weights.a_log[i] as i8 as i32;
        let x_val = scratch.x_ssm[i] as i32;

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
        scratch.y_ssm[i] = (y_acc >> 8).clamp(-128, 127) as i8;
    }

    // ── Step 4: Gate ────────────────────────────────────────────────────
    // gate = SiLU(z)
    scratch.gate.copy_from_slice(&scratch.z);
    lut::silu_slice(lut_data, &mut scratch.gate);

    // y_gated = y_ssm * gate
    matmul::elementwise_mul_i8(
        &scratch.y_ssm,
        &scratch.gate,
        &mut scratch.y_gated,
        d_inner,
        7, // shift: INT8 * INT8 has ~14 bits, shift 7 to center
    );

    // ── Step 5: out_proj matmul ─────────────────────────────────────────
    // y_gated → residual via out_proj
    // out_proj shape: (d_model, d_inner)
    matmul::matmul_i8(
        weights.out_proj,
        &scratch.y_gated,
        &mut scratch.out_i32,
        d_model,
        d_inner,
    );

    matmul::requantize_per_channel(
        &scratch.out_i32,
        weights.out_proj_scales,
        &mut scratch.y_out,
        d_model,
    );

    // ── Step 6: Residual add ────────────────────────────────────────────
    let residual = x.to_vec();
    matmul::add_i8(&residual, &scratch.y_out, x, d_model);
}

/// Encode game state + controller inputs into model input vector.
///
/// Maps the structured game state (positions, percents, action states, etc.)
/// plus controller inputs into a flat INT8 vector that the model's first layer
/// expects.
///
/// The encoding matches the v2 encoding from nojohns-training:
///   - Continuous values: quantized to INT8 with field-specific scales
///   - Categorical values: embedding lookup (small, stored in manifest)
///   - Controller inputs: normalized to INT8 range
pub fn encode_input(
    players: &[(i32, i32, u16, u16, i16, i16, i16, i16, i16, u16, u8, u8, u8, u8, u16, u8, u8); 2],
    controller_inputs: &[(i8, i8, i8, i8, u8, u8, u8); 2],
    stage: u8,
    output: &mut [i8],
    d_model: usize,
) {
    // Zero the output vector
    for v in output.iter_mut() {
        *v = 0;
    }

    // Encode each player's state into the first portion of the vector
    let mut offset = 0;
    for p_idx in 0..2 {
        let p = &players[p_idx];
        let c = &controller_inputs[p_idx];

        // Continuous fields (quantized to INT8)
        if offset < d_model { output[offset] = (p.0 / 256).clamp(-128, 127) as i8; } // x
        offset += 1;
        if offset < d_model { output[offset] = (p.1 / 256).clamp(-128, 127) as i8; } // y
        offset += 1;
        if offset < d_model { output[offset] = (p.2 as i32 / 4).clamp(-128, 127) as i8; } // percent
        offset += 1;
        if offset < d_model { output[offset] = (p.3 as i8); } // shield
        offset += 1;
        if offset < d_model { output[offset] = (p.4 / 2).clamp(-128, 127) as i8; } // speed_air_x
        offset += 1;
        if offset < d_model { output[offset] = (p.5 / 2).clamp(-128, 127) as i8; } // speed_y
        offset += 1;
        if offset < d_model { output[offset] = (p.6 / 2).clamp(-128, 127) as i8; } // speed_ground_x
        offset += 1;
        if offset < d_model { output[offset] = (p.7 / 2).clamp(-128, 127) as i8; } // speed_attack_x
        offset += 1;
        if offset < d_model { output[offset] = (p.8 / 2).clamp(-128, 127) as i8; } // speed_attack_y
        offset += 1;
        if offset < d_model { output[offset] = (p.9 as i8); } // state_age (capped)
        offset += 1;
        if offset < d_model { output[offset] = p.10 as i8; } // hitlag
        offset += 1;
        if offset < d_model { output[offset] = p.11 as i8; } // stocks
        offset += 1;

        // Binary fields
        if offset < d_model { output[offset] = if p.12 != 0 { 64 } else { -64 }; } // facing
        offset += 1;
        if offset < d_model { output[offset] = if p.13 != 0 { 64 } else { -64 }; } // on_ground
        offset += 1;

        // Categorical (as INT8 indices — embedding handled by first layer)
        if offset < d_model { output[offset] = (p.14 as i8); } // action_state (truncated)
        offset += 1;
        if offset < d_model { output[offset] = p.15 as i8; } // jumps_left
        offset += 1;
        if offset < d_model { output[offset] = p.16 as i8; } // character
        offset += 1;

        // Controller inputs
        if offset < d_model { output[offset] = c.0; } // stick_x
        offset += 1;
        if offset < d_model { output[offset] = c.1; } // stick_y
        offset += 1;
        if offset < d_model { output[offset] = c.2; } // c_stick_x
        offset += 1;
        if offset < d_model { output[offset] = c.3; } // c_stick_y
        offset += 1;
        if offset < d_model { output[offset] = c.4 as i8; } // trigger_l
        offset += 1;
        if offset < d_model { output[offset] = c.5 as i8; } // trigger_r
        offset += 1;
        if offset < d_model { output[offset] = c.6 as i8; } // buttons
        offset += 1;
    }

    // Stage
    if offset < d_model {
        output[offset] = stage as i8;
    }
}

/// Decode model output vector into structured game state.
///
/// The model's final layer output is a flat INT8 vector.
/// This function extracts per-player state fields using known offsets
/// and field-specific dequantization.
///
/// For categorical outputs (action_state), the output head produces logits
/// across classes — we take the argmax. For continuous outputs, we dequantize.
pub struct DecodedPlayerState {
    pub x: i32,
    pub y: i32,
    pub percent: u16,
    pub shield_strength: u16,
    pub speed_air_x: i16,
    pub speed_y: i16,
    pub speed_ground_x: i16,
    pub speed_attack_x: i16,
    pub speed_attack_y: i16,
    pub state_age: u16,
    pub hitlag: u8,
    pub stocks: u8,
    pub facing: u8,
    pub on_ground: u8,
    pub action_state: u16,
    pub jumps_left: u8,
    pub character: u8,
}

pub fn decode_output(
    model_output: &[i8],
    _d_model: usize,
) -> [DecodedPlayerState; 2] {
    let mut players = [
        DecodedPlayerState {
            x: 0, y: 0, percent: 0, shield_strength: 0,
            speed_air_x: 0, speed_y: 0, speed_ground_x: 0,
            speed_attack_x: 0, speed_attack_y: 0,
            state_age: 0, hitlag: 0, stocks: 4,
            facing: 1, on_ground: 1, action_state: 0, jumps_left: 2, character: 0,
        },
        DecodedPlayerState {
            x: 0, y: 0, percent: 0, shield_strength: 0,
            speed_air_x: 0, speed_y: 0, speed_ground_x: 0,
            speed_attack_x: 0, speed_attack_y: 0,
            state_age: 0, hitlag: 0, stocks: 4,
            facing: 0, on_ground: 1, action_state: 0, jumps_left: 2, character: 0,
        },
    ];

    let mut offset = 0;
    for p_idx in 0..2 {
        let p = &mut players[p_idx];

        // Continuous fields (dequantize from INT8)
        if offset < model_output.len() { p.x = model_output[offset] as i32 * 256; }
        offset += 1;
        if offset < model_output.len() { p.y = model_output[offset] as i32 * 256; }
        offset += 1;
        if offset < model_output.len() { p.percent = (model_output[offset] as i16 * 4).max(0) as u16; }
        offset += 1;
        if offset < model_output.len() { p.shield_strength = (model_output[offset] as u16); }
        offset += 1;
        if offset < model_output.len() { p.speed_air_x = model_output[offset] as i16 * 2; }
        offset += 1;
        if offset < model_output.len() { p.speed_y = model_output[offset] as i16 * 2; }
        offset += 1;
        if offset < model_output.len() { p.speed_ground_x = model_output[offset] as i16 * 2; }
        offset += 1;
        if offset < model_output.len() { p.speed_attack_x = model_output[offset] as i16 * 2; }
        offset += 1;
        if offset < model_output.len() { p.speed_attack_y = model_output[offset] as i16 * 2; }
        offset += 1;
        if offset < model_output.len() { p.state_age = model_output[offset] as u16; }
        offset += 1;
        if offset < model_output.len() { p.hitlag = model_output[offset].max(0) as u8; }
        offset += 1;
        if offset < model_output.len() { p.stocks = model_output[offset].max(0) as u8; }
        offset += 1;

        // Binary fields (threshold at 0)
        if offset < model_output.len() { p.facing = if model_output[offset] > 0 { 1 } else { 0 }; }
        offset += 1;
        if offset < model_output.len() { p.on_ground = if model_output[offset] > 0 { 1 } else { 0 }; }
        offset += 1;

        // Categorical (direct index for now; real implementation uses argmax over logit head)
        if offset < model_output.len() { p.action_state = model_output[offset].max(0) as u16; }
        offset += 1;
        if offset < model_output.len() { p.jumps_left = model_output[offset].max(0) as u8; }
        offset += 1;
        if offset < model_output.len() { p.character = model_output[offset].max(0) as u8; }
        offset += 1;

        // Skip controller input positions in output
        offset += 7;
    }

    players
}

/// Execute the full Mamba2 forward pass: all layers, encode → layers → decode.
///
/// This is the top-level function called by run_inference for each frame.
///
/// In the multi-tx pipeline variant, this function would be split:
///   - TX 1: encode + layers 0-3
///   - TX 2: layers 4-7
///   - TX 3: layers 8-11 + decode
/// Each TX reads/writes the HiddenState account.
pub fn forward_pass(
    input: &[i8],
    hidden_state: &mut [i8],
    weight_data: &[&[u8]],
    lut_data: &[u8],
    config: &Mamba2Config,
    layer_in_scales: &[&[u16]],
    layer_out_scales: &[&[u16]],
    norm_weights: &[&[u8]],
    a_logs: &[&[u8]],
    dt_biases: &[&[u8]],
) -> Vec<i8> {
    let d_model = config.d_model;
    let d_inner = config.d_inner;
    let d_state = config.d_state;
    let h_per_layer = d_inner * d_state;

    let mut x = input.to_vec();
    let mut scratch = ScratchBuffers::new(d_model, d_inner);

    for layer_idx in 0..config.num_layers {
        let h_offset = layer_idx * h_per_layer;
        let h_slice = &mut hidden_state[h_offset..h_offset + h_per_layer];

        // Compute weight offsets for this layer
        // in_proj: (2*d_inner, d_model) = 2*d_inner*d_model bytes
        // out_proj: (d_model, d_inner) = d_model*d_inner bytes
        let in_proj_size = 2 * d_inner * d_model;
        let out_proj_size = d_model * d_inner;
        let layer_weight_offset = layer_idx * (in_proj_size + out_proj_size);

        // Determine which shard this layer's weights are in
        let shard_idx = if layer_weight_offset < weight_data[0].len() { 0 } else { 1 };
        let shard = weight_data[shard_idx];
        let offset_in_shard = if shard_idx == 0 {
            layer_weight_offset
        } else {
            layer_weight_offset - weight_data[0].len()
        };

        let in_proj_end = (offset_in_shard + in_proj_size).min(shard.len());
        let out_proj_start = in_proj_end;
        let out_proj_end = (out_proj_start + out_proj_size).min(shard.len());

        let weights = LayerWeights {
            in_proj: &shard[offset_in_shard..in_proj_end],
            out_proj: &shard[out_proj_start..out_proj_end],
            norm: norm_weights.get(layer_idx).copied().unwrap_or(&[]),
            a_log: a_logs.get(layer_idx).copied().unwrap_or(&[]),
            dt_bias: dt_biases.get(layer_idx).copied().unwrap_or(&[]),
            in_proj_scales: layer_in_scales.get(layer_idx).copied().unwrap_or(&[]),
            out_proj_scales: layer_out_scales.get(layer_idx).copied().unwrap_or(&[]),
        };

        mamba2_layer_step(
            &mut x,
            h_slice,
            &weights,
            lut_data,
            config,
            &mut scratch,
        );
    }

    x
}
