use anchor_lang::prelude::*;

// ── Constants ────────────────────────────────────────────────────────────────

pub const MAX_LAYERS: usize = 16;
pub const MAX_SHARDS: usize = 4;
pub const LUT_TOTAL_SIZE: usize = crate::lut::LUT_TOTAL_SIZE;
pub const NUM_PLAYERS: usize = 2;
pub const MAX_CHUNK_SIZE: usize = 1000;

/// Session status values
pub const STATUS_WAITING_PLAYERS: u8 = 1;
pub const STATUS_ACTIVE: u8 = 2;
pub const STATUS_ENDED: u8 = 3;

// ── ModelManifestAccount ─────────────────────────────────────────────────────

/// Model manifest — the "cartridge label" of the autonomous world.
///
/// Contains architecture params, weight shard references, quantization scales,
/// and activation LUTs. Created once per model version. ~2KB.
#[account]
#[derive(Default)]
pub struct ModelManifestAccount {
    /// Human-readable model name (e.g., "melee-mamba2-v1")
    pub name: [u8; 32],

    /// Model version
    pub version: u16,

    // ── Architecture parameters ──────────────────────────────────────────
    pub d_model: u16,
    pub d_inner: u16,
    pub d_state: u16,
    pub num_layers: u8,
    pub num_heads: u8,

    // ── Weight shard references ──────────────────────────────────────────
    pub num_shards: u8,
    pub shard_keys: [Pubkey; MAX_SHARDS],
    pub shard_sizes: [u32; MAX_SHARDS],

    // ── Per-layer quantization parameters ────────────────────────────────
    pub layer_input_scales: [u16; MAX_LAYERS],
    pub layer_output_scales: [u16; MAX_LAYERS],

    // ── Activation LUTs (4 × 256 = 1024 bytes) ──────────────────────────
    pub luts: [u8; LUT_TOTAL_SIZE],

    // ── Input/Output encoding ────────────────────────────────────────────
    pub num_continuous: u8,
    pub num_action_states: u16,
    pub num_binary: u8,
    pub input_size: u16,

    // ── Metadata ─────────────────────────────────────────────────────────
    pub authority: Pubkey,
    pub ready: bool,
    pub total_params: u32,
    pub total_weight_bytes: u32,
}

// ── WeightAccount ────────────────────────────────────────────────────────────

/// Weight account header — typed access to the structured header.
/// Actual INT8 weight data lives past this header in raw account data.
#[account]
#[derive(Default)]
pub struct WeightAccount {
    pub shard_index: u8,
    pub data_size: u32,
    pub authority: Pubkey,
    pub finalized: bool,
    pub data_hash: [u8; 32],
    pub bytes_written: u32,
}

/// Header size: 8 (discriminator) + 1 + 4 + 32 + 1 + 32 + 4 = 82 bytes
pub const WEIGHT_HEADER_SIZE: usize = 82;

// ── PlayerState ──────────────────────────────────────────────────────────────

/// Per-player state output from the world model.
/// Matches the v2 encoding from nojohns-training.
#[derive(Default, Clone, Copy, AnchorSerialize, AnchorDeserialize)]
pub struct PlayerState {
    // ── Continuous (regression heads) ────────────────────────────────────
    pub x: i32,                 // Fixed-point: actual = x / 256.0
    pub y: i32,                 // Fixed-point: actual = y / 256.0
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

    // ── Binary (classification) ──────────────────────────────────────────
    pub facing: u8,
    pub on_ground: u8,

    // ── Categorical (classification heads) ───────────────────────────────
    pub action_state: u16,
    pub jumps_left: u8,
    pub character: u8,
}

// ── SessionStateAccount ──────────────────────────────────────────────────────

/// Session state — the current frame of the autonomous world.
/// Updated every frame by run_inference.
#[account]
#[derive(Default)]
pub struct SessionStateAccount {
    pub status: u8,
    pub frame: u32,
    pub max_frames: u32,
    pub player1: Pubkey,
    pub player2: Pubkey,
    pub stage: u8,
    pub players: [PlayerState; NUM_PLAYERS],
    pub model: Pubkey,
    pub created_at: i64,
    pub last_update: i64,
    pub seed: u64,
}

// ── ControllerInput ──────────────────────────────────────────────────────────

/// Melee controller input for one player (8 bytes).
#[derive(Default, Clone, Copy, AnchorSerialize, AnchorDeserialize)]
pub struct ControllerInput {
    pub stick_x: i8,
    pub stick_y: i8,
    pub c_stick_x: i8,
    pub c_stick_y: i8,
    pub trigger_l: u8,
    pub trigger_r: u8,
    pub buttons: u8,
    pub buttons_ext: u8,
}

// ── InputBufferAccount ───────────────────────────────────────────────────────

/// Input buffer — controller inputs for the current frame.
/// Both players submit inputs, then inference reads this buffer.
#[account]
#[derive(Default)]
pub struct InputBufferAccount {
    pub frame: u32,
    pub player1: ControllerInput,
    pub player2: ControllerInput,
    pub p1_ready: bool,
    pub p2_ready: bool,
}

// ── Hidden state constants ───────────────────────────────────────────────────

/// Hidden state is accessed via raw AccountInfo (too large for Borsh).
/// Layout: [header (16 bytes)] [h_data (num_layers * d_inner * d_state bytes)]
///
/// Header:
///   - num_layers: u8     (offset 0)
///   - d_inner: u16 LE    (offset 1)
///   - d_state: u16 LE    (offset 3)
///   - data_size: u32 LE  (offset 5)
///   - frame: u32 LE      (offset 9)
///   - initialized: u8    (offset 13)
///   - padding: [u8; 2]   (offset 14)
pub const HIDDEN_HEADER_SIZE: usize = 16;

/// Read hidden state header fields from raw account data.
pub fn read_hidden_header(data: &[u8]) -> (u8, u16, u16, u32, u32, bool) {
    let num_layers = data[0];
    let d_inner = u16::from_le_bytes([data[1], data[2]]);
    let d_state = u16::from_le_bytes([data[3], data[4]]);
    let data_size = u32::from_le_bytes([data[5], data[6], data[7], data[8]]);
    let frame = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);
    let initialized = data[13] != 0;
    (num_layers, d_inner, d_state, data_size, frame, initialized)
}

/// Write hidden state header fields to raw account data.
pub fn write_hidden_header(
    data: &mut [u8],
    num_layers: u8,
    d_inner: u16,
    d_state: u16,
    data_size: u32,
    frame: u32,
    initialized: bool,
) {
    data[0] = num_layers;
    data[1..3].copy_from_slice(&d_inner.to_le_bytes());
    data[3..5].copy_from_slice(&d_state.to_le_bytes());
    data[5..9].copy_from_slice(&data_size.to_le_bytes());
    data[9..13].copy_from_slice(&frame.to_le_bytes());
    data[13] = initialized as u8;
    data[14] = 0;
    data[15] = 0;
}
