use anchor_lang::prelude::*;

declare_id!("AucQsnqWYXeVcig4puWFjnd8NXruCtjS8EVgA2B5KxUk");

/// Maximum number of layers supported
pub const MAX_LAYERS: usize = 16;

/// Maximum number of weight shards
pub const MAX_SHARDS: usize = 4;

/// LUT size: 256 entries per activation function
pub const LUT_SIZE: usize = 256;

/// Number of activation LUTs: SiLU, softplus, rsqrt, exp_neg
pub const NUM_LUTS: usize = 4;

/// Model manifest — the "cartridge label" of the autonomous world.
///
/// Contains everything needed to configure inference:
///   - Architecture parameters (d_model, d_inner, d_state, num_layers)
///   - References to weight shard accounts
///   - Per-layer quantization parameters (scale, zero-point)
///   - Activation function lookup tables
///   - Input/output encoding parameters
///
/// Lifecycle: Permanent on mainnet. Created once per model version.
/// Size: ~2KB (well within single account limits).
#[account]
pub struct ModelManifest {
    /// Human-readable model name (e.g., "melee-mamba2-v1")
    pub name: [u8; 32],

    /// Model version
    pub version: u16,

    // ── Architecture parameters ─────────────────────────────────────────

    /// Model dimension (embedding size)
    pub d_model: u16,

    /// Inner dimension (typically 2 × d_model)
    pub d_inner: u16,

    /// SSM state dimension
    pub d_state: u16,

    /// Number of Mamba2 layers
    pub num_layers: u8,

    /// Number of SSM heads (Mamba2 multi-head)
    pub num_heads: u8,

    // ── Weight shard references ─────────────────────────────────────────

    /// Number of weight shards
    pub num_shards: u8,

    /// Public keys of WeightShard accounts
    pub shard_keys: [Pubkey; MAX_SHARDS],

    /// Size of each shard in bytes
    pub shard_sizes: [u32; MAX_SHARDS],

    // ── Per-layer quantization parameters ───────────────────────────────
    // Each layer needs scale/zero-point for requantization between layers.
    // Stored as fixed-point: actual_scale = raw_value / 65536.0

    /// Per-layer input quantization scales (fixed-point u16)
    pub layer_input_scales: [u16; MAX_LAYERS],

    /// Per-layer output quantization scales (fixed-point u16)
    pub layer_output_scales: [u16; MAX_LAYERS],

    // ── Activation LUTs ─────────────────────────────────────────────────
    // 4 LUTs × 256 bytes = 1024 bytes total
    // Order: SiLU, softplus, rsqrt, exp_neg

    /// Packed activation lookup tables
    pub luts: [u8; LUT_SIZE * NUM_LUTS],

    // ── Input/Output encoding ───────────────────────────────────────────

    /// Number of continuous output fields per player
    pub num_continuous: u8,

    /// Number of action state classes
    pub num_action_states: u16,

    /// Number of binary output fields per player
    pub num_binary: u8,

    /// Input encoding size (controller inputs per frame)
    pub input_size: u16,

    // ── Model metadata ──────────────────────────────────────────────────

    /// Authority that created this manifest
    pub authority: Pubkey,

    /// Whether the model is ready for inference (all shards finalized)
    pub ready: bool,

    /// Total parameter count
    pub total_params: u32,

    /// Total INT8 weight bytes across all shards
    pub total_weight_bytes: u32,
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            name: [0u8; 32],
            version: 0,
            d_model: 0,
            d_inner: 0,
            d_state: 0,
            num_layers: 0,
            num_heads: 0,
            num_shards: 0,
            shard_keys: [Pubkey::default(); MAX_SHARDS],
            shard_sizes: [0u32; MAX_SHARDS],
            layer_input_scales: [0u16; MAX_LAYERS],
            layer_output_scales: [0u16; MAX_LAYERS],
            luts: [0u8; LUT_SIZE * NUM_LUTS],
            num_continuous: 0,
            num_action_states: 0,
            num_binary: 0,
            input_size: 0,
            authority: Pubkey::default(),
            ready: false,
            total_params: 0,
            total_weight_bytes: 0,
        }
    }
}
