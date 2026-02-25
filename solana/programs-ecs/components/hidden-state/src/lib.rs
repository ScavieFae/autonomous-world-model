use bolt_component::*;

declare_id!("HdnState11111111111111111111111111111111111");

/// Maximum model dimensions (for sizing the hidden state buffer)
/// d_inner=1024, d_state=16, num_layers=12 → 1024*16*12 = 196,608 bytes per hidden state
/// With INT8 storage: 196,608 bytes ≈ 192KB
/// With metadata overhead: ~200KB
pub const MAX_HIDDEN_SIZE: usize = 200_000;

/// Mamba2 SSM recurrent hidden state — the "memory" of the world.
///
/// This is the fixed-size buffer that carries temporal context between frames.
/// Unlike transformers (growing KV cache), Mamba2's state is constant size:
///   h[layer][d_inner][d_state] — one matrix per layer
///
/// At ~200KB, this is the largest per-session account. It needs its own
/// account separate from SessionState to keep the session state small
/// for frequent reads by clients.
///
/// Lifecycle: Created per session, mutated every frame by run_inference,
/// committed to mainnet on session end.
///
/// The hidden state IS the world's memory. After 1000 frames, this buffer
/// contains a compressed representation of everything that happened —
/// every hit, every dodge, every stock taken. It's the Mamba2 equivalent
/// of "experience."
#[component]
#[derive(Default)]
pub struct HiddenState {
    /// Number of layers in the model
    pub num_layers: u8,

    /// Inner dimension per layer
    pub d_inner: u16,

    /// State dimension per layer
    pub d_state: u16,

    /// Total bytes of hidden state data
    pub data_size: u32,

    /// Frame number this state corresponds to
    /// (should match SessionState.frame)
    pub frame: u32,

    /// Whether the state has been initialized (zeroed on first frame)
    pub initialized: bool,

    // The actual hidden state data is stored in the account's remaining space
    // after this header, accessed via zero-copy:
    //
    //   Layout: [layer_0_h][layer_1_h]...[layer_N_h]
    //   Each layer_h: d_inner × d_state INT8 values
    //   Total: num_layers × d_inner × d_state bytes
    //
    // The inference system reads/writes this region directly without
    // deserializing — pure zero-copy access for performance.
}
