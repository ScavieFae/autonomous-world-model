use bolt_component::*;

declare_id!("FrameLog11111111111111111111111111111111111");

/// Number of frames in the ring buffer
pub const RING_BUFFER_SIZE: usize = 256;

/// Compressed frame entry for the ring buffer.
///
/// Stores essential state for replay/spectating at ~66 bytes per frame.
/// 256 frames × 66 bytes = ~17KB total.
///
/// Uses delta encoding where possible to save space:
///   - Positions: absolute (needed for rendering)
///   - Velocities: quantized to i8 (less precision ok for replay)
///   - Action state: absolute (needed for animation lookup)
#[derive(Default, Clone, Copy, AnchorSerialize, AnchorDeserialize)]
pub struct CompressedFrame {
    /// Frame number
    pub frame: u32,

    // ── Player 1 ────────────────────────────────────────────────────────
    pub p1_x: i16,           // Position quantized to i16 (±128 game units)
    pub p1_y: i16,
    pub p1_percent: u16,
    pub p1_action_state: u16,
    pub p1_state_age: u8,    // Capped at 255
    pub p1_stocks: u8,
    pub p1_facing: u8,
    pub p1_on_ground: u8,
    pub p1_speed_x: i8,      // Velocity quantized to i8
    pub p1_speed_y: i8,

    // ── Player 2 ────────────────────────────────────────────────────────
    pub p2_x: i16,
    pub p2_y: i16,
    pub p2_percent: u16,
    pub p2_action_state: u16,
    pub p2_state_age: u8,
    pub p2_stocks: u8,
    pub p2_facing: u8,
    pub p2_on_ground: u8,
    pub p2_speed_x: i8,
    pub p2_speed_y: i8,

    /// Controller inputs (packed: stick_x, stick_y, buttons for each player)
    pub p1_input_packed: u32,  // stick_x(8) | stick_y(8) | c_x(8) | buttons(8)
    pub p2_input_packed: u32,

    /// Stage ID
    pub stage: u8,
}

/// Frame log — ring buffer of recent frames for spectating and replay.
///
/// Stores the last 256 frames (~4.3 seconds at 60fps) in a compressed format.
/// Spectators can read this account to render recent history without needing
/// to subscribe from frame 0.
///
/// Also serves as the replay data committed to mainnet when the session ends —
/// the permanent record of what happened in this world.
///
/// Lifecycle: Per-session, written every frame by run_inference.
#[component]
#[derive(Default)]
pub struct FrameLog {
    /// Write index in the ring buffer (wraps at RING_BUFFER_SIZE)
    pub write_index: u16,

    /// Total frames written (may exceed RING_BUFFER_SIZE)
    pub total_frames: u32,

    /// Session ID reference
    pub session: Pubkey,

    // The actual ring buffer data is stored in the account's remaining space:
    //   frames: [CompressedFrame; RING_BUFFER_SIZE]
    //
    // At ~66 bytes per frame × 256 frames = ~16,896 bytes
    // Accessed via zero-copy by index: data[header_size + (index % 256) * frame_size]
}
