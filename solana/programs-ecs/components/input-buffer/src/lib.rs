use bolt_lang::*;

declare_id!("3R2RbzwP54qdyXcyiwHW2Sj6uVwf4Dhy7Zy8RcSVHFpq");

/// Melee controller input for one player.
///
/// Matches the GCC (GameCube Controller) input format:
///   - Main stick: X/Y axes (-128 to 127)
///   - C-stick: X/Y axes (-128 to 127)
///   - Triggers: L/R analog (0-255)
///   - Buttons: digital bitmask
///
/// Total: 8 bytes per player, 16 bytes per frame.
#[component_deserialize]
#[derive(Default)]
pub struct ControllerInput {
    /// Main stick X axis (-128 = full left, 127 = full right)
    pub stick_x: i8,
    /// Main stick Y axis (-128 = full down, 127 = full up)
    pub stick_y: i8,
    /// C-stick X axis
    pub c_stick_x: i8,
    /// C-stick Y axis
    pub c_stick_y: i8,
    /// Left trigger analog (0 = released, 255 = full press)
    pub trigger_l: u8,
    /// Right trigger analog
    pub trigger_r: u8,
    /// Digital button bitmask:
    ///   bit 0: A, bit 1: B, bit 2: X, bit 3: Y,
    ///   bit 4: Z, bit 5: Start, bit 6: D-left, bit 7: D-right
    pub buttons: u8,
    /// Extended buttons (D-up, D-down, L digital, R digital)
    pub buttons_ext: u8,
}

/// Input buffer — controller inputs for the current frame.
///
/// Both players submit their inputs via submit_input, then run_inference
/// reads this buffer to produce the next frame state.
///
/// Lifecycle: Per-session, overwritten every frame.
/// Size: ~20 bytes (tiny — just two controller states + metadata).
#[component]
#[derive(Default)]
pub struct InputBuffer {
    /// Frame number these inputs are for
    pub frame: u32,

    /// Player 1 input
    pub player1: ControllerInput,

    /// Player 2 input
    pub player2: ControllerInput,

    /// Whether player 1 has submitted input for this frame
    pub p1_ready: bool,

    /// Whether player 2 has submitted input for this frame
    pub p2_ready: bool,
}
