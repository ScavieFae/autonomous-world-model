use bolt_component::*;

declare_id!("SessState1111111111111111111111111111111111");

/// Number of players per session
pub const NUM_PLAYERS: usize = 2;

/// Number of continuous state fields per player
/// x, y, percent, shield_strength, speed_air_x, speed_y, speed_ground_x,
/// speed_attack_x, speed_attack_y, state_age, hitlag, stocks
pub const NUM_CONTINUOUS: usize = 12;

/// Session status values
pub const STATUS_CREATED: u8 = 0;
pub const STATUS_WAITING_PLAYERS: u8 = 1;
pub const STATUS_ACTIVE: u8 = 2;
pub const STATUS_ENDED: u8 = 3;

/// Per-player state output from the world model.
///
/// Matches the v2 encoding from nojohns-training and the JSON format
/// consumed by viz/visualizer-juicy.html.
#[derive(Default, Clone, Copy, AnchorSerialize, AnchorDeserialize)]
pub struct PlayerState {
    // ── Continuous (regression heads) ───────────────────────────────────
    /// Horizontal position in game units
    pub x: i32,           // Fixed-point: actual = x / 256.0
    /// Vertical position in game units
    pub y: i32,           // Fixed-point: actual = y / 256.0
    /// Damage percent (0-999)
    pub percent: u16,
    /// Shield HP (0-60), fixed-point / 256
    pub shield_strength: u16,
    /// Aerial horizontal velocity, fixed-point / 256
    pub speed_air_x: i16,
    /// Vertical velocity, fixed-point / 256
    pub speed_y: i16,
    /// Ground horizontal velocity, fixed-point / 256
    pub speed_ground_x: i16,
    /// Knockback X velocity, fixed-point / 256
    pub speed_attack_x: i16,
    /// Knockback Y velocity, fixed-point / 256
    pub speed_attack_y: i16,
    /// Frames in current action state
    pub state_age: u16,
    /// Hitlag frames remaining
    pub hitlag: u8,
    /// Remaining stocks (0-4)
    pub stocks: u8,

    // ── Binary (classification) ─────────────────────────────────────────
    /// Direction facing (1 = right, 0 = left)
    pub facing: u8,
    /// Grounded state (1 = ground, 0 = airborne)
    pub on_ground: u8,

    // ── Categorical (classification heads) ──────────────────────────────
    /// Action state ID (0-399)
    pub action_state: u16,
    /// Aerial jumps remaining (0-7)
    pub jumps_left: u8,
    /// Character ID (0-32, internal Melee ID)
    pub character: u8,
}

/// Session state — the current frame of the autonomous world.
///
/// Updated every frame by run_inference. Clients subscribe to this account
/// via WebSocket to receive real-time state updates for rendering.
///
/// Lifecycle: Created per session in ephemeral rollup, committed to mainnet on end.
#[component]
#[derive(Default)]
pub struct SessionState {
    /// Session status (Created → WaitingPlayers → Active → Ended)
    pub status: u8,

    /// Current frame number (monotonically increasing)
    pub frame: u32,

    /// Maximum frames before auto-end (0 = unlimited, 28800 = 8 minutes at 60fps)
    pub max_frames: u32,

    /// Player 1 wallet public key
    pub player1: Pubkey,

    /// Player 2 wallet public key
    pub player2: Pubkey,

    /// Stage ID (0-32, matches Melee internal stage IDs)
    pub stage: u8,

    /// Per-player state (the world model's output for current frame)
    pub players: [PlayerState; NUM_PLAYERS],

    /// Reference to the ModelManifest used for this session
    pub model: Pubkey,

    /// Timestamp of session creation (Unix seconds)
    pub created_at: i64,

    /// Timestamp of last frame update
    pub last_update: i64,

    /// Session seed (for deterministic initialization)
    pub seed: u64,
}
