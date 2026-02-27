use anchor_lang::prelude::*;
use frame_log::FrameLog;
use hidden_state::HiddenState;
use session_state::{
    PlayerState, SessionState, STATUS_ACTIVE, STATUS_CREATED,
    STATUS_ENDED, STATUS_WAITING_PLAYERS,
};

declare_id!("4ozheJvvMhG7yMrp1UR2kq1fhRvjXoY5Pn3NJ4nvAcyE");

/// Lifecycle action codes
pub const ACTION_CREATE: u8 = 0;
pub const ACTION_JOIN: u8 = 1;
pub const ACTION_END: u8 = 2;

/// Session lifecycle system — manages session creation, joining, and ending.
///
/// The "console" metaphor:
///   CREATE = insert cartridge (select model, allocate accounts)
///   JOIN   = plug in controller (player 2 connects)
///   END    = power off (commit state to mainnet, reclaim rent)
///
/// Session flow:
///   1. Player 1 calls CREATE with model reference and character selection
///      → SessionState: Created → WaitingPlayers
///      → HiddenState: allocated and zeroed
///      → InputBuffer: allocated
///      → FrameLog: allocated
///      → All accounts delegated to ephemeral rollup
///
///   2. Player 2 calls JOIN with session ID and character selection
///      → SessionState: WaitingPlayers → Active
///      → Players' initial state set (start positions, 4 stocks, etc.)
///
///   3. Either player calls END (or auto-end after max_frames)
///      → SessionState: Active → Ended
///      → Accounts undelegated back to mainnet
///      → Session accounts closeable for rent reclaim
#[program]
pub mod session_lifecycle {
    use super::*;

    pub fn execute(
        ctx: Context<Components>,
        args: Args,
    ) -> Result<()> {
        let session_info = ctx.accounts.session_state.to_account_info();
        let hidden_info = ctx.accounts.hidden_state.to_account_info();
        let frame_log_info = ctx.accounts.frame_log.to_account_info();

        let mut session = load_component::<SessionState>(&session_info)?;
        let mut hidden = load_component::<HiddenState>(&hidden_info)?;
        let mut frame_log = load_component::<FrameLog>(&frame_log_info)?;

        match args.action {
            ACTION_CREATE => create_session(&mut session, &mut hidden, &mut frame_log, &args),
            ACTION_JOIN => join_session(&mut session, &args),
            ACTION_END => end_session(&mut session),
            _ => return Err(LifecycleError::InvalidAction.into()),
        }?;

        store_component(&session_info, &session)?;
        store_component(&hidden_info, &hidden)?;
        store_component(&frame_log_info, &frame_log)?;

        Ok(())
    }
}

#[derive(Accounts)]
pub struct Components<'info> {
    #[account(mut)]
    pub session_state: UncheckedAccount<'info>,
    #[account(mut)]
    pub hidden_state: UncheckedAccount<'info>,
    #[account(mut)]
    pub input_buffer: UncheckedAccount<'info>,
    #[account(mut)]
    pub frame_log: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct Args {
    /// Action: 0=create, 1=join, 2=end
    pub action: u8,
    /// Player public key
    pub player: Pubkey,
    /// Character ID (0-32) for the joining player
    pub character: u8,
    /// Stage ID (0-32) — only used on CREATE
    pub stage: u8,
    /// Model manifest public key — only used on CREATE
    pub model: Pubkey,
    /// Max frames (0 = unlimited) — only used on CREATE
    pub max_frames: u32,
    /// Session seed for deterministic init — only used on CREATE
    pub seed: u64,
    /// Model d_inner — used to configure hidden state on CREATE
    pub d_inner: u16,
    /// Model d_state — used to configure hidden state on CREATE
    pub d_state: u16,
    /// Model num_layers — used to configure hidden state on CREATE
    pub num_layers: u8,
}

fn create_session(
    session: &mut SessionState,
    hidden: &mut HiddenState,
    frame_log: &mut FrameLog,
    args: &Args,
) -> Result<()> {
    // Can only create from initial state
    require!(
        session.status == STATUS_CREATED || session.status == 0,
        LifecycleError::InvalidStateTransition
    );

    // Initialize session
    session.status = STATUS_WAITING_PLAYERS;
    session.frame = 0;
    session.max_frames = args.max_frames;
    session.player1 = args.player;
    session.player2 = Pubkey::default(); // Empty until join
    session.stage = args.stage;
    session.model = args.model;
    session.seed = args.seed;

    // Set player 1's character
    session.players[0] = PlayerState::default();
    session.players[0].character = args.character;
    session.players[0].stocks = 4;

    // Initialize hidden state dimensions
    hidden.num_layers = args.num_layers;
    hidden.d_inner = args.d_inner;
    hidden.d_state = args.d_state;
    hidden.data_size = (args.num_layers as u32) * (args.d_inner as u32) * (args.d_state as u32);
    hidden.frame = 0;
    hidden.initialized = false;

    // Initialize frame log
    frame_log.write_index = 0;
    frame_log.total_frames = 0;

    // Clock timestamp would be set here in production:
    // session.created_at = Clock::get()?.unix_timestamp;

    msg!("Session created: player1={}, stage={}, model={}",
         args.player, args.stage, args.model);
    Ok(())
}

fn join_session(
    session: &mut SessionState,
    args: &Args,
) -> Result<()> {
    require!(
        session.status == STATUS_WAITING_PLAYERS,
        LifecycleError::InvalidStateTransition
    );

    require!(
        args.player != session.player1,
        LifecycleError::CannotJoinOwnSession
    );

    // Set player 2
    session.player2 = args.player;
    session.players[1] = PlayerState::default();
    session.players[1].character = args.character;
    session.players[1].stocks = 4;

    // Set initial positions (stage-dependent, using FD defaults)
    // Player 1: left side, Player 2: right side
    // Fixed-point: multiply by 256
    session.players[0].x = -30 * 256;  // -30.0 game units
    session.players[0].y = 0;
    session.players[0].facing = 1;     // Facing right
    session.players[0].on_ground = 1;
    session.players[0].jumps_left = 2;
    session.players[0].shield_strength = 60 * 256;

    session.players[1].x = 30 * 256;   // 30.0 game units
    session.players[1].y = 0;
    session.players[1].facing = 0;     // Facing left
    session.players[1].on_ground = 1;
    session.players[1].jumps_left = 2;
    session.players[1].shield_strength = 60 * 256;

    // Activate session
    session.status = STATUS_ACTIVE;
    // session.last_update = Clock::get()?.unix_timestamp;

    msg!("Player 2 joined: player2={}, character={}", args.player, args.character);
    msg!("Session ACTIVE — game on!");
    Ok(())
}

fn end_session(session: &mut SessionState) -> Result<()> {
    require!(
        session.status == STATUS_ACTIVE || session.status == STATUS_WAITING_PLAYERS,
        LifecycleError::InvalidStateTransition
    );

    session.status = STATUS_ENDED;
    msg!("Session ended at frame {}", session.frame);

    // In production:
    // - Undelegate all session accounts back to mainnet
    // - Mark accounts as closeable for rent reclaim
    // - Emit final state as event for indexers

    Ok(())
}

#[error_code]
pub enum LifecycleError {
    #[msg("Invalid lifecycle action code")]
    InvalidAction,
    #[msg("Invalid state transition for current session status")]
    InvalidStateTransition,
    #[msg("Cannot join your own session")]
    CannotJoinOwnSession,
    #[msg("Failed to deserialize component data")]
    DeserializeFailed,
    #[msg("Failed to serialize component data")]
    SerializeFailed,
}

fn load_component<T: AnchorDeserialize + Default>(info: &AccountInfo) -> Result<T> {
    let data = info.try_borrow_data()?;
    if data.len() <= 8 {
        return Ok(T::default());
    }

    let mut slice: &[u8] = &data[8..];
    T::deserialize(&mut slice).map_err(|_| LifecycleError::DeserializeFailed.into())
}

fn store_component<T: AnchorSerialize>(info: &AccountInfo, value: &T) -> Result<()> {
    let mut data = info.try_borrow_mut_data()?;
    if data.len() <= 8 {
        return Err(LifecycleError::SerializeFailed.into());
    }

    let mut dst = &mut data[8..];
    value
        .serialize(&mut dst)
        .map_err(|_| LifecycleError::SerializeFailed.into())
}
