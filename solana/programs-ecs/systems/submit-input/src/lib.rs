use anchor_lang::prelude::*;
use input_buffer::{ControllerInput, InputBuffer};
use session_state::{SessionState, STATUS_ACTIVE};

declare_id!("F9ZqWHVDtsXZdHLU8MXfybsS1W3TTGv4NegcJZK9LnWx");

/// Submit input system — receives controller inputs from a player.
///
/// Called by each player once per frame. When both players have submitted,
/// the input buffer is ready for run_inference.
///
/// Flow:
///   1. Player signs a tx calling submit_input with their ControllerInput
///   2. System validates player identity (must match session's player1 or player2)
///   3. Writes input to the correct slot in InputBuffer
///   4. Sets the ready flag for that player
///
/// In the ephemeral rollup, this tx is sent via WebSocket for minimal latency.
/// Expected cadence: 60 calls per second per player (16.67ms intervals).
#[program]
pub mod submit_input {
    use super::*;

    pub fn execute(
        ctx: Context<Components>,
        args: Args,
    ) -> Result<()> {
        let session_info = ctx.accounts.session_state.to_account_info();
        let input_info = ctx.accounts.input_buffer.to_account_info();

        let session = load_component::<SessionState>(&session_info)?;
        let mut input_buf = load_component::<InputBuffer>(&input_info)?;

        // Validate session is active
        require!(
            session.status == STATUS_ACTIVE,
            InputError::SessionNotActive
        );

        // Determine which player is submitting
        let player = args.player;
        let is_p1 = player == session.player1;
        let is_p2 = player == session.player2;

        require!(
            is_p1 || is_p2,
            InputError::UnauthorizedPlayer
        );

        // Build controller input from args
        let controller = ControllerInput {
            stick_x: args.stick_x,
            stick_y: args.stick_y,
            c_stick_x: args.c_stick_x,
            c_stick_y: args.c_stick_y,
            trigger_l: args.trigger_l,
            trigger_r: args.trigger_r,
            buttons: args.buttons,
            buttons_ext: args.buttons_ext,
        };

        // Write to correct player slot
        if is_p1 {
            input_buf.player1 = controller;
            input_buf.p1_ready = true;
        } else {
            input_buf.player2 = controller;
            input_buf.p2_ready = true;
        }

        // Update frame number if this is a new frame
        let expected_frame = session.frame + 1;
        if input_buf.frame != expected_frame {
            input_buf.frame = expected_frame;
            // Reset ready flags for new frame (the player who submitted
            // first is already marked ready above)
            if is_p1 {
                input_buf.p2_ready = false;
            } else {
                input_buf.p1_ready = false;
            }
        }

        store_component(&input_info, &input_buf)?;
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Components<'info> {
    #[account()]
    pub session_state: UncheckedAccount<'info>,
    #[account(mut)]
    pub input_buffer: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct Args {
    /// Public key of the submitting player (verified against session)
    pub player: Pubkey,
    pub stick_x: i8,
    pub stick_y: i8,
    pub c_stick_x: i8,
    pub c_stick_y: i8,
    pub trigger_l: u8,
    pub trigger_r: u8,
    pub buttons: u8,
    pub buttons_ext: u8,
}

#[error_code]
pub enum InputError {
    #[msg("Session is not active")]
    SessionNotActive,
    #[msg("Player is not part of this session")]
    UnauthorizedPlayer,
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
    T::deserialize(&mut slice).map_err(|_| InputError::DeserializeFailed.into())
}

fn store_component<T: AnchorSerialize>(info: &AccountInfo, value: &T) -> Result<()> {
    let mut data = info.try_borrow_mut_data()?;
    if data.len() <= 8 {
        return Err(InputError::SerializeFailed.into());
    }

    let mut dst = &mut data[8..];
    value
        .serialize(&mut dst)
        .map_err(|_| InputError::SerializeFailed.into())
}
