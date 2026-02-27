use bolt_lang::*;
use frame_log::{CompressedFrame, FrameLog, RING_BUFFER_SIZE};
use hidden_state::HiddenState;
use input_buffer::InputBuffer;
use session_state::{PlayerState, SessionState, STATUS_ACTIVE};

pub mod lut;
pub mod matmul;
pub mod mamba2;

declare_id!("3tHPJJSNhKwbp7K5vSYCUdYVX9bGxRCmpddwaJWRKPyb");

#[error_code]
pub enum InferenceError {
    #[msg("Session is not active")]
    SessionNotActive,
    #[msg("Both players must submit inputs before inference")]
    InputsNotReady,
}

/// Run inference system — the heart of the autonomous world.
///
/// Executes one Mamba2 forward pass per call:
///   (controller_inputs, current_state, hidden_state) → (next_state, new_hidden_state)
///
/// Called by a cranker/scheduler at 60fps cadence (every 16.67ms).
///
/// Phase 3 implementation: STUB. Copies inputs through with default state changes.
/// Phase 4 will replace this with the real INT8 Mamba2 inference kernel.
///
/// Accounts read:
///   - InputBuffer: controller inputs for current frame
///   - SessionState: current world state
///   - HiddenState: Mamba2 recurrent state
///
/// Accounts written:
///   - SessionState: updated with new frame state
///   - HiddenState: updated recurrent state
///   - FrameLog: compressed frame appended to ring buffer
#[system]
pub mod run_inference {

    pub fn execute(ctx: Context<Components>, _args: Vec<u8>) -> Result<Components> {
        let session = &mut ctx.accounts.session_state;
        let hidden = &mut ctx.accounts.hidden_state;
        let input_buf = &ctx.accounts.input_buffer;
        let frame_log = &mut ctx.accounts.frame_log;

        // Validate session is active
        require!(
            session.status == STATUS_ACTIVE,
            InferenceError::SessionNotActive
        );

        // Validate inputs are ready
        require!(
            input_buf.p1_ready && input_buf.p2_ready,
            InferenceError::InputsNotReady
        );

        // ── STUB INFERENCE (Phase 3) ────────────────────────────────────
        // In Phase 4, this will be replaced with:
        //   1. Encode inputs (controller + current state → model input vector)
        //   2. For each layer: RMSNorm → in_proj → SSM step → gate → out_proj
        //   3. Decode output (model output → next PlayerState per player)
        //
        // For now: apply simple physics-like rules to demonstrate the pipeline.

        let frame = session.frame + 1;

        // Simple stub: apply controller inputs as velocity
        for player_idx in 0..2 {
            let input = if player_idx == 0 {
                &input_buf.player1
            } else {
                &input_buf.player2
            };

            let p = &mut session.players[player_idx];

            // Apply stick input as velocity (simplified physics)
            let stick_x = input.stick_x as i32;
            let stick_y = input.stick_y as i32;

            // Move based on stick (scale by 2 for reasonable speed)
            p.x += stick_x * 2;
            p.y += stick_y * 2;

            // Apply gravity if airborne
            if p.on_ground == 0 {
                p.speed_y -= 4; // Gravity (fixed-point)
                p.y += p.speed_y as i32;

                // Ground collision at y=0
                if p.y <= 0 {
                    p.y = 0;
                    p.speed_y = 0;
                    p.on_ground = 1;
                }
            }

            // Jump (button A)
            if input.buttons & 0x01 != 0 && p.jumps_left > 0 {
                p.speed_y = 40; // Jump velocity (fixed-point)
                p.on_ground = 0;
                p.jumps_left = p.jumps_left.saturating_sub(1);
            }

            // Update facing based on stick direction
            if stick_x > 10 {
                p.facing = 1;
            } else if stick_x < -10 {
                p.facing = 0;
            }

            // Update ground speed
            p.speed_ground_x = (stick_x * 2).clamp(-32767, 32767) as i16;

            // Increment state age
            p.state_age = p.state_age.saturating_add(1);
        }

        // ── END STUB ────────────────────────────────────────────────────

        // Update frame counter
        session.frame = frame;
        hidden.frame = frame;

        // Write to frame log ring buffer
        let _log_entry = compress_frame(frame, &session.players, session.stage, input_buf);
        let write_idx = (frame_log.write_index as usize) % RING_BUFFER_SIZE;
        // In production, write directly to account data via zero-copy:
        //   let offset = HEADER_SIZE + write_idx * COMPRESSED_FRAME_SIZE;
        //   account_data[offset..offset+COMPRESSED_FRAME_SIZE].copy_from_slice(&log_entry_bytes);
        // For now, just update metadata:
        frame_log.write_index = ((write_idx + 1) % RING_BUFFER_SIZE) as u16;
        frame_log.total_frames = frame;

        Ok(ctx.accounts)
    }

    #[system_input]
    pub struct Components {
        pub session_state: SessionState,
        pub hidden_state: HiddenState,
        pub input_buffer: InputBuffer,
        pub frame_log: FrameLog,
    }
    // Phase 4 will add:
    // pub model_manifest: ModelManifest,
    // pub weight_shard_0: WeightShard,
    // pub weight_shard_1: WeightShard,
}

/// Compress a full frame state into the compact ring buffer format.
fn compress_frame(
    frame: u32,
    players: &[PlayerState; 2],
    stage: u8,
    input: &Account<InputBuffer>,
) -> CompressedFrame {
    let p1 = &players[0];
    let p2 = &players[1];

    CompressedFrame {
        frame,
        // Player 1
        p1_x: (p1.x / 256) as i16,     // Convert from fixed-point
        p1_y: (p1.y / 256) as i16,
        p1_percent: p1.percent,
        p1_action_state: p1.action_state,
        p1_state_age: p1.state_age.min(255) as u8,
        p1_stocks: p1.stocks,
        p1_facing: p1.facing,
        p1_on_ground: p1.on_ground,
        p1_speed_x: (p1.speed_ground_x / 4).clamp(-128, 127) as i8,
        p1_speed_y: (p1.speed_y / 4).clamp(-128, 127) as i8,
        // Player 2
        p2_x: (p2.x / 256) as i16,
        p2_y: (p2.y / 256) as i16,
        p2_percent: p2.percent,
        p2_action_state: p2.action_state,
        p2_state_age: p2.state_age.min(255) as u8,
        p2_stocks: p2.stocks,
        p2_facing: p2.facing,
        p2_on_ground: p2.on_ground,
        p2_speed_x: (p2.speed_ground_x / 4).clamp(-128, 127) as i8,
        p2_speed_y: (p2.speed_y / 4).clamp(-128, 127) as i8,
        // Inputs (packed)
        p1_input_packed: pack_input(&input.player1),
        p2_input_packed: pack_input(&input.player2),
        stage,
    }
}

fn pack_input(input: &input_buffer::ControllerInput) -> u32 {
    ((input.stick_x as u8 as u32) << 24)
        | ((input.stick_y as u8 as u32) << 16)
        | ((input.c_stick_x as u8 as u32) << 8)
        | (input.buttons as u32)
}
