use anchor_lang::prelude::*;

pub mod error;
pub mod inference;
pub mod lut;
pub mod matmul;
pub mod ssm;
pub mod state;

use error::WorldModelError;
use state::*;

declare_id!("WrLd1111111111111111111111111111111111111111");

#[program]
pub mod world_model {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. init_manifest — populate the model "cartridge label"
    // ═══════════════════════════════════════════════════════════════════════

    pub fn init_manifest(
        ctx: Context<InitManifest>,
        name: [u8; 32],
        version: u16,
        d_model: u16,
        d_inner: u16,
        d_state: u16,
        num_layers: u8,
        num_heads: u8,
        luts: [u8; LUT_TOTAL_SIZE],
        num_continuous: u8,
        num_action_states: u16,
        num_binary: u8,
        input_size: u16,
        total_params: u32,
        total_weight_bytes: u32,
    ) -> Result<()> {
        let manifest = &mut ctx.accounts.manifest;

        manifest.name = name;
        manifest.version = version;
        manifest.d_model = d_model;
        manifest.d_inner = d_inner;
        manifest.d_state = d_state;
        manifest.num_layers = num_layers;
        manifest.num_heads = num_heads;
        manifest.luts = luts;
        manifest.num_continuous = num_continuous;
        manifest.num_action_states = num_action_states;
        manifest.num_binary = num_binary;
        manifest.input_size = input_size;
        manifest.total_params = total_params;
        manifest.total_weight_bytes = total_weight_bytes;
        manifest.authority = ctx.accounts.authority.key();
        manifest.ready = false;
        manifest.num_shards = 0;

        msg!("Manifest initialized: d_model={}, d_inner={}, layers={}",
             d_model, d_inner, num_layers);
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. upload_weights — chunked weight upload with finalization
    // ═══════════════════════════════════════════════════════════════════════

    pub fn upload_weights(
        ctx: Context<UploadWeights>,
        offset: u32,
        data: Vec<u8>,
    ) -> Result<()> {
        let weight = &mut ctx.accounts.weight_account;

        require!(
            ctx.accounts.authority.key() == weight.authority,
            WorldModelError::Unauthorized
        );
        require!(!weight.finalized, WorldModelError::AlreadyFinalized);
        require!(data.len() <= MAX_CHUNK_SIZE, WorldModelError::ChunkTooLarge);

        let offset = offset as usize;
        let end = offset + data.len();
        require!(
            end <= weight.data_size as usize,
            WorldModelError::ChunkOutOfBounds
        );

        // Write to raw account data past the header
        let weight_data = &ctx.accounts.weight_data;
        let mut account_data = weight_data.try_borrow_mut_data()?;
        let dest = &mut account_data[WEIGHT_HEADER_SIZE + offset..WEIGHT_HEADER_SIZE + end];
        dest.copy_from_slice(&data);

        // Track high-water mark
        let new_written = end as u32;
        if new_written > weight.bytes_written {
            weight.bytes_written = new_written;
        }

        Ok(())
    }

    pub fn finalize_weights(
        ctx: Context<FinalizeWeights>,
        expected_hash: [u8; 32],
    ) -> Result<()> {
        let weight = &mut ctx.accounts.weight_account;

        require!(
            ctx.accounts.authority.key() == weight.authority,
            WorldModelError::Unauthorized
        );
        require!(!weight.finalized, WorldModelError::AlreadyFinalized);
        require!(
            weight.bytes_written >= weight.data_size,
            WorldModelError::IncompleteUpload
        );

        // Verify hash of data region
        let weight_data = &ctx.accounts.weight_data;
        let account_data = weight_data.try_borrow_data()?;
        let data_region = &account_data[WEIGHT_HEADER_SIZE..WEIGHT_HEADER_SIZE + weight.data_size as usize];
        let hash = solana_program::hash::hash(data_region);

        require!(
            hash.to_bytes() == expected_hash,
            WorldModelError::HashMismatch
        );

        weight.finalized = true;
        weight.data_hash = expected_hash;

        msg!("Weight shard {} finalized ({} bytes, hash verified)",
             weight.shard_index, weight.data_size);
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. create_session — insert cartridge, allocate session accounts
    // ═══════════════════════════════════════════════════════════════════════

    pub fn create_session(
        ctx: Context<CreateSession>,
        stage: u8,
        character: u8,
        max_frames: u32,
        seed: u64,
    ) -> Result<()> {
        let session = &mut ctx.accounts.session;
        let manifest = &ctx.accounts.manifest;

        // Initialize session state
        session.status = STATUS_WAITING_PLAYERS;
        session.frame = 0;
        session.max_frames = max_frames;
        session.player1 = ctx.accounts.player1.key();
        session.player2 = Pubkey::default();
        session.stage = stage;
        session.model = manifest.key();
        session.seed = seed;

        // Set player 1 defaults
        session.players[0] = PlayerState::default();
        session.players[0].character = character;
        session.players[0].stocks = 4;

        // Initialize hidden state header (raw AccountInfo)
        let hidden = &ctx.accounts.hidden_state;
        let mut h_data = hidden.try_borrow_mut_data()?;
        let d_inner = manifest.d_inner;
        let d_state = manifest.d_state;
        let num_layers = manifest.num_layers;
        let data_size = (num_layers as u32) * (d_inner as u32) * (d_state as u32);
        write_hidden_header(
            &mut h_data,
            num_layers,
            d_inner,
            d_state,
            data_size,
            0,     // frame
            false, // initialized
        );

        // Initialize input buffer
        let input_buf = &mut ctx.accounts.input_buffer;
        input_buf.frame = 0;
        input_buf.p1_ready = false;
        input_buf.p2_ready = false;

        msg!("Session created: player1={}, stage={}", ctx.accounts.player1.key(), stage);
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. join_session — plug in controller, activate game
    // ═══════════════════════════════════════════════════════════════════════

    pub fn join_session(
        ctx: Context<JoinSession>,
        character: u8,
    ) -> Result<()> {
        let session = &mut ctx.accounts.session;

        require!(
            session.status == STATUS_WAITING_PLAYERS,
            WorldModelError::InvalidStateTransition
        );
        require!(
            ctx.accounts.player2.key() != session.player1,
            WorldModelError::CannotJoinOwnSession
        );

        // Set player 2
        session.player2 = ctx.accounts.player2.key();
        session.players[1] = PlayerState::default();
        session.players[1].character = character;
        session.players[1].stocks = 4;

        // Set initial positions (FD defaults)
        session.players[0].x = -30 * 256;
        session.players[0].y = 0;
        session.players[0].facing = 1;
        session.players[0].on_ground = 1;
        session.players[0].jumps_left = 2;
        session.players[0].shield_strength = 60 * 256;

        session.players[1].x = 30 * 256;
        session.players[1].y = 0;
        session.players[1].facing = 0;
        session.players[1].on_ground = 1;
        session.players[1].jumps_left = 2;
        session.players[1].shield_strength = 60 * 256;

        session.status = STATUS_ACTIVE;

        msg!("Player 2 joined: character={}. Session ACTIVE!", character);
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. close_session — power off, end game
    // ═══════════════════════════════════════════════════════════════════════

    pub fn close_session(
        ctx: Context<CloseSession>,
    ) -> Result<()> {
        let session = &mut ctx.accounts.session;

        require!(
            session.status == STATUS_ACTIVE || session.status == STATUS_WAITING_PLAYERS,
            WorldModelError::InvalidStateTransition
        );

        // Verify the closer is a participant
        let player_key = ctx.accounts.player.key();
        require!(
            player_key == session.player1 || player_key == session.player2,
            WorldModelError::UnauthorizedPlayer
        );

        session.status = STATUS_ENDED;
        msg!("Session ended at frame {}", session.frame);
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. submit_input — receive controller input from a player
    // ═══════════════════════════════════════════════════════════════════════

    pub fn submit_input(
        ctx: Context<SubmitInput>,
        stick_x: i8,
        stick_y: i8,
        c_stick_x: i8,
        c_stick_y: i8,
        trigger_l: u8,
        trigger_r: u8,
        buttons: u8,
        buttons_ext: u8,
    ) -> Result<()> {
        let session = &ctx.accounts.session;
        let input_buf = &mut ctx.accounts.input_buffer;
        let player_key = ctx.accounts.player.key();

        require!(
            session.status == STATUS_ACTIVE,
            WorldModelError::SessionNotActive
        );

        let is_p1 = player_key == session.player1;
        let is_p2 = player_key == session.player2;
        require!(
            is_p1 || is_p2,
            WorldModelError::UnauthorizedPlayer
        );

        let controller = ControllerInput {
            stick_x,
            stick_y,
            c_stick_x,
            c_stick_y,
            trigger_l,
            trigger_r,
            buttons,
            buttons_ext,
        };

        if is_p1 {
            input_buf.player1 = controller;
            input_buf.p1_ready = true;
        } else {
            input_buf.player2 = controller;
            input_buf.p2_ready = true;
        }

        // Reset other player's ready flag on new frame
        let expected_frame = session.frame + 1;
        if input_buf.frame != expected_frame {
            input_buf.frame = expected_frame;
            if is_p1 {
                input_buf.p2_ready = false;
            } else {
                input_buf.p1_ready = false;
            }
        }

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 7. run_inference — the heart of the autonomous world
    // ═══════════════════════════════════════════════════════════════════════

    pub fn run_inference(
        ctx: Context<RunInference>,
    ) -> Result<()> {
        let session = &mut ctx.accounts.session;
        let input_buf = &ctx.accounts.input_buffer;

        require!(
            session.status == STATUS_ACTIVE,
            WorldModelError::SessionNotActive
        );
        require!(
            input_buf.p1_ready && input_buf.p2_ready,
            WorldModelError::InputsNotReady
        );

        // ── STUB INFERENCE ──────────────────────────────────────────────
        // Phase 4 will replace this with real Mamba2 forward pass.
        // For now: apply simple physics-like rules to demonstrate the pipeline.

        let frame = session.frame + 1;

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

            p.x += stick_x * 2;
            p.y += stick_y * 2;

            // Gravity if airborne
            if p.on_ground == 0 {
                p.speed_y -= 4;
                p.y += p.speed_y as i32;

                if p.y <= 0 {
                    p.y = 0;
                    p.speed_y = 0;
                    p.on_ground = 1;
                }
            }

            // Jump (button A = bit 0)
            if input.buttons & 0x01 != 0 && p.jumps_left > 0 {
                p.speed_y = 40;
                p.on_ground = 0;
                p.jumps_left = p.jumps_left.saturating_sub(1);
            }

            // Facing direction
            if stick_x > 10 {
                p.facing = 1;
            } else if stick_x < -10 {
                p.facing = 0;
            }

            p.speed_ground_x = (stick_x * 2).clamp(-32767, 32767) as i16;
            p.state_age = p.state_age.saturating_add(1);
        }

        // Update frame counters
        session.frame = frame;

        // Update hidden state frame counter
        let hidden = &ctx.accounts.hidden_state;
        let mut h_data = hidden.try_borrow_mut_data()?;
        if h_data.len() >= HIDDEN_HEADER_SIZE {
            let frame_bytes = frame.to_le_bytes();
            h_data[9..13].copy_from_slice(&frame_bytes);
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Account Contexts
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Accounts)]
pub struct InitManifest<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + std::mem::size_of::<ModelManifestAccount>()
    )]
    pub manifest: Account<'info, ModelManifestAccount>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UploadWeights<'info> {
    #[account(mut)]
    pub weight_account: Account<'info, WeightAccount>,
    /// CHECK: Same underlying account as weight_account — raw data access for weight bytes.
    #[account(mut)]
    pub weight_data: AccountInfo<'info>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct FinalizeWeights<'info> {
    #[account(mut)]
    pub weight_account: Account<'info, WeightAccount>,
    /// CHECK: Same underlying account — raw data access for hash verification.
    pub weight_data: AccountInfo<'info>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct CreateSession<'info> {
    #[account(zero)]
    pub session: Account<'info, SessionStateAccount>,
    /// CHECK: Hidden state — too large for Borsh, accessed as raw data.
    #[account(mut)]
    pub hidden_state: AccountInfo<'info>,
    #[account(zero)]
    pub input_buffer: Account<'info, InputBufferAccount>,
    pub manifest: Account<'info, ModelManifestAccount>,
    #[account(mut)]
    pub player1: Signer<'info>,
}

#[derive(Accounts)]
pub struct JoinSession<'info> {
    #[account(mut)]
    pub session: Account<'info, SessionStateAccount>,
    pub player2: Signer<'info>,
}

#[derive(Accounts)]
pub struct CloseSession<'info> {
    #[account(mut)]
    pub session: Account<'info, SessionStateAccount>,
    pub player: Signer<'info>,
}

#[derive(Accounts)]
pub struct SubmitInput<'info> {
    pub session: Account<'info, SessionStateAccount>,
    #[account(mut)]
    pub input_buffer: Account<'info, InputBufferAccount>,
    pub player: Signer<'info>,
}

#[derive(Accounts)]
pub struct RunInference<'info> {
    #[account(mut)]
    pub session: Account<'info, SessionStateAccount>,
    /// CHECK: Hidden state — raw data access for Mamba2 recurrent state.
    #[account(mut)]
    pub hidden_state: AccountInfo<'info>,
    #[account(mut)]
    pub input_buffer: Account<'info, InputBufferAccount>,
    pub manifest: Account<'info, ModelManifestAccount>,
    /// CHECK: Weight data — read-only raw access for INT8 weights.
    pub weights: AccountInfo<'info>,
}
