use anchor_lang::prelude::*;

declare_id!("UploadWt11111111111111111111111111111111111");

/// Maximum bytes per upload chunk.
/// Solana transaction size limit is ~1232 bytes, minus overhead.
/// Account data writes are separate from tx size, but we chunk for reliability.
pub const MAX_CHUNK_SIZE: usize = 1000;

/// Weight upload program — chunked writes to zero-copy weight shard accounts.
///
/// Uploading 15MB of INT8 weights to Solana requires chunked writes because:
///   1. Transaction data is limited (~1232 bytes)
///   2. Accounts must be created before writing (createAccountWithSeed or realloc)
///   3. Network reliability: smaller chunks = easier retry on failure
///
/// Upload flow:
///   1. CLI creates WeightShard accounts with correct size (via create_shard)
///   2. CLI sends weight data in chunks (via upload_chunk)
///   3. CLI finalizes each shard with SHA-256 verification (via finalize_shard)
///   4. CLI creates ModelManifest pointing to shard accounts (via create_manifest)
///
/// ~15MB at 1000 bytes/chunk = ~15,000 transactions.
/// At ~400 TPS on devnet, upload takes ~40 seconds.
#[program]
pub mod upload_weights {
    use super::*;

    /// Create a new weight shard account with the specified size.
    /// The account is allocated but filled with zeros until upload_chunk is called.
    pub fn create_shard(
        ctx: Context<CreateShard>,
        shard_index: u8,
        data_size: u32,
    ) -> Result<()> {
        let shard = &mut ctx.accounts.shard;
        shard.shard_index = shard_index;
        shard.data_size = data_size;
        shard.authority = ctx.accounts.authority.key();
        shard.finalized = false;
        shard.bytes_written = 0;
        shard.data_hash = [0u8; 32];

        msg!(
            "Shard {} created: {} bytes, authority={}",
            shard_index,
            data_size,
            ctx.accounts.authority.key()
        );
        Ok(())
    }

    /// Upload a chunk of weight data to a shard at the specified offset.
    ///
    /// Chunks can be uploaded in any order and are idempotent (re-uploading
    /// the same offset overwrites). This enables easy retry on network failure.
    pub fn upload_chunk(
        ctx: Context<UploadChunk>,
        offset: u32,
        data: Vec<u8>,
    ) -> Result<()> {
        let shard = &mut ctx.accounts.shard;

        // Only the authority can upload
        require!(
            ctx.accounts.authority.key() == shard.authority,
            UploadError::Unauthorized
        );

        // Cannot upload to finalized shard
        require!(!shard.finalized, UploadError::ShardFinalized);

        // Validate chunk bounds
        let offset = offset as usize;
        let end = offset + data.len();
        require!(
            end <= shard.data_size as usize,
            UploadError::ChunkOutOfBounds
        );

        require!(
            data.len() <= MAX_CHUNK_SIZE,
            UploadError::ChunkTooLarge
        );

        // Write chunk to account data (after the header)
        // In BOLT ECS, the component data is serialized first, then raw bytes follow.
        // For a standalone program, we write directly to the account data region.
        //
        // The actual write happens via the account's data field:
        let account_data = &mut ctx.accounts.shard_data.data.borrow_mut();
        let header_size = 8 + 1 + 4 + 32 + 1 + 32 + 4; // discriminator + fields
        let write_offset = header_size + offset;

        require!(
            write_offset + data.len() <= account_data.len(),
            UploadError::ChunkOutOfBounds
        );

        account_data[write_offset..write_offset + data.len()].copy_from_slice(&data);

        // Track progress
        let new_written = shard.bytes_written.max(end as u32);
        shard.bytes_written = new_written;

        Ok(())
    }

    /// Finalize a shard by verifying the SHA-256 hash of all uploaded data.
    ///
    /// After finalization, the shard is immutable and ready for inference.
    /// The hash is stored for verification by anyone.
    pub fn finalize_shard(
        ctx: Context<FinalizeShard>,
        expected_hash: [u8; 32],
    ) -> Result<()> {
        let shard = &mut ctx.accounts.shard;

        require!(
            ctx.accounts.authority.key() == shard.authority,
            UploadError::Unauthorized
        );

        require!(!shard.finalized, UploadError::ShardFinalized);

        // Verify all bytes have been written
        require!(
            shard.bytes_written >= shard.data_size,
            UploadError::IncompleteUpload
        );

        // Compute SHA-256 of the uploaded data
        // In production, use sol_sha256 syscall for efficiency
        let account_data = &ctx.accounts.shard_data.data.borrow();
        let header_size = 8 + 1 + 4 + 32 + 1 + 32 + 4;
        let data_region = &account_data[header_size..header_size + shard.data_size as usize];

        let computed_hash = anchor_lang::solana_program::hash::hash(data_region);

        require!(
            computed_hash.to_bytes() == expected_hash,
            UploadError::HashMismatch
        );

        shard.data_hash = expected_hash;
        shard.finalized = true;

        msg!(
            "Shard {} finalized: {} bytes, hash={}",
            shard.shard_index,
            shard.data_size,
            hex::encode(expected_hash)
        );
        Ok(())
    }
}

// ── Account structures ──────────────────────────────────────────────────────

#[account]
pub struct WeightShardAccount {
    pub shard_index: u8,
    pub data_size: u32,
    pub authority: Pubkey,
    pub finalized: bool,
    pub data_hash: [u8; 32],
    pub bytes_written: u32,
    // Followed by `data_size` bytes of raw weight data
}

#[derive(Accounts)]
#[instruction(shard_index: u8, data_size: u32)]
pub struct CreateShard<'info> {
    #[account(
        init,
        payer = authority,
        // Header (discriminator + fields) + data
        space = 8 + 1 + 4 + 32 + 1 + 32 + 4 + data_size as usize,
    )]
    pub shard: Account<'info, WeightShardAccount>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UploadChunk<'info> {
    #[account(mut)]
    pub shard: Account<'info, WeightShardAccount>,
    /// CHECK: Raw account data access for writing chunks past the header
    #[account(mut)]
    pub shard_data: AccountInfo<'info>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct FinalizeShard<'info> {
    #[account(mut)]
    pub shard: Account<'info, WeightShardAccount>,
    /// CHECK: Raw account data access for hash verification
    pub shard_data: AccountInfo<'info>,
    pub authority: Signer<'info>,
}

// ── Errors ──────────────────────────────────────────────────────────────────

#[error_code]
pub enum UploadError {
    #[msg("Only the shard authority can upload")]
    Unauthorized,
    #[msg("Shard is already finalized")]
    ShardFinalized,
    #[msg("Chunk extends past shard data boundary")]
    ChunkOutOfBounds,
    #[msg("Chunk exceeds maximum size")]
    ChunkTooLarge,
    #[msg("Not all bytes have been uploaded")]
    IncompleteUpload,
    #[msg("SHA-256 hash does not match expected value")]
    HashMismatch,
}
