use anchor_lang::prelude::*;

#[error_code]
pub enum WorldModelError {
    // ── Lifecycle errors ─────────────────────────────────────────────────
    #[msg("Invalid state transition for current session status")]
    InvalidStateTransition,
    #[msg("Cannot join your own session")]
    CannotJoinOwnSession,

    // ── Input errors ─────────────────────────────────────────────────────
    #[msg("Session is not active")]
    SessionNotActive,
    #[msg("Player is not part of this session")]
    UnauthorizedPlayer,
    #[msg("Both players must submit inputs before inference")]
    InputsNotReady,

    // ── Weight upload errors ─────────────────────────────────────────────
    #[msg("Unauthorized — signer does not match authority")]
    Unauthorized,
    #[msg("Weight account is already finalized")]
    AlreadyFinalized,
    #[msg("Chunk would write past end of data region")]
    ChunkOutOfBounds,
    #[msg("Chunk exceeds maximum size")]
    ChunkTooLarge,
    #[msg("Not all bytes have been written")]
    IncompleteUpload,
    #[msg("SHA-256 hash does not match expected")]
    HashMismatch,

    // ── Inference errors ─────────────────────────────────────────────────
    #[msg("Account data too small for specified dimensions")]
    InsufficientData,
    #[msg("Model manifest is not ready (shards not finalized)")]
    ModelNotReady,
    #[msg("Hidden state dimensions do not match manifest")]
    HiddenStateMismatch,
}
