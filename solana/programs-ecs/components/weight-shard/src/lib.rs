use bolt_lang::*;

declare_id!("A56nQANMn1ThuqZLZkAVooDmUMrSoEddyNHF41WbqvXE");

/// INT8 weight shard — stores quantized model weights for onchain inference.
///
/// Architecture: Two shards hold the complete INT8 Mamba2 model (~15MB total).
/// Each shard is a zero-copy account accessed directly by the inference system.
///
/// Lifecycle: Permanent on mainnet, delegated to ephemeral rollup for sessions.
/// Forkable: anyone can read weight accounts and deploy alternate worlds.
///
/// Layout: Raw INT8 bytes, indexed by offsets from ModelManifest.
/// The data field is sized at creation and populated via the upload-weights program.
#[component]
#[derive(Default)]
pub struct WeightShard {
    /// Shard index (0 or 1 for 2-shard model)
    pub shard_index: u8,

    /// Total size of weight data in bytes
    pub data_size: u32,

    /// Authority that can write to this shard (upload program PDA)
    pub authority: Pubkey,

    /// Whether the shard is fully uploaded and verified
    pub finalized: bool,

    /// SHA-256 hash of the weight data (verified on finalization)
    pub data_hash: [u8; 32],

    // NOTE: The actual weight data is stored in the account's remaining data
    // space, accessed via zero-copy (account_info.data). The fields above are
    // the header; weight bytes follow immediately after the component header.
}
