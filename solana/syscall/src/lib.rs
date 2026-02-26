#![allow(deprecated)] // InvokeContext marked unstable-api in Agave 3.x, still functional

pub mod matmul;

use solana_program_runtime::{
    invoke_context::InvokeContext,
    solana_sbpf::{
        declare_builtin_function,
        memory_region::{AccessType, MemoryMapping},
    },
};

/// CU cost: base + 1 per MAC. Tunable by MagicBlock.
pub const CU_BASE: u64 = 100;
pub const CU_PER_MAC: u64 = 1;

/// Translate a BPF VM address to a host address via MemoryMapping.
/// Converts StableResult -> Result for use with `?`.
fn map_mem(
    mm: &MemoryMapping,
    access: AccessType,
    addr: u64,
    len: u64,
) -> Result<u64, Box<dyn std::error::Error>> {
    Result::from(mm.map(access, addr, len)).map_err(|e| format!("{e:?}").into())
}

declare_builtin_function!(
    /// Native INT8 matrix-vector multiply for MagicBlock ephemeral rollups.
    ///
    /// Register mapping (standard 5-register syscall convention):
    ///   r1 (weights_addr): VM pointer to row-major i8 weight matrix [rows * cols]
    ///   r2 (input_addr):   VM pointer to i8 input vector [cols]
    ///   r3 (output_addr):  VM pointer to caller-allocated i32 output buffer [rows]
    ///   r4 (rows):         Number of rows in weight matrix
    ///   r5 (cols):         Number of columns in weight matrix
    SyscallMatmulI8,
    fn rust(
        invoke_context: &mut InvokeContext,
        weights_addr: u64,
        input_addr: u64,
        output_addr: u64,
        rows: u64,
        cols: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let rows_usize = rows as usize;
        let cols_usize = cols as usize;

        // Charge CU proportional to work
        let macs = rows
            .checked_mul(cols)
            .ok_or("matmul dimensions overflow")?;
        let cu_cost = CU_BASE.saturating_add(macs.saturating_mul(CU_PER_MAC));
        invoke_context.consume_checked(cu_cost)?;

        // Translate BPF virtual addresses to host memory
        let weights_len = (rows_usize * cols_usize) as u64;
        let input_len = cols;
        let output_len = (rows_usize * 4) as u64; // i32 = 4 bytes

        let weights_host = map_mem(memory_mapping, AccessType::Load, weights_addr, weights_len)?;
        let input_host = map_mem(memory_mapping, AccessType::Load, input_addr, input_len)?;
        let output_host = map_mem(memory_mapping, AccessType::Store, output_addr, output_len)?;

        // SAFETY: memory_mapping.map() validated these regions are accessible
        // and within BPF memory bounds.
        let weights = unsafe {
            std::slice::from_raw_parts(weights_host as *const i8, rows_usize * cols_usize)
        };
        let input = unsafe {
            std::slice::from_raw_parts(input_host as *const i8, cols_usize)
        };
        let output = unsafe {
            std::slice::from_raw_parts_mut(output_host as *mut i32, rows_usize)
        };

        matmul::matmul_i8(weights, input, output, rows_usize, cols_usize);

        Ok(0)
    }
);
