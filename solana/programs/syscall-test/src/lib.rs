use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    program_error::ProgramError,
    pubkey::Pubkey,
};

extern "C" {
    fn sol_matmul_i8(
        weights: *const i8,
        input: *const i8,
        output: *mut i32,
        rows: u64,
        cols: u64,
    ) -> u64;
}

entrypoint!(process_instruction);

fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let accounts_iter = &mut accounts.iter();
    let output_account = next_account_info(accounts_iter)?;

    // Instruction data layout:
    //   [0..4]  rows (u32 LE)
    //   [4..8]  cols (u32 LE)
    //   [8 .. 8 + rows*cols]  weights (i8, row-major)
    //   [8 + rows*cols .. 8 + rows*cols + cols]  input (i8)

    if instruction_data.len() < 8 {
        return Err(ProgramError::InvalidInstructionData);
    }

    let rows = u32::from_le_bytes(
        instruction_data[0..4]
            .try_into()
            .map_err(|_| ProgramError::InvalidInstructionData)?,
    ) as usize;
    let cols = u32::from_le_bytes(
        instruction_data[4..8]
            .try_into()
            .map_err(|_| ProgramError::InvalidInstructionData)?,
    ) as usize;

    let weights_start = 8;
    let weights_end = weights_start + rows * cols;
    let input_start = weights_end;
    let input_end = input_start + cols;

    if instruction_data.len() < input_end {
        return Err(ProgramError::InvalidInstructionData);
    }

    let weights_ptr = instruction_data[weights_start..weights_end].as_ptr() as *const i8;
    let input_ptr = instruction_data[input_start..input_end].as_ptr() as *const i8;

    // Allocate output buffer
    let mut output_buf = vec![0i32; rows];

    let ret = unsafe {
        sol_matmul_i8(
            weights_ptr,
            input_ptr,
            output_buf.as_mut_ptr(),
            rows as u64,
            cols as u64,
        )
    };

    if ret != 0 {
        return Err(ProgramError::Custom(ret as u32));
    }

    // Write i32 results to the output account
    let mut data = output_account.try_borrow_mut_data()?;
    for (i, &val) in output_buf.iter().enumerate() {
        let offset = i * 4;
        if offset + 4 <= data.len() {
            data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    Ok(())
}
