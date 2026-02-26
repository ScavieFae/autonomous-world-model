/// Mollusk integration test — proves sol_matmul_i8 works end-to-end in the SVM.
///
/// Prerequisites: `cargo build-sbf --manifest-path programs/syscall-test/Cargo.toml`
/// (the compiled .so must exist at programs/syscall-test/target/deploy/syscall_test.so)
use awm_syscall::SyscallMatmulI8;
use mollusk_svm::{result::Check, Mollusk};
use solana_account::Account;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;

fn build_instruction_data(rows: u32, cols: u32, weights: &[i8], input: &[i8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&rows.to_le_bytes());
    data.extend_from_slice(&cols.to_le_bytes());
    data.extend(weights.iter().map(|&b| b as u8));
    data.extend(input.iter().map(|&b| b as u8));
    data
}

fn read_i32_output(data: &[u8], count: usize) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let offset = i * 4;
            i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
        })
        .collect()
}

fn setup_mollusk(program_id: &Pubkey) -> Mollusk {
    // syscall-test is excluded from the workspace, so its .so lives in its own target dir.
    // Use absolute path to avoid working-directory ambiguity.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let sbf_dir = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .join("programs/syscall-test/target/deploy");
    std::env::set_var("SBF_OUT_DIR", sbf_dir);

    let mut mollusk = Mollusk::default();

    // Register the syscall BEFORE loading the program
    mollusk
        .program_cache
        .program_runtime_environment
        .register_function("sol_matmul_i8", SyscallMatmulI8::vm)
        .unwrap();

    // Load the compiled BPF test program
    mollusk.add_program_with_loader(
        program_id,
        "syscall_test",
        &mollusk_svm::program::loader_keys::LOADER_V3,
    );

    mollusk
}

fn make_output_account(size: usize, owner: &Pubkey) -> Account {
    Account {
        lamports: 1_000_000,
        data: vec![0u8; size],
        owner: *owner,
        executable: false,
        rent_epoch: 0,
    }
}

#[test]
fn matmul_2x2_known_values() {
    let program_id = Pubkey::new_unique();
    let mollusk = setup_mollusk(&program_id);

    // [[1,2],[3,4]] x [5,6] = [17, 39]
    let ix_data = build_instruction_data(2, 2, &[1, 2, 3, 4], &[5, 6]);

    let output_key = Pubkey::new_unique();
    let output_account = make_output_account(8, &program_id);

    let ix = Instruction {
        program_id,
        accounts: vec![AccountMeta::new(output_key, false)],
        data: ix_data,
    };

    let result = mollusk.process_and_validate_instruction(
        &ix,
        &[(output_key, output_account)],
        &[Check::success()],
    );

    let output = read_i32_output(&result.resulting_accounts[0].1.data, 2);
    assert_eq!(output, vec![17, 39]);
}

#[test]
fn matmul_negative_values() {
    let program_id = Pubkey::new_unique();
    let mollusk = setup_mollusk(&program_id);

    // [[-1,2],[3,-4]] x [-5,6] = [5+12=17, -15-24=-39]
    let weights: Vec<i8> = vec![-1, 2, 3, -4];
    let input: Vec<i8> = vec![-5, 6];
    let ix_data = build_instruction_data(2, 2, &weights, &input);

    let output_key = Pubkey::new_unique();
    let output_account = make_output_account(8, &program_id);

    let ix = Instruction {
        program_id,
        accounts: vec![AccountMeta::new(output_key, false)],
        data: ix_data,
    };

    let result = mollusk.process_and_validate_instruction(
        &ix,
        &[(output_key, output_account)],
        &[Check::success()],
    );

    let output = read_i32_output(&result.resulting_accounts[0].1.data, 2);
    assert_eq!(output[0], 17);  // (-1)*(-5) + 2*6
    assert_eq!(output[1], -39); // 3*(-5) + (-4)*6
}

#[test]
fn matmul_larger_matrix() {
    let program_id = Pubkey::new_unique();
    let mollusk = setup_mollusk(&program_id);

    let rows: u32 = 4;
    let cols: u32 = 8;
    let weights: Vec<i8> = (0..(rows * cols) as usize)
        .map(|i| ((i * 3 + 7) % 256) as i8)
        .collect();
    let input: Vec<i8> = (0..cols as usize)
        .map(|i| ((i * 5 + 1) % 256) as i8)
        .collect();
    let ix_data = build_instruction_data(rows, cols, &weights, &input);

    let output_key = Pubkey::new_unique();
    let output_account = make_output_account((rows as usize) * 4, &program_id);

    let ix = Instruction {
        program_id,
        accounts: vec![AccountMeta::new(output_key, false)],
        data: ix_data,
    };

    let result = mollusk.process_and_validate_instruction(
        &ix,
        &[(output_key, output_account)],
        &[Check::success()],
    );

    let output = read_i32_output(&result.resulting_accounts[0].1.data, rows as usize);

    // Verify against naive computation
    for i in 0..rows as usize {
        let expected: i32 = (0..cols as usize)
            .map(|j| weights[i * cols as usize + j] as i32 * input[j] as i32)
            .sum();
        assert_eq!(output[i], expected, "row {} mismatch", i);
    }
}
