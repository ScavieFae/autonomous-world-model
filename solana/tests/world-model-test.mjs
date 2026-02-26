/**
 * World Model integration test — full lifecycle without Anchor IDL.
 *
 * Flow:
 *   1. init_manifest → create model configuration
 *   2. upload_weights → upload dummy weight data
 *   3. create_session → player 1 starts a session
 *   4. join_session → player 2 joins
 *   5. submit_input × 2 → both players submit controller inputs
 *   6. run_inference → advance one frame
 *   7. Verify state changes
 *
 * Uses the bench-direct.mjs pattern: manual instruction construction
 * with Anchor discriminators (sha256("global:<method>")[0..8]).
 */
import {
  Connection, Keypair, SystemProgram, Transaction,
  TransactionInstruction, PublicKey, ComputeBudgetProgram,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import { createHash } from "crypto";
import { readFileSync } from "fs";

// ── Config ───────────────────────────────────────────────────────────────

const PROGRAM_ID = new PublicKey("WrLd1111111111111111111111111111111111111111");
const RPC = "http://localhost:8899";
const conn = new Connection(RPC, "confirmed");

// Load local keypair (player 1)
const walletPath = `${process.env.HOME}/.config/solana/id.json`;
const player1 = Keypair.fromSecretKey(
  Uint8Array.from(JSON.parse(readFileSync(walletPath, "utf-8")))
);

// Player 2 is a fresh keypair
const player2 = Keypair.generate();

// ── Helpers ──────────────────────────────────────────────────────────────

function disc(methodName) {
  const hash = createHash("sha256").update(`global:${methodName}`).digest();
  return hash.subarray(0, 8);
}

function u8buf(n) { return Buffer.from([n]); }
function i8buf(n) { return Buffer.from([n & 0xFF]); }
function u16le(n) { const b = Buffer.alloc(2); b.writeUInt16LE(n); return b; }
function u32le(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; }
function u64le(n) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n));
  return b;
}
function i32le(n) { const b = Buffer.alloc(4); b.writeInt32LE(n); return b; }
function pubkeyBuf(pk) { return pk.toBuffer(); }

async function airdrop(pubkey, sol = 10) {
  const sig = await conn.requestAirdrop(pubkey, sol * 1e9);
  await conn.confirmTransaction(sig);
}

async function createProgramAccount(size, payer, signers) {
  const account = Keypair.generate();
  const rent = await conn.getMinimumBalanceForRentExemption(size);
  const tx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: payer.publicKey,
      newAccountPubkey: account.publicKey,
      space: size,
      lamports: rent,
      programId: PROGRAM_ID,
    })
  );
  await sendAndConfirmTransaction(conn, tx, [payer, account]);
  return account;
}

async function createSystemAccount(size, payer) {
  const account = Keypair.generate();
  const rent = await conn.getMinimumBalanceForRentExemption(size);
  const tx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: payer.publicKey,
      newAccountPubkey: account.publicKey,
      space: size,
      lamports: rent,
      programId: SystemProgram.programId,
    })
  );
  await sendAndConfirmTransaction(conn, tx, [payer, account]);
  return account;
}

async function sendIx(label, ix, signers, cuLimit = 400_000) {
  const tx = new Transaction();
  tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: cuLimit }));
  tx.add(ix);

  try {
    const sig = await sendAndConfirmTransaction(conn, tx, signers);
    await new Promise(r => setTimeout(r, 300));
    const info = await conn.getTransaction(sig, {
      commitment: "confirmed",
      maxSupportedTransactionVersion: 0,
    });
    const cu = info?.meta?.computeUnitsConsumed ?? 0;
    console.log(`  ✓ ${label}: ${cu.toLocaleString()} CU`);
    return { success: true, cu, sig };
  } catch (e) {
    const msg = e.message?.slice(0, 200) || String(e).slice(0, 200);
    console.log(`  ✗ ${label}: FAILED — ${msg}`);
    return { success: false, error: msg };
  }
}

// ── Account sizes (8-byte discriminator + struct fields) ─────────────────

// ModelManifestAccount size (approximate — Anchor adds 8-byte discriminator)
// Fields: 32 + 2 + 2*4 + 2*2 + 1 + 1 + 1 + 32*4 + 4*4 + 16*2 + 16*2 + 1024 + 1 + 2 + 1 + 2 + 32 + 1 + 4 + 4
// = ~1350 bytes. Round up generously.
const MANIFEST_SIZE = 1500;

// WeightAccount header: 8 + 1 + 4 + 32 + 1 + 32 + 4 = 82
const WEIGHT_HEADER = 82;

// SessionStateAccount: 8 + 1 + 4 + 4 + 32 + 32 + 1 + (2 * PlayerState) + 32 + 8 + 8 + 8
// PlayerState: 4 + 4 + 2 + 2 + 2*5 + 2 + 1 + 1 + 1 + 1 + 2 + 1 + 1 = 32 bytes
const SESSION_SIZE = 300;

// InputBufferAccount: 8 + 4 + 2*(8 bytes ControllerInput) + 1 + 1 = 30
const INPUT_BUFFER_SIZE = 40;

// Hidden state: header (16) + data (num_layers * d_inner * d_state)
// For test: 2 layers, d_inner=8, d_state=4 = 64 bytes of data
const HIDDEN_STATE_SIZE = 16 + 64;

// ── Test ─────────────────────────────────────────────────────────────────

async function main() {
  console.log("World Model Integration Test");
  console.log("Program:", PROGRAM_ID.toBase58());
  console.log("RPC:", RPC);
  console.log("");

  // Fund wallets
  console.log("── Setup ──");
  const bal = await conn.getBalance(player1.publicKey);
  if (bal < 5e9) {
    await airdrop(player1.publicKey, 10);
    console.log("  Airdropped 10 SOL to player1");
  } else {
    console.log(`  Player1: ${(bal / 1e9).toFixed(2)} SOL`);
  }
  await airdrop(player2.publicKey, 10);
  console.log("  Airdropped 10 SOL to player2");

  // ── 1. Init Manifest ──────────────────────────────────────────────────
  console.log("\n── 1. Init Manifest ──");

  const manifestKp = Keypair.generate();
  const manifestRent = await conn.getMinimumBalanceForRentExemption(MANIFEST_SIZE);
  const createManifestTx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: player1.publicKey,
      newAccountPubkey: manifestKp.publicKey,
      space: MANIFEST_SIZE,
      lamports: manifestRent,
      programId: PROGRAM_ID,
    })
  );
  await sendAndConfirmTransaction(conn, createManifestTx, [player1, manifestKp]);

  // Build init_manifest instruction data
  const modelName = Buffer.alloc(32);
  modelName.write("test-mamba2-v1");
  const lutData = Buffer.alloc(1024); // Dummy LUTs
  for (let i = 0; i < 1024; i++) lutData[i] = i & 0xFF;

  const initData = Buffer.concat([
    disc("init_manifest"),
    modelName,          // name: [u8; 32]
    u16le(1),           // version: u16
    u16le(64),          // d_model: u16
    u16le(128),         // d_inner: u16
    u16le(4),           // d_state: u16
    u8buf(2),           // num_layers: u8
    u8buf(4),           // num_heads: u8
    lutData,            // luts: [u8; 1024]
    u8buf(12),          // num_continuous: u8
    u16le(400),         // num_action_states: u16
    u8buf(2),           // num_binary: u8
    u16le(49),          // input_size: u16
    u32le(100000),      // total_params: u32
    u32le(100000),      // total_weight_bytes: u32
  ]);

  const initIx = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: manifestKp.publicKey, isSigner: false, isWritable: true },
      { pubkey: player1.publicKey, isSigner: true, isWritable: true },
      { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
    ],
    data: initData,
  });

  const r1 = await sendIx("init_manifest", initIx, [player1]);
  if (!r1.success) { console.log("FATAL: init_manifest failed"); return; }

  // ── 2. Upload Weights (create + upload chunk) ─────────────────────────
  console.log("\n── 2. Upload Weights ──");

  const weightDataSize = 256; // Small test weights
  const weightKp = Keypair.generate();
  const weightAccountSize = WEIGHT_HEADER + weightDataSize;
  const weightRent = await conn.getMinimumBalanceForRentExemption(weightAccountSize);

  // Create weight account (program-owned so we can write to it)
  const createWeightTx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: player1.publicKey,
      newAccountPubkey: weightKp.publicKey,
      space: weightAccountSize,
      lamports: weightRent,
      programId: PROGRAM_ID,
    })
  );
  await sendAndConfirmTransaction(conn, createWeightTx, [player1, weightKp]);

  // Note: We need to initialize the weight account header.
  // For the stub test, we'll skip actual weight upload and just test the session flow.
  console.log("  (Weight upload skipped — stub inference doesn't use weights)");

  // ── 3. Create Session ─────────────────────────────────────────────────
  console.log("\n── 3. Create Session ──");

  const sessionKp = Keypair.generate();
  const sessionRent = await conn.getMinimumBalanceForRentExemption(SESSION_SIZE);
  const createSessionAccTx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: player1.publicKey,
      newAccountPubkey: sessionKp.publicKey,
      space: SESSION_SIZE,
      lamports: sessionRent,
      programId: PROGRAM_ID,
    })
  );
  await sendAndConfirmTransaction(conn, createSessionAccTx, [player1, sessionKp]);

  const hiddenKp = Keypair.generate();
  const hiddenRent = await conn.getMinimumBalanceForRentExemption(HIDDEN_STATE_SIZE);
  const createHiddenTx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: player1.publicKey,
      newAccountPubkey: hiddenKp.publicKey,
      space: HIDDEN_STATE_SIZE,
      lamports: hiddenRent,
      programId: PROGRAM_ID,
    })
  );
  await sendAndConfirmTransaction(conn, createHiddenTx, [player1, hiddenKp]);

  const inputBufKp = Keypair.generate();
  const inputBufRent = await conn.getMinimumBalanceForRentExemption(INPUT_BUFFER_SIZE);
  const createInputBufTx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: player1.publicKey,
      newAccountPubkey: inputBufKp.publicKey,
      space: INPUT_BUFFER_SIZE,
      lamports: inputBufRent,
      programId: PROGRAM_ID,
    })
  );
  await sendAndConfirmTransaction(conn, createInputBufTx, [player1, inputBufKp]);

  const createSessionData = Buffer.concat([
    disc("create_session"),
    u8buf(2),            // stage: u8 (FD = 2)
    u8buf(0),            // character: u8 (Fox = 0)
    u32le(28800),        // max_frames: u32
    u64le(42),           // seed: u64
  ]);

  const createSessionIx = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: sessionKp.publicKey, isSigner: false, isWritable: true },
      { pubkey: hiddenKp.publicKey, isSigner: false, isWritable: true },
      { pubkey: inputBufKp.publicKey, isSigner: false, isWritable: true },
      { pubkey: manifestKp.publicKey, isSigner: false, isWritable: false },
      { pubkey: player1.publicKey, isSigner: true, isWritable: true },
    ],
    data: createSessionData,
  });

  const r3 = await sendIx("create_session", createSessionIx, [player1]);
  if (!r3.success) { console.log("FATAL: create_session failed"); return; }

  // ── 4. Join Session ───────────────────────────────────────────────────
  console.log("\n── 4. Join Session ──");

  const joinData = Buffer.concat([
    disc("join_session"),
    u8buf(9),            // character: u8 (Marth = 9)
  ]);

  const joinIx = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: sessionKp.publicKey, isSigner: false, isWritable: true },
      { pubkey: player2.publicKey, isSigner: true, isWritable: false },
    ],
    data: joinData,
  });

  const r4 = await sendIx("join_session", joinIx, [player1, player2]);
  if (!r4.success) { console.log("FATAL: join_session failed"); return; }

  // ── 5-6. Submit + Inference loop (3 frames) ───────────────────────────
  console.log("\n── 5-6. Input + Inference Loop (3 frames) ──");

  for (let frame = 0; frame < 3; frame++) {
    // Player 1 submits: move right
    const p1Input = Buffer.concat([
      disc("submit_input"),
      i8buf(40),    // stick_x
      i8buf(0),     // stick_y
      i8buf(0),     // c_stick_x
      i8buf(0),     // c_stick_y
      u8buf(0),     // trigger_l
      u8buf(0),     // trigger_r
      u8buf(0),     // buttons
      u8buf(0),     // buttons_ext
    ]);

    const p1Ix = new TransactionInstruction({
      programId: PROGRAM_ID,
      keys: [
        { pubkey: sessionKp.publicKey, isSigner: false, isWritable: false },
        { pubkey: inputBufKp.publicKey, isSigner: false, isWritable: true },
        { pubkey: player1.publicKey, isSigner: true, isWritable: false },
      ],
      data: p1Input,
    });

    await sendIx(`frame ${frame + 1}: p1 submit_input`, p1Ix, [player1]);

    // Player 2 submits: move left
    const p2Input = Buffer.concat([
      disc("submit_input"),
      i8buf(-40),   // stick_x (0xD8 = -40 as i8)
      i8buf(0),
      i8buf(0),
      i8buf(0),
      u8buf(0),
      u8buf(0),
      u8buf(0),
      u8buf(0),
    ]);

    const p2Ix = new TransactionInstruction({
      programId: PROGRAM_ID,
      keys: [
        { pubkey: sessionKp.publicKey, isSigner: false, isWritable: false },
        { pubkey: inputBufKp.publicKey, isSigner: false, isWritable: true },
        { pubkey: player2.publicKey, isSigner: true, isWritable: false },
      ],
      data: p2Input,
    });

    await sendIx(`frame ${frame + 1}: p2 submit_input`, p2Ix, [player1, player2]);

    // Run inference
    const inferData = disc("run_inference");

    const inferIx = new TransactionInstruction({
      programId: PROGRAM_ID,
      keys: [
        { pubkey: sessionKp.publicKey, isSigner: false, isWritable: true },
        { pubkey: hiddenKp.publicKey, isSigner: false, isWritable: true },
        { pubkey: inputBufKp.publicKey, isSigner: false, isWritable: true },
        { pubkey: manifestKp.publicKey, isSigner: false, isWritable: false },
        { pubkey: weightKp.publicKey, isSigner: false, isWritable: false },
      ],
      data: inferData,
    });

    await sendIx(`frame ${frame + 1}: run_inference`, inferIx, [player1]);
  }

  // ── 7. Verify state ───────────────────────────────────────────────────
  console.log("\n── 7. Verify Final State ──");

  const sessionData = await conn.getAccountInfo(sessionKp.publicKey);
  if (sessionData) {
    const data = sessionData.data;
    // Skip 8-byte discriminator
    const status = data[8];
    const frame = data.readUInt32LE(9);
    console.log(`  Status: ${status} (expected: ${STATUS_ACTIVE} = ACTIVE)`);
    console.log(`  Frame: ${frame} (expected: 3)`);

    // Player 1 x position (offset: 8 + 1 + 4 + 4 + 32 + 32 + 1 = 82, then i32)
    const p1_x = data.readInt32LE(82);
    // Player 2 starts after player 1 state
    // PlayerState is ~32 bytes via Borsh serialization
    console.log(`  Player 1 x: ${p1_x} (fixed-point, should be > initial -7680)`);

    if (frame === 3) {
      console.log("\n  ✓ All 3 frames processed successfully!");
    } else {
      console.log(`\n  ✗ Expected frame=3, got frame=${frame}`);
    }
  } else {
    console.log("  ✗ Could not read session account");
  }

  // ── 8. Close Session ──────────────────────────────────────────────────
  console.log("\n── 8. Close Session ──");

  const closeData = disc("close_session");
  const closeIx = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: sessionKp.publicKey, isSigner: false, isWritable: true },
      { pubkey: player1.publicKey, isSigner: true, isWritable: false },
    ],
    data: closeData,
  });

  await sendIx("close_session", closeIx, [player1]);

  // Verify ended
  const finalData = await conn.getAccountInfo(sessionKp.publicKey);
  if (finalData) {
    const status = finalData.data[8];
    console.log(`  Final status: ${status} (expected: ${STATUS_ENDED} = ENDED)`);
  }

  console.log("\n═══ Test Complete ═══");
}

const STATUS_ACTIVE = 2;
const STATUS_ENDED = 3;

main().catch(e => {
  console.error("Test failed:", e);
  process.exit(1);
});
