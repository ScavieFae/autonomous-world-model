/**
 * Direct CU benchmark — no Anchor IDL needed.
 * Constructs instructions manually using Anchor's discriminator convention:
 *   sha256("global:<snake_case_method>")[0..8]
 */
import {
  Connection, Keypair, SystemProgram, Transaction,
  TransactionInstruction, PublicKey, ComputeBudgetProgram,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import { createHash } from "crypto";
import { readFileSync } from "fs";

const PROGRAM_ID = new PublicKey("2ugkUeQwNdfFpQXKHja4LiFxFgvn1VNn7w1YLp6XeNEJ");
const RPC = "http://localhost:8899";

const conn = new Connection(RPC, "confirmed");

// Load local keypair
const walletPath = `${process.env.HOME}/.config/solana/id.json`;
const wallet = Keypair.fromSecretKey(
  Uint8Array.from(JSON.parse(readFileSync(walletPath, "utf-8")))
);

// Anchor discriminator: first 8 bytes of sha256("global:<method_name>")
function disc(methodName) {
  const hash = createHash("sha256").update(`global:${methodName}`).digest();
  return hash.subarray(0, 8);
}

// Encode u32 as little-endian 4 bytes
function u32le(n) {
  const buf = Buffer.alloc(4);
  buf.writeUInt32LE(n);
  return buf;
}

// Encode u8
function u8(n) {
  return Buffer.from([n]);
}

// Create an account owned by the SYSTEM program (we just need data, not program-owned)
// Actually for AccountInfo reads, the account can be any owner — Anchor's
// unchecked accounts don't validate ownership.
async function createDataAccount(size) {
  const account = Keypair.generate();
  const rent = await conn.getMinimumBalanceForRentExemption(size);
  const tx = new Transaction().add(
    SystemProgram.createAccount({
      fromPubkey: wallet.publicKey,
      newAccountPubkey: account.publicKey,
      space: size,
      lamports: rent,
      programId: SystemProgram.programId, // system-owned is fine for CHECK accounts
    })
  );
  await sendAndConfirmTransaction(conn, tx, [wallet, account]);
  return account;
}

// Run an instruction and return CU consumed
async function runAndMeasure(label, ix, cuLimit = 1_400_000) {
  const tx = new Transaction();
  // Request higher CU limit
  tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: cuLimit }));
  tx.add(ix);

  try {
    const sig = await sendAndConfirmTransaction(conn, tx, [wallet]);
    // Small delay for finalization
    await new Promise(r => setTimeout(r, 500));
    const info = await conn.getTransaction(sig, {
      commitment: "confirmed",
      maxSupportedTransactionVersion: 0,
    });
    const cu = info?.meta?.computeUnitsConsumed ?? 0;
    return { label, cu, success: true };
  } catch (e) {
    const msg = e.message || String(e);
    if (msg.includes("exceeded") || msg.includes("computational budget")) {
      return { label, cu: cuLimit, success: false, exceeded: true };
    }
    return { label, cu: 0, success: false, error: msg.slice(0, 120) };
  }
}

// ── Benchmarks ──────────────────────────────────────────────────────────

async function benchMatmul(rows, cols, tiled = false) {
  const size = rows * cols + cols + rows;
  const account = await createDataAccount(size);

  const method = tiled ? "bench_matmul_tiled" : "bench_matmul";
  const data = Buffer.concat([disc(method), u32le(rows), u32le(cols)]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [{ pubkey: account.publicKey, isSigner: false, isWritable: false }],
    data,
  });

  const label = `${tiled ? "matmul_tiled" : "matmul"} ${rows}x${cols}`;
  return runAndMeasure(label, ix);
}

async function benchMatmulVariant(rows, cols, method) {
  const size = rows * cols + cols + rows;
  const account = await createDataAccount(size);

  const data = Buffer.concat([disc(method), u32le(rows), u32le(cols)]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [{ pubkey: account.publicKey, isSigner: false, isWritable: false }],
    data,
  });

  const label = `${method.replace("bench_", "")} ${rows}x${cols}`;
  return runAndMeasure(label, ix);
}

async function benchLut(numElements, activationType, activationName) {
  const size = 768 + numElements;
  const account = await createDataAccount(size);

  const data = Buffer.concat([
    disc("bench_lut_activation"),
    u32le(numElements),
    u8(activationType),
  ]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [{ pubkey: account.publicKey, isSigner: false, isWritable: false }],
    data,
  });

  return runAndMeasure(`lut_${activationName} ${numElements}`, ix);
}

async function benchSsm(dInner, dState) {
  const hSize = dInner * dState;
  const size = 512 + dInner * 2 + hSize * 3 + dInner;
  const account = await createDataAccount(size);

  const data = Buffer.concat([
    disc("bench_ssm_step"),
    u32le(dInner),
    u32le(dState),
  ]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [{ pubkey: account.publicKey, isSigner: false, isWritable: false }],
    data,
  });

  return runAndMeasure(`ssm_step ${dInner}x${dState}`, ix);
}

async function benchFullLayer(dModel, dInner, dState) {
  const weightSize = Math.min(dInner * 2 * dModel, 1_000_000);
  const stateSize = dModel + dInner * dState;
  const weights = await createDataAccount(weightSize);
  const state = await createDataAccount(stateSize);

  const data = Buffer.concat([
    disc("bench_full_layer"),
    u32le(dModel),
    u32le(dInner),
    u32le(dState),
  ]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: weights.publicKey, isSigner: false, isWritable: false },
      { pubkey: state.publicKey, isSigner: false, isWritable: false },
    ],
    data,
  });

  return runAndMeasure(`full_layer ${dModel}/${dInner}/${dState}`, ix);
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  console.log("CU Benchmark: INT8 Matmul + LUT Activations + SSM + Full Layer");
  console.log("Program:", PROGRAM_ID.toBase58());
  console.log("RPC:", RPC);

  const bal = await conn.getBalance(wallet.publicKey);
  console.log("Wallet:", wallet.publicKey.toBase58(), `(${bal / 1e9} SOL)\n`);

  const results = [];

  // Matmul
  console.log("── INT8 Matmul ──");
  for (const [r, c] of [[64, 64], [128, 128], [256, 256], [512, 512]]) {
    const res = await benchMatmul(r, c);
    results.push(res);
    const macs = r * c;
    const ratio = res.cu > 0 ? (macs / res.cu).toFixed(2) : "N/A";
    console.log(`  ${res.label}: ${res.cu.toLocaleString()} CU (${macs.toLocaleString()} MACs, ${ratio} MACs/CU)${res.exceeded ? " [EXCEEDED]" : ""}${res.error ? ` [ERROR: ${res.error}]` : ""}`);
  }

  // Tiled comparison
  console.log("\n── INT8 Matmul Tiled (4x unroll) ──");
  for (const [r, c] of [[128, 128], [256, 256]]) {
    const res = await benchMatmul(r, c, true);
    results.push(res);
    console.log(`  ${res.label}: ${res.cu.toLocaleString()} CU${res.exceeded ? " [EXCEEDED]" : ""}${res.error ? ` [ERROR: ${res.error}]` : ""}`);
  }

  // Unsafe (no bounds checks)
  console.log("\n── INT8 Matmul Unsafe (no bounds checks) ──");
  for (const [r, c] of [[64, 64], [128, 128], [256, 256], [512, 512]]) {
    const res = await benchMatmulVariant(r, c, "bench_matmul_unsafe");
    results.push(res);
    const macs = r * c;
    const ratio = res.cu > 0 && res.success ? (macs / res.cu).toFixed(2) : "N/A";
    console.log(`  ${res.label}: ${res.cu.toLocaleString()} CU (${macs.toLocaleString()} MACs, ${ratio} MACs/CU)${res.exceeded ? " [EXCEEDED]" : ""}${res.error ? ` [ERROR: ${res.error}]` : ""}`);
  }

  // Packed (u32 loads, unsafe)
  console.log("\n── INT8 Matmul Packed (u32 loads + unsafe) ──");
  for (const [r, c] of [[64, 64], [128, 128], [256, 256], [512, 512]]) {
    const res = await benchMatmulVariant(r, c, "bench_matmul_packed");
    results.push(res);
    const macs = r * c;
    const ratio = res.cu > 0 && res.success ? (macs / res.cu).toFixed(2) : "N/A";
    console.log(`  ${res.label}: ${res.cu.toLocaleString()} CU (${macs.toLocaleString()} MACs, ${ratio} MACs/CU)${res.exceeded ? " [EXCEEDED]" : ""}${res.error ? ` [ERROR: ${res.error}]` : ""}`);
  }

  // LUT activations
  console.log("\n── LUT Activations ──");
  for (const [type, name] of [[0, "SiLU"], [1, "softplus"], [2, "rsqrt"]]) {
    for (const n of [256, 512, 1024]) {
      const res = await benchLut(n, type, name);
      results.push(res);
      const perLookup = res.cu > 0 ? (res.cu / n).toFixed(1) : "N/A";
      console.log(`  ${res.label}: ${res.cu.toLocaleString()} CU (${perLookup} CU/lookup)${res.error ? ` [ERROR: ${res.error}]` : ""}`);
    }
  }

  // SSM step
  console.log("\n── SSM Selective Scan Step ──");
  for (const [di, ds] of [[256, 16], [512, 16], [1024, 16]]) {
    const res = await benchSsm(di, ds);
    results.push(res);
    console.log(`  ${res.label}: ${res.cu.toLocaleString()} CU${res.exceeded ? " [EXCEEDED]" : ""}${res.error ? ` [ERROR: ${res.error}]` : ""}`);
  }

  // Full layer
  console.log("\n── Full Mamba2 Layer ──");
  const fullRes = await benchFullLayer(512, 1024, 16);
  results.push(fullRes);
  console.log(`  ${fullRes.label}: ${fullRes.cu.toLocaleString()} CU${fullRes.exceeded ? " [EXCEEDED]" : ""}${fullRes.error ? ` [ERROR: ${fullRes.error}]` : ""}`);

  // Projections
  console.log("\n═══ Projections ═══");
  if (fullRes.cu > 0 && fullRes.success) {
    const perLayer = fullRes.cu;
    const fullModel = perLayer * 12;
    console.log(`  Per layer:          ${perLayer.toLocaleString()} CU`);
    console.log(`  12-layer model:     ${fullModel.toLocaleString()} CU`);
    console.log(`  Single-tx feasible: ${fullModel < 60_000_000 ? "YES (<60M)" : "NO (>60M)"}`);
    console.log(`  Txs at 1.4M CU:    ${Math.ceil(fullModel / 1_400_000)}`);
    console.log(`  Txs at 5M CU:      ${Math.ceil(fullModel / 5_000_000)}`);
    console.log(`  Txs at 10M CU:     ${Math.ceil(fullModel / 10_000_000)}`);
  } else {
    console.log("  Full layer exceeded or failed — projections unavailable");
    console.log("  Implies multi-tx pipeline is mandatory");
  }
}

main().catch(console.error);
