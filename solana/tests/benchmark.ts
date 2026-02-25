import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Keypair, SystemProgram, Transaction } from "@solana/web3.js";
import { expect } from "chai";

// CU Benchmark Tests
//
// These tests measure the compute unit cost of INT8 operations on Solana BPF.
// Results determine the inference pipeline strategy:
//   - Single-tx (~60M CU): needs ephemeral rollup with high CU ceiling
//   - Multi-tx (5-6M CU/tx): needs batched ordered transactions
//   - Layer-chunked (15-20M CU/tx): middle ground
//
// Run: anchor test --skip-lint -- --grep 'benchmark'
// Deploy to local: anchor build && anchor deploy
// Deploy to ephemeral: configure Anchor.toml cluster + deploy

describe("cu-benchmark", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.CuBenchmark as Program;

  // Helper: create an account filled with random INT8 data
  async function createDataAccount(size: number): Promise<Keypair> {
    const account = Keypair.generate();
    const data = Buffer.alloc(size);

    // Fill with pseudo-random INT8 values
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(Math.random() * 256);
    }

    const rentExempt =
      await provider.connection.getMinimumBalanceForRentExemption(size);

    const tx = new Transaction().add(
      SystemProgram.createAccount({
        fromPubkey: provider.wallet.publicKey,
        newAccountPubkey: account.publicKey,
        space: size,
        lamports: rentExempt,
        programId: program.programId,
      })
    );

    await provider.sendAndConfirm(tx, [account]);

    // Write data to the account
    // Note: In production, use a separate write instruction or upload program.
    // For benchmarking, the account just needs to exist with sufficient size.

    return account;
  }

  // Helper: create account with LUT data
  async function createLutAccount(
    numInputElements: number
  ): Promise<Keypair> {
    const size = 768 + numInputElements; // 3 LUTs (256 bytes each) + input data
    const account = Keypair.generate();
    const data = Buffer.alloc(size);

    // SiLU LUT (256 bytes): SiLU(x) = x * sigmoid(x), quantized to INT8
    for (let i = 0; i < 256; i++) {
      const x = (i - 128) / 16; // Map 0..255 to -8..7.9375
      const silu = x / (1 + Math.exp(-x));
      data[i] = Math.round(silu * 16 + 128); // Requantize to 0..255
    }

    // Softplus LUT (256 bytes): softplus(x) = ln(1 + exp(x))
    for (let i = 0; i < 256; i++) {
      const x = (i - 128) / 16;
      const softplus = Math.log(1 + Math.exp(x));
      data[256 + i] = Math.round(Math.min(softplus * 16 + 128, 255));
    }

    // rsqrt LUT (256 bytes): 1/sqrt(x) for positive values
    for (let i = 0; i < 256; i++) {
      const x = Math.max((i + 1) / 32, 0.001); // Avoid division by zero
      const rsqrt = 1 / Math.sqrt(x);
      data[512 + i] = Math.round(Math.min(rsqrt * 32, 255));
    }

    // Random input data
    for (let i = 0; i < numInputElements; i++) {
      data[768 + i] = Math.floor(Math.random() * 256);
    }

    const rentExempt =
      await provider.connection.getMinimumBalanceForRentExemption(size);

    const tx = new Transaction().add(
      SystemProgram.createAccount({
        fromPubkey: provider.wallet.publicKey,
        newAccountPubkey: account.publicKey,
        space: size,
        lamports: rentExempt,
        programId: program.programId,
      })
    );

    await provider.sendAndConfirm(tx, [account]);
    return account;
  }

  // Helper: create SSM step data account
  async function createSsmAccount(
    dInner: number,
    dState: number
  ): Promise<Keypair> {
    const hSize = dInner * dState;
    // Layout: softplus_lut(256) + exp_lut(256) + dt_raw(d_inner) + x(d_inner)
    //         + B(h_size) + C(h_size) + h(h_size) + A(d_inner)
    const size = 512 + dInner * 2 + hSize * 3 + dInner;
    const account = Keypair.generate();
    const data = Buffer.alloc(size);

    // Softplus LUT
    for (let i = 0; i < 256; i++) {
      const x = (i - 128) / 16;
      data[i] = Math.round(Math.min(Math.log(1 + Math.exp(x)) * 16 + 128, 255));
    }

    // Exp(-x) LUT
    for (let i = 0; i < 256; i++) {
      const x = i / 32; // 0..8 range
      data[256 + i] = Math.round(Math.exp(-x) * 127);
    }

    // Fill rest with random data
    for (let i = 512; i < size; i++) {
      data[i] = Math.floor(Math.random() * 256);
    }

    const rentExempt =
      await provider.connection.getMinimumBalanceForRentExemption(size);

    const tx = new Transaction().add(
      SystemProgram.createAccount({
        fromPubkey: provider.wallet.publicKey,
        newAccountPubkey: account.publicKey,
        space: size,
        lamports: rentExempt,
        programId: program.programId,
      })
    );

    await provider.sendAndConfirm(tx, [account]);
    return account;
  }

  // ── Matmul Benchmarks ─────────────────────────────────────────────────

  describe("benchmark: INT8 matmul", () => {
    const dimensions = [
      { rows: 64, cols: 64, label: "64x64 (warmup)" },
      { rows: 256, cols: 256, label: "256x256 (small)" },
      { rows: 512, cols: 512, label: "512x512 (d_model)" },
      { rows: 1024, cols: 512, label: "1024x512 (in_proj half)" },
      { rows: 2048, cols: 512, label: "2048x512 (in_proj full)" },
      { rows: 512, cols: 1024, label: "512x1024 (out_proj)" },
    ];

    for (const { rows, cols, label } of dimensions) {
      it(`matmul ${label}`, async () => {
        const size = rows * cols + cols + rows; // weights + input + output
        const account = await createDataAccount(size);

        try {
          const tx = await program.methods
            .benchMatmul(rows, cols)
            .accounts({ benchmark: account.publicKey })
            .rpc();

          const txInfo = await provider.connection.getTransaction(tx, {
            commitment: "confirmed",
          });

          const cuUsed = txInfo?.meta?.computeUnitsConsumed ?? 0;
          const macsPerCu = (rows * cols) / cuUsed;

          console.log(
            `  ${label}: ${cuUsed.toLocaleString()} CU ` +
              `(${(rows * cols).toLocaleString()} MACs, ` +
              `${macsPerCu.toFixed(2)} MACs/CU)`
          );

          expect(cuUsed).to.be.greaterThan(0);
        } catch (e: any) {
          // CU exceeded is expected for large dimensions on mainnet
          if (e.message?.includes("exceeded CU meter")) {
            console.log(`  ${label}: EXCEEDED CU LIMIT (expected on mainnet)`);
          } else {
            throw e;
          }
        }
      });
    }

    it(`matmul_tiled 512x512`, async () => {
      const rows = 512,
        cols = 512;
      const size = rows * cols + cols + rows;
      const account = await createDataAccount(size);

      try {
        const tx = await program.methods
          .benchMatmulTiled(rows, cols)
          .accounts({ benchmark: account.publicKey })
          .rpc();

        const txInfo = await provider.connection.getTransaction(tx, {
          commitment: "confirmed",
        });

        const cuUsed = txInfo?.meta?.computeUnitsConsumed ?? 0;
        console.log(
          `  tiled 512x512: ${cuUsed.toLocaleString()} CU ` +
            `(${(rows * cols).toLocaleString()} MACs)`
        );

        expect(cuUsed).to.be.greaterThan(0);
      } catch (e: any) {
        if (e.message?.includes("exceeded CU meter")) {
          console.log(`  tiled 512x512: EXCEEDED CU LIMIT`);
        } else {
          throw e;
        }
      }
    });
  });

  // ── LUT Benchmarks ────────────────────────────────────────────────────

  describe("benchmark: LUT activations", () => {
    const sizes = [256, 512, 1024];
    const activations = [
      { type: 0, name: "SiLU" },
      { type: 1, name: "softplus" },
      { type: 2, name: "rsqrt" },
    ];

    for (const { type: actType, name } of activations) {
      for (const numElements of sizes) {
        it(`${name} LUT: ${numElements} elements`, async () => {
          const account = await createLutAccount(numElements);

          try {
            const tx = await program.methods
              .benchLutActivation(numElements, actType)
              .accounts({ lut: account.publicKey })
              .rpc();

            const txInfo = await provider.connection.getTransaction(tx, {
              commitment: "confirmed",
            });

            const cuUsed = txInfo?.meta?.computeUnitsConsumed ?? 0;
            const cuPerLookup = cuUsed / numElements;

            console.log(
              `  ${name}(${numElements}): ${cuUsed.toLocaleString()} CU ` +
                `(${cuPerLookup.toFixed(1)} CU/lookup)`
            );

            expect(cuUsed).to.be.greaterThan(0);
          } catch (e: any) {
            if (e.message?.includes("exceeded CU meter")) {
              console.log(`  ${name}(${numElements}): EXCEEDED CU LIMIT`);
            } else {
              throw e;
            }
          }
        });
      }
    }
  });

  // ── SSM Step Benchmark ────────────────────────────────────────────────

  describe("benchmark: SSM selective scan step", () => {
    const configs = [
      { dInner: 256, dState: 16, label: "256x16 (small)" },
      { dInner: 512, dState: 16, label: "512x16 (half)" },
      { dInner: 1024, dState: 16, label: "1024x16 (full, target)" },
    ];

    for (const { dInner, dState, label } of configs) {
      it(`ssm_step ${label}`, async () => {
        const account = await createSsmAccount(dInner, dState);

        try {
          const tx = await program.methods
            .benchSsmStep(dInner, dState)
            .accounts({ ssmData: account.publicKey })
            .rpc();

          const txInfo = await provider.connection.getTransaction(tx, {
            commitment: "confirmed",
          });

          const cuUsed = txInfo?.meta?.computeUnitsConsumed ?? 0;
          const opsPerStep = dInner * dState * 3; // multiply-accumulate ops

          console.log(
            `  ssm ${label}: ${cuUsed.toLocaleString()} CU ` +
              `(${opsPerStep.toLocaleString()} ops)`
          );

          expect(cuUsed).to.be.greaterThan(0);
        } catch (e: any) {
          if (e.message?.includes("exceeded CU meter")) {
            console.log(`  ssm ${label}: EXCEEDED CU LIMIT`);
          } else {
            throw e;
          }
        }
      });
    }
  });

  // ── Full Layer Benchmark ──────────────────────────────────────────────

  describe("benchmark: full Mamba2 layer", () => {
    it("full layer d_model=512, d_inner=1024, d_state=16", async () => {
      const dModel = 512;
      const dInner = 1024;
      const dState = 16;

      // Weight account: needs enough for in_proj + out_proj
      // in_proj: 2048 * 512 = 1,048,576 bytes
      // out_proj: 512 * 1024 = 524,288 bytes
      // Total: ~1.5MB (may exceed account creation in single tx)
      const weightSize = Math.min(dInner * 2 * dModel, 1_000_000); // Cap for test
      const stateSize = dModel + dInner * dState; // input + hidden

      const weights = await createDataAccount(weightSize);
      const state = await createDataAccount(stateSize);

      try {
        const tx = await program.methods
          .benchFullLayer(dModel, dInner, dState)
          .accounts({
            weights: weights.publicKey,
            state: state.publicKey,
          })
          .rpc();

        const txInfo = await provider.connection.getTransaction(tx, {
          commitment: "confirmed",
        });

        const cuUsed = txInfo?.meta?.computeUnitsConsumed ?? 0;

        // Projected full model: 12 layers
        const projectedFullModel = cuUsed * 12;

        console.log(`  full layer: ${cuUsed.toLocaleString()} CU`);
        console.log(
          `  projected 12-layer model: ${projectedFullModel.toLocaleString()} CU`
        );
        console.log(
          `  single-tx feasibility: ${
            projectedFullModel < 60_000_000 ? "YES" : "NO"
          } (need <60M CU)`
        );
        console.log(
          `  txs needed at 1.4M CU: ${Math.ceil(projectedFullModel / 1_400_000)}`
        );
        console.log(
          `  txs needed at 5M CU: ${Math.ceil(projectedFullModel / 5_000_000)}`
        );

        expect(cuUsed).to.be.greaterThan(0);
      } catch (e: any) {
        if (e.message?.includes("exceeded CU meter")) {
          console.log(`  full layer: EXCEEDED CU LIMIT — need ephemeral rollup`);
        } else {
          throw e;
        }
      }
    });
  });

  // ── Summary ───────────────────────────────────────────────────────────

  describe("benchmark: summary & projections", () => {
    it("prints CU budget analysis", () => {
      console.log("\n═══ CU Budget Analysis ═══");
      console.log("Mamba2 architecture (estimated):");
      console.log("  d_model=512, d_inner=1024, d_state=16, num_layers=12");
      console.log("");
      console.log("Per-layer estimate from plan:");
      console.log("  in_proj:  ~3.1M CU (512×2048 matmul)");
      console.log("  ssm_step: ~147K CU (1024×16 scan)");
      console.log("  gate:     ~5K CU (1024 LUT + multiply)");
      console.log("  out_proj: ~1.6M CU (1024×512 matmul)");
      console.log("  total:    ~4.9M CU per layer");
      console.log("");
      console.log("Full model: ~59M CU (12 layers)");
      console.log("");
      console.log("Pipeline options:");
      console.log("  A) Single-tx:  1 tx × 60M CU (needs ER support)");
      console.log("  B) Per-layer:  12 tx × 5M CU (120ms if sequential)");
      console.log("  B') Chunked:   4 tx × 15M CU (40ms if sequential)");
      console.log("");
      console.log(
        "Run actual benchmarks above to get real CU numbers!"
      );
    });
  });
});
