import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Keypair, PublicKey, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";

// Session lifecycle + input + inference integration tests.
//
// Tests the full session flow: create → join → submit input → run inference → end
// Uses stub inference (Phase 3) — actual model inference comes in Phase 4.
//
// Run: anchor test --skip-lint -- --grep 'session'

describe("session", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  // Programs
  const sessionLifecycle = anchor.workspace.SessionLifecycle as Program;
  const submitInput = anchor.workspace.SubmitInput as Program;
  const runInference = anchor.workspace.RunInference as Program;

  // Session accounts (created per test)
  let sessionState: Keypair;
  let hiddenState: Keypair;
  let inputBuffer: Keypair;
  let frameLog: Keypair;

  // Players
  const player1 = Keypair.generate();
  const player2 = Keypair.generate();

  // Model reference (dummy for Phase 3)
  const modelManifest = Keypair.generate();

  // Constants
  const STAGE_FINAL_DESTINATION = 31;
  const CHAR_MARTH = 18;
  const CHAR_FOX = 1;
  const D_MODEL = 512;
  const D_INNER = 1024;
  const D_STATE = 16;
  const NUM_LAYERS = 12;

  // Status codes
  const STATUS_CREATED = 0;
  const STATUS_WAITING_PLAYERS = 1;
  const STATUS_ACTIVE = 2;
  const STATUS_ENDED = 3;

  beforeEach(async () => {
    // Allocate session accounts
    sessionState = Keypair.generate();
    hiddenState = Keypair.generate();
    inputBuffer = Keypair.generate();
    frameLog = Keypair.generate();

    // Airdrop to players
    const sig1 = await provider.connection.requestAirdrop(
      player1.publicKey,
      10 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(sig1);

    const sig2 = await provider.connection.requestAirdrop(
      player2.publicKey,
      10 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(sig2);
  });

  // ── Session Lifecycle Tests ─────────────────────────────────────────

  describe("session: lifecycle", () => {
    it("creates a session", async () => {
      // Create session: player 1 selects Marth on Final Destination
      await sessionLifecycle.methods
        .execute({
          action: 0, // CREATE
          player: player1.publicKey,
          character: CHAR_MARTH,
          stage: STAGE_FINAL_DESTINATION,
          model: modelManifest.publicKey,
          maxFrames: 28800, // 8 minutes at 60fps
          seed: new anchor.BN(42),
          dInner: D_INNER,
          dState: D_STATE,
          numLayers: NUM_LAYERS,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player1])
        .rpc();

      // Verify session state
      const session = await sessionLifecycle.account.sessionState.fetch(
        sessionState.publicKey
      );

      expect(session.status).to.equal(STATUS_WAITING_PLAYERS);
      expect(session.frame).to.equal(0);
      expect(session.maxFrames).to.equal(28800);
      expect(session.player1.toString()).to.equal(player1.publicKey.toString());
      expect(session.stage).to.equal(STAGE_FINAL_DESTINATION);
      expect(session.players[0].character).to.equal(CHAR_MARTH);
      expect(session.players[0].stocks).to.equal(4);

      console.log("  Session created: WaitingPlayers");
    });

    it("player 2 joins and session becomes active", async () => {
      // First create the session
      await sessionLifecycle.methods
        .execute({
          action: 0,
          player: player1.publicKey,
          character: CHAR_MARTH,
          stage: STAGE_FINAL_DESTINATION,
          model: modelManifest.publicKey,
          maxFrames: 28800,
          seed: new anchor.BN(42),
          dInner: D_INNER,
          dState: D_STATE,
          numLayers: NUM_LAYERS,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player1])
        .rpc();

      // Player 2 joins as Fox
      await sessionLifecycle.methods
        .execute({
          action: 1, // JOIN
          player: player2.publicKey,
          character: CHAR_FOX,
          stage: 0, // Ignored for JOIN
          model: PublicKey.default, // Ignored for JOIN
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: 0,
          dState: 0,
          numLayers: 0,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player2])
        .rpc();

      const session = await sessionLifecycle.account.sessionState.fetch(
        sessionState.publicKey
      );

      expect(session.status).to.equal(STATUS_ACTIVE);
      expect(session.player2.toString()).to.equal(player2.publicKey.toString());
      expect(session.players[1].character).to.equal(CHAR_FOX);
      expect(session.players[1].stocks).to.equal(4);

      // Verify initial positions (fixed-point: value * 256)
      expect(session.players[0].x).to.equal(-30 * 256); // P1 left
      expect(session.players[1].x).to.equal(30 * 256);  // P2 right

      console.log("  Player 2 joined: Session ACTIVE");
      console.log(`  P1: Marth at x=${session.players[0].x / 256}`);
      console.log(`  P2: Fox at x=${session.players[1].x / 256}`);
    });

    it("ends a session", async () => {
      // Create + join
      await sessionLifecycle.methods
        .execute({
          action: 0,
          player: player1.publicKey,
          character: CHAR_MARTH,
          stage: STAGE_FINAL_DESTINATION,
          model: modelManifest.publicKey,
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: D_INNER,
          dState: D_STATE,
          numLayers: NUM_LAYERS,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player1])
        .rpc();

      await sessionLifecycle.methods
        .execute({
          action: 1,
          player: player2.publicKey,
          character: CHAR_FOX,
          stage: 0,
          model: PublicKey.default,
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: 0,
          dState: 0,
          numLayers: 0,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player2])
        .rpc();

      // End session
      await sessionLifecycle.methods
        .execute({
          action: 2, // END
          player: player1.publicKey,
          character: 0,
          stage: 0,
          model: PublicKey.default,
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: 0,
          dState: 0,
          numLayers: 0,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player1])
        .rpc();

      const session = await sessionLifecycle.account.sessionState.fetch(
        sessionState.publicKey
      );

      expect(session.status).to.equal(STATUS_ENDED);
      console.log("  Session ended");
    });
  });

  // ── Input + Inference Tests ─────────────────────────────────────────

  describe("session: input + inference loop", () => {
    it("runs a 10-frame stub inference loop", async () => {
      // Setup: create + join session
      await sessionLifecycle.methods
        .execute({
          action: 0,
          player: player1.publicKey,
          character: CHAR_MARTH,
          stage: STAGE_FINAL_DESTINATION,
          model: modelManifest.publicKey,
          maxFrames: 0,
          seed: new anchor.BN(42),
          dInner: D_INNER,
          dState: D_STATE,
          numLayers: NUM_LAYERS,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player1])
        .rpc();

      await sessionLifecycle.methods
        .execute({
          action: 1,
          player: player2.publicKey,
          character: CHAR_FOX,
          stage: 0,
          model: PublicKey.default,
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: 0,
          dState: 0,
          numLayers: 0,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player2])
        .rpc();

      // Run 10 frames
      for (let frame = 0; frame < 10; frame++) {
        // P1: walk right (stick_x = 64)
        await submitInput.methods
          .execute({
            player: player1.publicKey,
            stickX: 64,
            stickY: 0,
            cStickX: 0,
            cStickY: 0,
            triggerL: 0,
            triggerR: 0,
            buttons: frame === 5 ? 1 : 0, // Jump on frame 5
            buttonsExt: 0,
          })
          .accounts({
            sessionState: sessionState.publicKey,
            inputBuffer: inputBuffer.publicKey,
          })
          .signers([player1])
          .rpc();

        // P2: walk left (stick_x = -64)
        await submitInput.methods
          .execute({
            player: player2.publicKey,
            stickX: -64,
            stickY: 0,
            cStickX: 0,
            cStickY: 0,
            triggerL: 0,
            triggerR: 0,
            buttons: 0,
            buttonsExt: 0,
          })
          .accounts({
            sessionState: sessionState.publicKey,
            inputBuffer: inputBuffer.publicKey,
          })
          .signers([player2])
          .rpc();

        // Run inference (stub)
        await runInference.methods
          .execute({})
          .accounts({
            sessionState: sessionState.publicKey,
            hiddenState: hiddenState.publicKey,
            inputBuffer: inputBuffer.publicKey,
            frameLog: frameLog.publicKey,
          })
          .rpc();

        // Read state
        const session = await runInference.account.sessionState.fetch(
          sessionState.publicKey
        );

        const p1x = session.players[0].x / 256;
        const p2x = session.players[1].x / 256;

        if (frame === 0 || frame === 5 || frame === 9) {
          console.log(
            `  Frame ${session.frame}: P1 x=${p1x.toFixed(1)}, P2 x=${p2x.toFixed(1)}, ` +
              `P1 on_ground=${session.players[0].onGround}, ` +
              `P1 jumps=${session.players[0].jumpsLeft}`
          );
        }
      }

      // Verify final state
      const finalSession = await runInference.account.sessionState.fetch(
        sessionState.publicKey
      );

      expect(finalSession.frame).to.equal(10);

      // P1 walked right: x should have increased from -30
      expect(finalSession.players[0].x).to.be.greaterThan(-30 * 256);

      // P2 walked left: x should have decreased from 30
      expect(finalSession.players[1].x).to.be.lessThan(30 * 256);

      // Frame log should have recorded frames
      const log = await runInference.account.frameLog.fetch(frameLog.publicKey);
      expect(log.totalFrames).to.equal(10);

      console.log(`  10-frame loop complete. Final frame: ${finalSession.frame}`);
    });
  });

  // ── State Format Verification ─────────────────────────────────────

  describe("session: output format matches visualizer", () => {
    it("produces state compatible with viz/visualizer-juicy.html JSON format", async () => {
      // Create a session and run a few frames to get state
      await sessionLifecycle.methods
        .execute({
          action: 0,
          player: player1.publicKey,
          character: CHAR_MARTH,
          stage: STAGE_FINAL_DESTINATION,
          model: modelManifest.publicKey,
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: D_INNER,
          dState: D_STATE,
          numLayers: NUM_LAYERS,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player1])
        .rpc();

      await sessionLifecycle.methods
        .execute({
          action: 1,
          player: player2.publicKey,
          character: CHAR_FOX,
          stage: 0,
          model: PublicKey.default,
          maxFrames: 0,
          seed: new anchor.BN(0),
          dInner: 0,
          dState: 0,
          numLayers: 0,
        })
        .accounts({
          sessionState: sessionState.publicKey,
          hiddenState: hiddenState.publicKey,
          inputBuffer: inputBuffer.publicKey,
          frameLog: frameLog.publicKey,
        })
        .signers([player2])
        .rpc();

      const session = await sessionLifecycle.account.sessionState.fetch(
        sessionState.publicKey
      );

      // Convert onchain state to visualizer JSON format
      const vizFrame = {
        players: session.players.map((p: any) => ({
          x: p.x / 256.0,
          y: p.y / 256.0,
          percent: p.percent,
          shield_strength: p.shieldStrength / 256.0,
          speed_air_x: p.speedAirX / 256.0,
          speed_y: p.speedY / 256.0,
          speed_ground_x: p.speedGroundX / 256.0,
          speed_attack_x: p.speedAttackX / 256.0,
          speed_attack_y: p.speedAttackY / 256.0,
          state_age: p.stateAge,
          hitlag: p.hitlag,
          stocks: p.stocks,
          facing: p.facing,
          on_ground: p.onGround,
          action_state: p.actionState,
          jumps_left: p.jumpsLeft,
          character: p.character,
        })),
        stage: session.stage,
      };

      // Verify the frame has all required fields
      for (const player of vizFrame.players) {
        expect(player).to.have.property("x").that.is.a("number");
        expect(player).to.have.property("y").that.is.a("number");
        expect(player).to.have.property("percent").that.is.a("number");
        expect(player).to.have.property("shield_strength").that.is.a("number");
        expect(player).to.have.property("speed_air_x").that.is.a("number");
        expect(player).to.have.property("speed_y").that.is.a("number");
        expect(player).to.have.property("speed_ground_x").that.is.a("number");
        expect(player).to.have.property("speed_attack_x").that.is.a("number");
        expect(player).to.have.property("speed_attack_y").that.is.a("number");
        expect(player).to.have.property("state_age").that.is.a("number");
        expect(player).to.have.property("hitlag").that.is.a("number");
        expect(player).to.have.property("stocks").that.is.a("number");
        expect(player).to.have.property("facing").that.is.a("number");
        expect(player).to.have.property("on_ground").that.is.a("number");
        expect(player).to.have.property("action_state").that.is.a("number");
        expect(player).to.have.property("jumps_left").that.is.a("number");
        expect(player).to.have.property("character").that.is.a("number");
      }
      expect(vizFrame).to.have.property("stage").that.is.a("number");

      console.log("  Visualizer-compatible frame:");
      console.log(JSON.stringify(vizFrame, null, 2));
    });
  });
});
