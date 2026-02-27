import { Keypair, PublicKey } from "@solana/web3.js";
import {
  InitializeNewWorld,
  AddEntity,
  InitializeComponent,
  ApplySystem,
  anchor,
} from "@magicblock-labs/bolt-sdk";
import { expect } from "chai";

import { defaultInput } from "../client/src/input";
import {
  SESSION_LIFECYCLE_PROGRAM_ID,
  SUBMIT_INPUT_PROGRAM_ID,
  SESSION_STATE_PROGRAM_ID,
  HIDDEN_STATE_PROGRAM_ID,
  INPUT_BUFFER_PROGRAM_ID,
  FRAME_LOG_PROGRAM_ID,
  deserializeSessionState,
} from "../client/src/session";
import { SessionStatus } from "../client/src/state";

// ── Helpers ─────────────────────────────────────────────────────────────────

async function airdrop(
  provider: anchor.AnchorProvider,
  pubkey: PublicKey,
  sol = 2,
) {
  const sig = await provider.connection.requestAirdrop(
    pubkey,
    sol * anchor.web3.LAMPORTS_PER_SOL,
  );
  await provider.connection.confirmTransaction(sig, "confirmed");
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("BOLT ECS session lifecycle", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const STAGE_FD = 31;
  const CHAR_MARTH = 18;
  const CHAR_FOX = 1;
  const MODEL_MANIFEST = Keypair.generate().publicKey;

  let worldPda: PublicKey;
  let entityPda: PublicKey;
  let sessionStatePda: PublicKey;
  let hiddenStatePda: PublicKey;
  let inputBufferPda: PublicKey;
  let frameLogPda: PublicKey;

  const player1 = Keypair.generate();
  const player2 = Keypair.generate();

  before(async function () {
    await airdrop(provider, player1.publicKey, 10);
    await airdrop(provider, player2.publicKey, 10);
  });

  it("initializes a new World", async () => {
    const initWorld = await InitializeNewWorld({
      payer: player1.publicKey,
      connection: provider.connection,
    });
    const txSign = await provider.sendAndConfirm(initWorld.transaction, [player1]);
    worldPda = initWorld.worldPda;
    expect(worldPda).to.not.be.undefined;
    console.log(`World initialized: ${worldPda.toBase58()}`);
  });

  it("adds a session entity", async () => {
    const addEntity = await AddEntity({
      payer: player1.publicKey,
      world: worldPda,
      connection: provider.connection,
    });
    const txSign = await provider.sendAndConfirm(addEntity.transaction, [player1]);
    entityPda = addEntity.entityPda;
    expect(entityPda).to.not.be.undefined;
    console.log(`Entity added: ${entityPda.toBase58()}`);
  });

  it("initializes session_state component", async () => {
    const initComp = await InitializeComponent({
      payer: player1.publicKey,
      entity: entityPda,
      componentId: SESSION_STATE_PROGRAM_ID,
    });
    const txSign = await provider.sendAndConfirm(initComp.transaction, [player1]);
    sessionStatePda = initComp.componentPda;
    console.log(`SessionState component: ${sessionStatePda.toBase58()}`);
  });

  it("initializes hidden_state component", async () => {
    const initComp = await InitializeComponent({
      payer: player1.publicKey,
      entity: entityPda,
      componentId: HIDDEN_STATE_PROGRAM_ID,
    });
    const txSign = await provider.sendAndConfirm(initComp.transaction, [player1]);
    hiddenStatePda = initComp.componentPda;
    console.log(`HiddenState component: ${hiddenStatePda.toBase58()}`);
  });

  it("initializes input_buffer component", async () => {
    const initComp = await InitializeComponent({
      payer: player1.publicKey,
      entity: entityPda,
      componentId: INPUT_BUFFER_PROGRAM_ID,
    });
    const txSign = await provider.sendAndConfirm(initComp.transaction, [player1]);
    inputBufferPda = initComp.componentPda;
    console.log(`InputBuffer component: ${inputBufferPda.toBase58()}`);
  });

  it("initializes frame_log component", async () => {
    const initComp = await InitializeComponent({
      payer: player1.publicKey,
      entity: entityPda,
      componentId: FRAME_LOG_PROGRAM_ID,
    });
    const txSign = await provider.sendAndConfirm(initComp.transaction, [player1]);
    frameLogPda = initComp.componentPda;
    console.log(`FrameLog component: ${frameLogPda.toBase58()}`);
  });

  it("CREATE: session_lifecycle creates session", async () => {
    const result = await ApplySystem({
      authority: player1.publicKey,
      systemId: SESSION_LIFECYCLE_PROGRAM_ID,
      world: worldPda,
      entities: [{
        entity: entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: HIDDEN_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
          { componentId: FRAME_LOG_PROGRAM_ID },
        ],
      }],
      args: {
        action: 0,
        player: player1.publicKey.toBase58(),
        character: CHAR_MARTH,
        stage: STAGE_FD,
        model: MODEL_MANIFEST.toBase58(),
        max_frames: 600,
        seed: 42,
        d_inner: 768,
        d_state: 64,
        num_layers: 4,
      },
    });
    await provider.sendAndConfirm(result.transaction, [player1]);

    // Verify session state
    const account = await provider.connection.getAccountInfo(sessionStatePda, "confirmed");
    expect(account).to.not.be.null;
    const session = deserializeSessionState(account!.data as Buffer);

    expect(session.status).to.equal(SessionStatus.WaitingPlayers);
    expect(session.frame).to.equal(0);
    expect(session.maxFrames).to.equal(600);
    expect(session.player1).to.equal(player1.publicKey.toBase58());
    expect(session.player2).to.equal(PublicKey.default.toBase58());
    expect(session.stage).to.equal(STAGE_FD);
    expect(session.players[0].character).to.equal(CHAR_MARTH);
    expect(session.players[0].stocks).to.equal(4);
  });

  it("JOIN: player 2 joins session", async () => {
    const result = await ApplySystem({
      authority: player2.publicKey,
      systemId: SESSION_LIFECYCLE_PROGRAM_ID,
      world: worldPda,
      entities: [{
        entity: entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: HIDDEN_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
          { componentId: FRAME_LOG_PROGRAM_ID },
        ],
      }],
      args: {
        action: 1,
        player: player2.publicKey.toBase58(),
        character: CHAR_FOX,
        stage: 0,
        model: PublicKey.default.toBase58(),
        max_frames: 0,
        seed: 0,
        d_inner: 0,
        d_state: 0,
        num_layers: 0,
      },
    });
    await provider.sendAndConfirm(result.transaction, [player2]);

    const account = await provider.connection.getAccountInfo(sessionStatePda, "confirmed");
    const session = deserializeSessionState(account!.data as Buffer);

    expect(session.status).to.equal(SessionStatus.Active);
    expect(session.player2).to.equal(player2.publicKey.toBase58());

    // Starting positions (fixed-point ×256)
    expect(session.players[0].x).to.equal(-30 * 256);
    expect(session.players[1].x).to.equal(30 * 256);
    expect(session.players[0].y).to.equal(0);
    expect(session.players[1].y).to.equal(0);
    expect(session.players[0].facing).to.equal(1);
    expect(session.players[1].facing).to.equal(0);
    expect(session.players[0].onGround).to.equal(1);
    expect(session.players[1].onGround).to.equal(1);
    expect(session.players[0].jumpsLeft).to.equal(2);
    expect(session.players[1].jumpsLeft).to.equal(2);
    expect(session.players[0].shieldStrength).to.equal(60 * 256);
    expect(session.players[1].shieldStrength).to.equal(60 * 256);
    expect(session.players[0].character).to.equal(CHAR_MARTH);
    expect(session.players[1].character).to.equal(CHAR_FOX);
    expect(session.players[0].stocks).to.equal(4);
    expect(session.players[1].stocks).to.equal(4);
  });

  it("SUBMIT_INPUT: both players submit inputs", async () => {
    // Player 1 input
    const p1Result = await ApplySystem({
      authority: player1.publicKey,
      systemId: SUBMIT_INPUT_PROGRAM_ID,
      world: worldPda,
      entities: [{
        entity: entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
        ],
      }],
      args: {
        player: player1.publicKey.toBase58(),
        stick_x: 64,
        stick_y: 0,
        c_stick_x: 0,
        c_stick_y: 0,
        trigger_l: 0,
        trigger_r: 0,
        buttons: 1, // A button
        buttons_ext: 0,
      },
    });
    await provider.sendAndConfirm(p1Result.transaction, [player1]);

    // Player 2 input
    const p2Result = await ApplySystem({
      authority: player2.publicKey,
      systemId: SUBMIT_INPUT_PROGRAM_ID,
      world: worldPda,
      entities: [{
        entity: entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
        ],
      }],
      args: {
        player: player2.publicKey.toBase58(),
        stick_x: -32,
        stick_y: 0,
        c_stick_x: 0,
        c_stick_y: 0,
        trigger_l: 255,
        trigger_r: 0,
        buttons: 0,
        buttons_ext: 4, // digital L
      },
    });
    await provider.sendAndConfirm(p2Result.transaction, [player2]);

    // Verify session frame didn't advance (submit_input doesn't advance frame)
    const account = await provider.connection.getAccountInfo(sessionStatePda, "confirmed");
    const session = deserializeSessionState(account!.data as Buffer);
    expect(session.status).to.equal(SessionStatus.Active);
    expect(session.frame).to.equal(0);
  });

  it("END: session lifecycle ends session", async () => {
    const result = await ApplySystem({
      authority: player1.publicKey,
      systemId: SESSION_LIFECYCLE_PROGRAM_ID,
      world: worldPda,
      entities: [{
        entity: entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: HIDDEN_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
          { componentId: FRAME_LOG_PROGRAM_ID },
        ],
      }],
      args: {
        action: 2,
        player: player1.publicKey.toBase58(),
        character: 0,
        stage: 0,
        model: PublicKey.default.toBase58(),
        max_frames: 0,
        seed: 0,
        d_inner: 0,
        d_state: 0,
        num_layers: 0,
      },
    });
    await provider.sendAndConfirm(result.transaction, [player1]);

    const account = await provider.connection.getAccountInfo(sessionStatePda, "confirmed");
    const session = deserializeSessionState(account!.data as Buffer);
    expect(session.status).to.equal(SessionStatus.Ended);
  });
});
