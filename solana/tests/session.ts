import * as anchor from "@coral-xyz/anchor";
import { expect } from "chai";
import { Keypair, PublicKey } from "@solana/web3.js";

import { defaultInput } from "../client/src/input";
import {
  SESSION_LIFECYCLE_PROGRAM_ID,
  SUBMIT_INPUT_PROGRAM_ID,
  SessionClient,
  type SessionConfig,
} from "../client/src/session";
import { SessionStatus } from "../client/src/state";

type DecodedControllerInput = {
  stickX: number;
  stickY: number;
  cStickX: number;
  cStickY: number;
  triggerL: number;
  triggerR: number;
  buttons: number;
  buttonsExt: number;
};

type DecodedInputBuffer = {
  frame: number;
  player1: DecodedControllerInput;
  player2: DecodedControllerInput;
  p1Ready: boolean;
  p2Ready: boolean;
};

function decodeControllerInput(buf: Buffer, offset: number): DecodedControllerInput {
  return {
    stickX: buf.readInt8(offset),
    stickY: buf.readInt8(offset + 1),
    cStickX: buf.readInt8(offset + 2),
    cStickY: buf.readInt8(offset + 3),
    triggerL: buf.readUInt8(offset + 4),
    triggerR: buf.readUInt8(offset + 5),
    buttons: buf.readUInt8(offset + 6),
    buttonsExt: buf.readUInt8(offset + 7),
  };
}

function decodeInputBufferAccount(data: Buffer): DecodedInputBuffer {
  let offset = 8; // Anchor discriminator
  const frame = data.readUInt32LE(offset);
  offset += 4;

  const player1 = decodeControllerInput(data, offset);
  offset += 8;
  const player2 = decodeControllerInput(data, offset);
  offset += 8;

  const p1Ready = data.readUInt8(offset) !== 0;
  offset += 1;
  const p2Ready = data.readUInt8(offset) !== 0;

  return { frame, player1, player2, p1Ready, p2Ready };
}

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

async function assertProgramIsExecutable(
  provider: anchor.AnchorProvider,
  programId: PublicKey,
) {
  const info = await provider.connection.getAccountInfo(programId, "confirmed");
  expect(info, `Program ${programId.toBase58()} is not deployed on localnet`).to.not.equal(null);
  expect(info?.executable, `${programId.toBase58()} is not executable`).to.equal(true);
}

describe("session sdk wire format audit", () => {
  const dummyPlayer = Keypair.generate();
  const dummyModel = Keypair.generate().publicKey;
  const client = new SessionClient(dummyPlayer, {
    cluster: "http://127.0.0.1:8899",
    modelManifest: dummyModel,
    stage: 31,
    character: 18,
  });

  it("encodes session_lifecycle and submit_input args with Rust field order/size", () => {
    const anyClient = client as any;
    const player = Keypair.generate().publicKey;
    const model = Keypair.generate().publicKey;

    const lifecycle: Buffer = anyClient.encodeLifecycleArgs({
      action: 1,
      player,
      character: 18,
      stage: 31,
      model,
      maxFrames: 600,
      seed: BigInt(42),
      dInner: 768,
      dState: 64,
      numLayers: 4,
    });

    expect(lifecycle.length).to.equal(92); // 8 + 84-byte Args
    expect(lifecycle.readUInt8(8)).to.equal(1); // action
    expect(new PublicKey(lifecycle.subarray(9, 41)).toBase58()).to.equal(player.toBase58());
    expect(lifecycle.readUInt8(41)).to.equal(18); // character
    expect(lifecycle.readUInt8(42)).to.equal(31); // stage
    expect(new PublicKey(lifecycle.subarray(43, 75)).toBase58()).to.equal(model.toBase58());
    expect(lifecycle.readUInt32LE(75)).to.equal(600);
    expect(lifecycle.readBigUInt64LE(79)).to.equal(BigInt(42));
    expect(lifecycle.readUInt16LE(87)).to.equal(768);
    expect(lifecycle.readUInt16LE(89)).to.equal(64);
    expect(lifecycle.readUInt8(91)).to.equal(4);

    const input: Buffer = anyClient.encodeInputArgs({
      player,
      stickX: -10,
      stickY: 11,
      cStickX: -12,
      cStickY: 13,
      triggerL: 14,
      triggerR: 15,
      buttons: 0x16,
      buttonsExt: 0x17,
    });

    expect(input.length).to.equal(48); // 8 + 40-byte Args
    expect(new PublicKey(input.subarray(8, 40)).toBase58()).to.equal(player.toBase58());
    expect(input.readInt8(40)).to.equal(-10);
    expect(input.readInt8(41)).to.equal(11);
    expect(input.readInt8(42)).to.equal(-12);
    expect(input.readInt8(43)).to.equal(13);
    expect(input.readUInt8(44)).to.equal(14);
    expect(input.readUInt8(45)).to.equal(15);
    expect(input.readUInt8(46)).to.equal(0x16);
    expect(input.readUInt8(47)).to.equal(0x17);
  });

  it("deserializes PlayerState at the correct 32-byte offsets", () => {
    const anyClient = client as any;
    const buf = Buffer.alloc(32);
    buf.writeInt32LE(-7680, 0);
    buf.writeInt32LE(512, 4);
    buf.writeUInt16LE(42, 8);
    buf.writeUInt16LE(60 * 256, 10);
    buf.writeInt16LE(-128, 12);
    buf.writeInt16LE(64, 14);
    buf.writeInt16LE(32, 16);
    buf.writeInt16LE(-16, 18);
    buf.writeInt16LE(8, 20);
    buf.writeUInt16LE(12, 22);
    buf.writeUInt8(3, 24);
    buf.writeUInt8(4, 25);
    buf.writeUInt8(1, 26);
    buf.writeUInt8(1, 27);
    buf.writeUInt16LE(44, 28);
    buf.writeUInt8(2, 30);
    buf.writeUInt8(18, 31);

    const p = anyClient.deserializePlayerState(buf, 0);
    expect(p.x).to.equal(-7680);
    expect(p.y).to.equal(512);
    expect(p.percent).to.equal(42);
    expect(p.shieldStrength).to.equal(60 * 256);
    expect(p.speedAirX).to.equal(-128);
    expect(p.speedY).to.equal(64);
    expect(p.speedGroundX).to.equal(32);
    expect(p.speedAttackX).to.equal(-16);
    expect(p.speedAttackY).to.equal(8);
    expect(p.stateAge).to.equal(12);
    expect(p.hitlag).to.equal(3);
    expect(p.stocks).to.equal(4);
    expect(p.facing).to.equal(1);
    expect(p.onGround).to.equal(1);
    expect(p.actionState).to.equal(44);
    expect(p.jumpsLeft).to.equal(2);
    expect(p.character).to.equal(18);
  });
});

describe("session sdk lifecycle (BOLT ECS localnet)", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const STAGE_FD = 31;
  const CHAR_MARTH = 18;
  const CHAR_FOX = 1;

  const MODEL_MANIFEST = Keypair.generate().publicKey;

  before(async function () {
    // These tests drive the raw SessionClient BOLT system calls. They require
    // the ECS system programs to already be deployed on the local validator.
    // If they are not present, fail fast with a clear message.
    await assertProgramIsExecutable(provider, SESSION_LIFECYCLE_PROGRAM_ID);
    await assertProgramIsExecutable(provider, SUBMIT_INPUT_PROGRAM_ID);
  });

  it("runs create -> join -> submit-input x2 -> deserialize session -> end", async () => {
    const player1 = Keypair.generate();
    const player2 = Keypair.generate();

    await airdrop(provider, player1.publicKey);
    await airdrop(provider, player2.publicKey);

    const baseConfig = {
      cluster: provider.connection.rpcEndpoint,
      modelManifest: MODEL_MANIFEST,
      stage: STAGE_FD,
      maxFrames: 600,
      dInner: 768,
      dState: 64,
      numLayers: 4,
    };

    const p1Client = new SessionClient(player1, {
      ...baseConfig,
      character: CHAR_MARTH,
    });

    const p2Client = new SessionClient(player2, {
      ...baseConfig,
      character: CHAR_FOX,
    });

    // CREATE
    const sessionKey = await p1Client.createSession();
    const accounts = p1Client.sessionAccounts;
    expect(accounts, "Session accounts should be populated after create").to.exist;

    // Owner sanity checks for allocated runtime accounts.
    const [sessionInfo, inputInfo, frameLogInfo] = await Promise.all([
      provider.connection.getAccountInfo(accounts!.sessionState.publicKey, "confirmed"),
      provider.connection.getAccountInfo(accounts!.inputBuffer.publicKey, "confirmed"),
      provider.connection.getAccountInfo(accounts!.frameLog.publicKey, "confirmed"),
    ]);
    expect(sessionInfo?.owner.toBase58()).to.equal(SESSION_LIFECYCLE_PROGRAM_ID.toBase58());
    expect(inputInfo?.owner.toBase58()).to.equal(SUBMIT_INPUT_PROGRAM_ID.toBase58());
    expect(frameLogInfo?.owner.toBase58()).to.equal(SESSION_LIFECYCLE_PROGRAM_ID.toBase58());

    let session = await p1Client.fetchSessionState();
    expect(session.status).to.equal(SessionStatus.WaitingPlayers);
    expect(session.frame).to.equal(0);
    expect(session.maxFrames).to.equal(600);
    expect(session.player1).to.equal(player1.publicKey.toBase58());
    expect(session.player2).to.equal(PublicKey.default.toBase58());
    expect(session.stage).to.equal(STAGE_FD);
    expect(session.model).to.equal(MODEL_MANIFEST.toBase58());
    expect(session.players[0].character).to.equal(CHAR_MARTH);
    expect(session.players[0].stocks).to.equal(4);
    expect(session.players[1].character).to.equal(0);

    // JOIN
    await p2Client.joinSession(sessionKey, accounts!);

    session = await p1Client.fetchSessionState();
    expect(session.status).to.equal(SessionStatus.Active);
    expect(session.player2).to.equal(player2.publicKey.toBase58());
    expect(session.stage).to.equal(STAGE_FD);

    // Starting positions (fixed-point x256)
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

    // SUBMIT INPUT (both players)
    await p1Client.sendInput({
      ...defaultInput(),
      stickX: 64,
      buttons: 0x01, // A
    });

    await p2Client.sendInput({
      ...defaultInput(),
      stickX: -32,
      triggerL: 255,
      buttonsExt: 0x04, // digital L
    });

    const inputAccount = await provider.connection.getAccountInfo(
      accounts!.inputBuffer.publicKey,
      "confirmed",
    );
    expect(inputAccount, "InputBuffer account missing").to.not.equal(null);
    const decodedInput = decodeInputBufferAccount(inputAccount!.data);

    // submit_input sets frame to session.frame + 1 (session.frame is still 0 until inference runs)
    expect(decodedInput.frame).to.equal(1);
    expect(decodedInput.p1Ready).to.equal(true);
    expect(decodedInput.p2Ready).to.equal(true);

    expect(decodedInput.player1.stickX).to.equal(64);
    expect(decodedInput.player1.buttons).to.equal(0x01);
    expect(decodedInput.player2.stickX).to.equal(-32);
    expect(decodedInput.player2.triggerL).to.equal(255);
    expect(decodedInput.player2.buttonsExt).to.equal(0x04);

    // Re-read and deserialize SessionState via SDK to verify offsets/math
    session = await p2Client.fetchSessionState();
    expect(session.status).to.equal(SessionStatus.Active);
    expect(session.frame).to.equal(0); // submit_input does not advance frame
    expect(session.maxFrames).to.equal(600);
    expect(session.stage).to.equal(STAGE_FD);
    expect(session.player1).to.equal(player1.publicKey.toBase58());
    expect(session.player2).to.equal(player2.publicKey.toBase58());
    expect(session.model).to.equal(MODEL_MANIFEST.toBase58());

    // Top-level timestamps are not set by lifecycle yet (Rust has TODOs)
    expect(session.createdAt).to.equal(0);
    expect(session.lastUpdate).to.equal(0);
    expect(session.seed).to.be.a("number");

    // END
    await p1Client.endSession();

    // p2Client is still attached, so use it to read final state after p1 client cleared its local session refs
    session = await p2Client.fetchSessionState();
    expect(session.status).to.equal(SessionStatus.Ended);
  });
});
