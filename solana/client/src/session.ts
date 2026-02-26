/**
 * Session management — create, join, play, and end sessions.
 *
 * Handles the full session lifecycle:
 *   1. Browse available models (read ModelManifest accounts)
 *   2. Create session (allocate accounts, delegate to ephemeral rollup)
 *   3. Join session (player 2 connects)
 *   4. Play (send inputs at 60fps, receive state via subscription)
 *   5. End session (undelegate, commit to mainnet)
 *
 * Uses BOLT ECS system calls (session_lifecycle, submit_input) via Anchor.
 */

import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  TransactionInstruction,
  SystemProgram,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import { SessionState, SessionStatus, PlayerState, VizFrame, sessionToVizFrame } from "./state";
import { ControllerInput, defaultInput } from "./input";

// ── Program IDs (must match declare_id! in Rust) ─────────────────────────────

/** Session lifecycle system program ID */
export const SESSION_LIFECYCLE_PROGRAM_ID = new PublicKey(
  "SessLife11111111111111111111111111111111111"
);

/** Submit input system program ID */
export const SUBMIT_INPUT_PROGRAM_ID = new PublicKey(
  "SubInput11111111111111111111111111111111111"
);

/** SessionState component program ID */
export const SESSION_STATE_PROGRAM_ID = new PublicKey(
  "SessState1111111111111111111111111111111111"
);

/** InputBuffer component program ID */
export const INPUT_BUFFER_PROGRAM_ID = new PublicKey(
  "InpBuffer1111111111111111111111111111111111"
);

/** HiddenState component program ID */
export const HIDDEN_STATE_PROGRAM_ID = new PublicKey(
  "HdnState11111111111111111111111111111111111"
);

// ── Lifecycle action codes ──────────────────────────────────────────────────

const ACTION_CREATE = 0;
const ACTION_JOIN = 1;
const ACTION_END = 2;

// ── Account sizes (for rent calculation) ────────────────────────────────────

/** Anchor discriminator (8) + SessionState fields */
const PLAYER_STATE_SIZE = 32; // 4+4+2+2+2*5+2+1*2+1*2+2+1*2 = 32 bytes
const SESSION_STATE_SIZE = 8 + 1 + 4 + 4 + 32 + 32 + 1 + (PLAYER_STATE_SIZE * 2) + 32 + 8 + 8 + 8;

/** Anchor discriminator (8) + HiddenState header + data buffer */
const HIDDEN_STATE_SIZE = 8 + 1 + 2 + 2 + 4 + 4 + 1 + 200_000;

/** Anchor discriminator (8) + InputBuffer fields */
const INPUT_BUFFER_SIZE = 8 + 4 + (8 * 2) + 1 + 1;

/** Anchor discriminator (8) + FrameLog header + ring buffer */
const FRAME_LOG_SIZE = 8 + 4 + 4 + (68 * 256);

// ── Session configuration ───────────────────────────────────────────────────

export interface SessionConfig {
  /** Cluster URL (mainnet, devnet, or ephemeral rollup endpoint) */
  cluster: string;
  /** Ephemeral rollup WebSocket endpoint (for low-latency play) */
  ephemeralWs?: string;
  /** Model manifest public key */
  modelManifest: PublicKey;
  /** Stage ID (0-32) */
  stage: number;
  /** Character ID for this player (0-32) */
  character: number;
  /** Max frames (0 = unlimited) */
  maxFrames?: number;
  /** Model dimensions (for HiddenState allocation) */
  dInner?: number;
  dState?: number;
  numLayers?: number;
}

// ── Session accounts ────────────────────────────────────────────────────────

export interface SessionAccounts {
  sessionState: Keypair;
  hiddenState: Keypair;
  inputBuffer: Keypair;
  frameLog: Keypair;
}

// ── Session client ──────────────────────────────────────────────────────────

export class SessionClient {
  private connection: Connection;
  private sessionKey?: PublicKey;
  private accounts?: SessionAccounts;
  private player: Keypair;
  private playerNumber: 1 | 2 = 1;
  private config: SessionConfig;
  private subscriptionId?: number;
  private inputInterval?: ReturnType<typeof setInterval>;
  private currentInput: ControllerInput = defaultInput();
  private frameCallbacks: ((frame: VizFrame) => void)[] = [];
  private statusCallbacks: ((status: string) => void)[] = [];

  constructor(player: Keypair, config: SessionConfig) {
    this.player = player;
    this.config = config;
    this.connection = new Connection(config.cluster, {
      commitment: "confirmed",
      wsEndpoint: config.ephemeralWs,
    });
  }

  /** Get the session public key (available after create or join). */
  get sessionPublicKey(): PublicKey | undefined {
    return this.sessionKey;
  }

  /** Get the session accounts (available after create). */
  get sessionAccounts(): SessionAccounts | undefined {
    return this.accounts;
  }

  /** Register a callback for new frames. */
  onFrame(cb: (frame: VizFrame) => void) {
    this.frameCallbacks.push(cb);
  }

  /** Register a callback for status changes. */
  onStatus(cb: (status: string) => void) {
    this.statusCallbacks.push(cb);
  }

  private emitStatus(status: string) {
    for (const cb of this.statusCallbacks) cb(status);
  }

  // ── Account allocation ────────────────────────────────────────────────

  /**
   * Allocate the 4 session accounts via SystemProgram.createAccount.
   * Each account needs enough space for the component data + Anchor discriminator.
   */
  private async allocateAccounts(): Promise<SessionAccounts> {
    const accounts: SessionAccounts = {
      sessionState: Keypair.generate(),
      hiddenState: Keypair.generate(),
      inputBuffer: Keypair.generate(),
      frameLog: Keypair.generate(),
    };

    const allocations = [
      { keypair: accounts.sessionState, size: SESSION_STATE_SIZE, owner: SESSION_STATE_PROGRAM_ID },
      { keypair: accounts.hiddenState, size: HIDDEN_STATE_SIZE, owner: HIDDEN_STATE_PROGRAM_ID },
      { keypair: accounts.inputBuffer, size: INPUT_BUFFER_SIZE, owner: INPUT_BUFFER_PROGRAM_ID },
      { keypair: accounts.frameLog, size: FRAME_LOG_SIZE, owner: SESSION_STATE_PROGRAM_ID },
    ];

    for (const { keypair, size, owner } of allocations) {
      const lamports = await this.connection.getMinimumBalanceForRentExemption(size);
      const tx = new Transaction().add(
        SystemProgram.createAccount({
          fromPubkey: this.player.publicKey,
          newAccountPubkey: keypair.publicKey,
          lamports,
          space: size,
          programId: owner,
        })
      );
      await sendAndConfirmTransaction(this.connection, tx, [this.player, keypair]);
    }

    return accounts;
  }

  // ── Session lifecycle ─────────────────────────────────────────────────

  /**
   * Create a new session. Allocates 4 accounts and calls session_lifecycle
   * with ACTION_CREATE.
   *
   * Returns the session public key.
   */
  async createSession(): Promise<PublicKey> {
    this.emitStatus("Creating session...");

    // 1. Allocate session accounts
    this.accounts = await this.allocateAccounts();
    this.sessionKey = this.accounts.sessionState.publicKey;
    this.playerNumber = 1;

    this.emitStatus("Accounts allocated, calling session_lifecycle CREATE...");

    // 2. Build session_lifecycle::execute instruction with ACTION_CREATE
    //
    // In a full Anchor/BOLT integration, this would use the generated IDL:
    //   await sessionLifecycle.methods.execute({ action: 0, ... }).accounts({...}).rpc()
    //
    // For the SDK, we build the instruction manually so we don't require
    // the Anchor workspace to be loaded. The instruction data is:
    //   [8-byte discriminator][serialized Args struct]
    const createArgs = this.encodeLifecycleArgs({
      action: ACTION_CREATE,
      player: this.player.publicKey,
      character: this.config.character,
      stage: this.config.stage,
      model: this.config.modelManifest,
      maxFrames: this.config.maxFrames ?? 0,
      seed: BigInt(Date.now()),
      dInner: this.config.dInner ?? 768,
      dState: this.config.dState ?? 64,
      numLayers: this.config.numLayers ?? 4,
    });

    const createIx = new TransactionInstruction({
      programId: SESSION_LIFECYCLE_PROGRAM_ID,
      keys: [
        { pubkey: this.accounts.sessionState.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.accounts.hiddenState.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.accounts.inputBuffer.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.accounts.frameLog.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.player.publicKey, isSigner: true, isWritable: false },
      ],
      data: createArgs,
    });

    const tx = new Transaction().add(createIx);
    await sendAndConfirmTransaction(this.connection, tx, [this.player]);

    this.emitStatus(`Session created: ${this.sessionKey.toBase58().slice(0, 8)}...`);
    this.emitStatus("Waiting for player 2...");

    // 3. TODO: Delegate accounts to ephemeral rollup via MagicBlock SDK
    // await delegateToEphemeralRollup(this.connection, this.accounts, this.player);

    return this.sessionKey;
  }

  /**
   * Join an existing session as player 2.
   * Calls session_lifecycle with ACTION_JOIN.
   */
  async joinSession(
    sessionKey: PublicKey,
    accounts: SessionAccounts,
  ): Promise<void> {
    this.sessionKey = sessionKey;
    this.accounts = accounts;
    this.playerNumber = 2;
    this.emitStatus(`Joining session ${sessionKey.toBase58().slice(0, 8)}...`);

    const joinArgs = this.encodeLifecycleArgs({
      action: ACTION_JOIN,
      player: this.player.publicKey,
      character: this.config.character,
      stage: 0,  // Ignored for JOIN
      model: PublicKey.default,  // Ignored for JOIN
      maxFrames: 0,
      seed: BigInt(0),
      dInner: 0,
      dState: 0,
      numLayers: 0,
    });

    const joinIx = new TransactionInstruction({
      programId: SESSION_LIFECYCLE_PROGRAM_ID,
      keys: [
        { pubkey: accounts.sessionState.publicKey, isSigner: false, isWritable: true },
        { pubkey: accounts.hiddenState.publicKey, isSigner: false, isWritable: true },
        { pubkey: accounts.inputBuffer.publicKey, isSigner: false, isWritable: true },
        { pubkey: accounts.frameLog.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.player.publicKey, isSigner: true, isWritable: false },
      ],
      data: joinArgs,
    });

    const tx = new Transaction().add(joinIx);
    await sendAndConfirmTransaction(this.connection, tx, [this.player]);

    this.emitStatus("Joined! Session active.");
  }

  /**
   * Start the play loop: subscribe to state changes, send inputs at 60fps.
   */
  async startPlaying(getInput: () => ControllerInput): Promise<void> {
    if (!this.sessionKey) throw new Error("No active session");

    this.emitStatus("Starting play loop...");

    // Subscribe to SessionState account changes
    this.subscriptionId = this.connection.onAccountChange(
      this.sessionKey,
      (accountInfo) => {
        try {
          const session = this.deserializeSessionState(accountInfo.data);
          const frame = sessionToVizFrame(session);
          for (const cb of this.frameCallbacks) cb(frame);
        } catch (e) {
          console.warn("Failed to deserialize frame:", e);
        }
      },
      "processed" // Fastest commitment for real-time play
    );

    // Send inputs at 60fps
    this.inputInterval = setInterval(() => {
      this.currentInput = getInput();
      this.sendInput(this.currentInput).catch((e) => {
        console.warn("Failed to send input:", e);
      });
    }, 16); // ~60fps (16.67ms)

    this.emitStatus("Playing!");
  }

  /**
   * Send a single frame's controller input via submit_input system.
   */
  private async sendInput(input: ControllerInput): Promise<void> {
    if (!this.sessionKey || !this.accounts) return;

    const inputArgs = this.encodeInputArgs({
      player: this.player.publicKey,
      stickX: input.stickX,
      stickY: input.stickY,
      cStickX: input.cStickX,
      cStickY: input.cStickY,
      triggerL: input.triggerL,
      triggerR: input.triggerR,
      buttons: input.buttons,
      buttonsExt: input.buttonsExt,
    });

    const inputIx = new TransactionInstruction({
      programId: SUBMIT_INPUT_PROGRAM_ID,
      keys: [
        { pubkey: this.accounts.sessionState.publicKey, isSigner: false, isWritable: false },
        { pubkey: this.accounts.inputBuffer.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.player.publicKey, isSigner: true, isWritable: false },
      ],
      data: inputArgs,
    });

    const tx = new Transaction().add(inputIx);
    await sendAndConfirmTransaction(this.connection, tx, [this.player]);
  }

  /**
   * Stop the play loop.
   */
  stopPlaying() {
    if (this.subscriptionId !== undefined) {
      this.connection.removeAccountChangeListener(this.subscriptionId);
      this.subscriptionId = undefined;
    }
    if (this.inputInterval) {
      clearInterval(this.inputInterval);
      this.inputInterval = undefined;
    }
    this.emitStatus("Stopped.");
  }

  /**
   * End the session. Calls session_lifecycle with ACTION_END,
   * then undelegates accounts back to mainnet.
   */
  async endSession(): Promise<void> {
    this.stopPlaying();

    if (!this.sessionKey || !this.accounts) return;

    this.emitStatus("Ending session...");

    const endArgs = this.encodeLifecycleArgs({
      action: ACTION_END,
      player: this.player.publicKey,
      character: 0,
      stage: 0,
      model: PublicKey.default,
      maxFrames: 0,
      seed: BigInt(0),
      dInner: 0,
      dState: 0,
      numLayers: 0,
    });

    const endIx = new TransactionInstruction({
      programId: SESSION_LIFECYCLE_PROGRAM_ID,
      keys: [
        { pubkey: this.accounts.sessionState.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.accounts.hiddenState.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.accounts.inputBuffer.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.accounts.frameLog.publicKey, isSigner: false, isWritable: true },
        { pubkey: this.player.publicKey, isSigner: true, isWritable: false },
      ],
      data: endArgs,
    });

    const tx = new Transaction().add(endIx);
    await sendAndConfirmTransaction(this.connection, tx, [this.player]);

    // TODO: Undelegate accounts from ephemeral rollup
    // TODO: Close session accounts to reclaim rent

    this.emitStatus("Session ended.");
    this.sessionKey = undefined;
    this.accounts = undefined;
  }

  // ── Instruction encoding ──────────────────────────────────────────────

  /**
   * Encode session_lifecycle Args struct to instruction data.
   * Layout: [8-byte Anchor discriminator][args fields]
   *
   * Note: In production with the Anchor IDL loaded, use
   * program.methods.execute({...}).instruction() instead.
   */
  private encodeLifecycleArgs(args: {
    action: number;
    player: PublicKey;
    character: number;
    stage: number;
    model: PublicKey;
    maxFrames: number;
    seed: bigint;
    dInner: number;
    dState: number;
    numLayers: number;
  }): Buffer {
    // Anchor discriminator for "execute" (first 8 bytes of SHA256("global:execute"))
    const discriminator = Buffer.from([0x0b, 0xed, 0x60, 0x84, 0x3d, 0x04, 0xea, 0xf8]);

    const buf = Buffer.alloc(
      8 +     // discriminator
      1 +     // action (u8)
      32 +    // player (Pubkey)
      1 +     // character (u8)
      1 +     // stage (u8)
      32 +    // model (Pubkey)
      4 +     // max_frames (u32)
      8 +     // seed (u64)
      2 +     // d_inner (u16)
      2 +     // d_state (u16)
      1       // num_layers (u8)
    );

    let offset = 0;
    discriminator.copy(buf, offset); offset += 8;
    buf.writeUInt8(args.action, offset); offset += 1;
    args.player.toBuffer().copy(buf, offset); offset += 32;
    buf.writeUInt8(args.character, offset); offset += 1;
    buf.writeUInt8(args.stage, offset); offset += 1;
    args.model.toBuffer().copy(buf, offset); offset += 32;
    buf.writeUInt32LE(args.maxFrames, offset); offset += 4;
    buf.writeBigUInt64LE(args.seed, offset); offset += 8;
    buf.writeUInt16LE(args.dInner, offset); offset += 2;
    buf.writeUInt16LE(args.dState, offset); offset += 2;
    buf.writeUInt8(args.numLayers, offset); offset += 1;

    return buf;
  }

  /**
   * Encode submit_input Args struct to instruction data.
   */
  private encodeInputArgs(args: {
    player: PublicKey;
    stickX: number;
    stickY: number;
    cStickX: number;
    cStickY: number;
    triggerL: number;
    triggerR: number;
    buttons: number;
    buttonsExt: number;
  }): Buffer {
    const discriminator = Buffer.from([0x0b, 0xed, 0x60, 0x84, 0x3d, 0x04, 0xea, 0xf8]);

    const buf = Buffer.alloc(
      8 +     // discriminator
      32 +    // player (Pubkey)
      1 + 1 + // stick_x, stick_y (i8)
      1 + 1 + // c_stick_x, c_stick_y (i8)
      1 + 1 + // trigger_l, trigger_r (u8)
      1 + 1   // buttons, buttons_ext (u8)
    );

    let offset = 0;
    discriminator.copy(buf, offset); offset += 8;
    args.player.toBuffer().copy(buf, offset); offset += 32;
    buf.writeInt8(args.stickX, offset); offset += 1;
    buf.writeInt8(args.stickY, offset); offset += 1;
    buf.writeInt8(args.cStickX, offset); offset += 1;
    buf.writeInt8(args.cStickY, offset); offset += 1;
    buf.writeUInt8(args.triggerL, offset); offset += 1;
    buf.writeUInt8(args.triggerR, offset); offset += 1;
    buf.writeUInt8(args.buttons, offset); offset += 1;
    buf.writeUInt8(args.buttonsExt, offset); offset += 1;

    return buf;
  }

  // ── Deserialization ───────────────────────────────────────────────────

  /**
   * Deserialize raw account data into SessionState.
   *
   * Reads the full account layout matching the Rust SessionState component:
   * [8-byte discriminator][status][frame][max_frames][player1][player2]
   * [stage][PlayerState×2][model][created_at][last_update][seed]
   */
  private deserializeSessionState(data: Buffer): SessionState {
    let offset = 8; // Skip Anchor discriminator

    const status = data.readUInt8(offset); offset += 1;
    const frame = data.readUInt32LE(offset); offset += 4;
    const maxFrames = data.readUInt32LE(offset); offset += 4;

    // Player pubkeys
    const player1 = new PublicKey(data.subarray(offset, offset + 32)).toBase58();
    offset += 32;
    const player2 = new PublicKey(data.subarray(offset, offset + 32)).toBase58();
    offset += 32;

    const stage = data.readUInt8(offset); offset += 1;

    // Deserialize both PlayerStates
    const players: [PlayerState, PlayerState] = [
      this.deserializePlayerState(data, offset),
      this.deserializePlayerState(data, offset + PLAYER_STATE_SIZE),
    ];
    offset += PLAYER_STATE_SIZE * 2;

    const model = new PublicKey(data.subarray(offset, offset + 32)).toBase58();
    offset += 32;

    const createdAtLow = data.readUInt32LE(offset);
    const createdAtHigh = data.readInt32LE(offset + 4);
    const createdAt = createdAtLow + createdAtHigh * 0x100000000;
    offset += 8;

    const lastUpdateLow = data.readUInt32LE(offset);
    const lastUpdateHigh = data.readInt32LE(offset + 4);
    const lastUpdate = lastUpdateLow + lastUpdateHigh * 0x100000000;
    offset += 8;

    const seedLow = data.readUInt32LE(offset);
    const seedHigh = data.readUInt32LE(offset + 4);
    const seed = seedLow + seedHigh * 0x100000000;

    return {
      status,
      frame,
      maxFrames,
      player1,
      player2,
      stage,
      players,
      model,
      createdAt,
      lastUpdate,
      seed,
    };
  }

  /**
   * Deserialize a single PlayerState from raw bytes.
   *
   * Layout (32 bytes total):
   * x(i32), y(i32), percent(u16), shield_strength(u16),
   * speed_air_x(i16), speed_y(i16), speed_ground_x(i16),
   * speed_attack_x(i16), speed_attack_y(i16),
   * state_age(u16), hitlag(u8), stocks(u8),
   * facing(u8), on_ground(u8), action_state(u16),
   * jumps_left(u8), character(u8)
   */
  private deserializePlayerState(data: Buffer, offset: number): PlayerState {
    return {
      x: data.readInt32LE(offset),
      y: data.readInt32LE(offset + 4),
      percent: data.readUInt16LE(offset + 8),
      shieldStrength: data.readUInt16LE(offset + 10),
      speedAirX: data.readInt16LE(offset + 12),
      speedY: data.readInt16LE(offset + 14),
      speedGroundX: data.readInt16LE(offset + 16),
      speedAttackX: data.readInt16LE(offset + 18),
      speedAttackY: data.readInt16LE(offset + 20),
      stateAge: data.readUInt16LE(offset + 22),
      hitlag: data.readUInt8(offset + 24),
      stocks: data.readUInt8(offset + 25),
      facing: data.readUInt8(offset + 26),
      onGround: data.readUInt8(offset + 27),
      actionState: data.readUInt16LE(offset + 28),
      jumpsLeft: data.readUInt8(offset + 30),
      character: data.readUInt8(offset + 31),
    };
  }
}

// ── Model browser ───────────────────────────────────────────────────────────

export interface ModelInfo {
  address: PublicKey;
  name: string;
  version: number;
  dModel: number;
  dInner: number;
  dState: number;
  numLayers: number;
  totalParams: number;
  ready: boolean;
}

/**
 * Browse available models by reading ModelManifest accounts on mainnet.
 */
export async function listModels(
  connection: Connection,
  programId: PublicKey
): Promise<ModelInfo[]> {
  // In production: use getProgramAccounts with memcmp filter on
  // the ModelManifest discriminator to find all manifest accounts

  // Placeholder: return empty list
  return [];
}
