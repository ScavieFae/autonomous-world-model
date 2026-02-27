/**
 * Session management — create, join, play, and end sessions via BOLT ECS.
 *
 * Uses MagicBlock's BOLT SDK to route all system calls through a World program.
 * Session lifecycle:
 *   1. Initialize World + Entity + Components (one-time setup)
 *   2. ApplySystem(session_lifecycle, CREATE) → session active
 *   3. ApplySystem(session_lifecycle, JOIN) → both players connected
 *   4. ApplySystem(submit_input, ...) at 60fps per player
 *   5. ApplySystem(session_lifecycle, END) → session closed
 */

import {
  Connection,
  Keypair,
  PublicKey,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import {
  InitializeNewWorld,
  AddEntity,
  InitializeComponent,
  ApplySystem,
} from "@magicblock-labs/bolt-sdk";
import { SessionState, SessionStatus, VizFrame, sessionToVizFrame } from "./state";
import { ControllerInput, defaultInput } from "./input";

// ── Program IDs (must match declare_id! in Rust) ─────────────────────────────

/** Session lifecycle system program ID */
export const SESSION_LIFECYCLE_PROGRAM_ID = new PublicKey(
  "4ozheJvvMhG7yMrp1UR2kq1fhRvjXoY5Pn3NJ4nvAcyE"
);

/** Submit input system program ID */
export const SUBMIT_INPUT_PROGRAM_ID = new PublicKey(
  "F9ZqWHVDtsXZdHLU8MXfybsS1W3TTGv4NegcJZK9LnWx"
);

/** Run inference system program ID */
export const RUN_INFERENCE_PROGRAM_ID = new PublicKey(
  "3tHPJJSNhKwbp7K5vSYCUdYVX9bGxRCmpddwaJWRKPyb"
);

/** Component program IDs */
export const SESSION_STATE_PROGRAM_ID = new PublicKey(
  "FJwbNTbGHSpq4a72ro1aza53kvs7YMNT7J5U34kaosFj"
);
export const HIDDEN_STATE_PROGRAM_ID = new PublicKey(
  "Ea3VKF8CW3svQwiT8pn13JVdbVhLHSBURtNuanagc4hs"
);
export const INPUT_BUFFER_PROGRAM_ID = new PublicKey(
  "3R2RbzwP54qdyXcyiwHW2Sj6uVwf4Dhy7Zy8RcSVHFpq"
);
export const FRAME_LOG_PROGRAM_ID = new PublicKey(
  "3mWTNv5jhzLnpG4Xt9XqM1b2nbNpizoGEJxepUhhoaNK"
);

// ── Lifecycle action codes ──────────────────────────────────────────────────

const ACTION_CREATE = 0;
const ACTION_JOIN = 1;
const ACTION_END = 2;

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

// ── BOLT session accounts (PDAs, not keypairs) ─────────────────────────────

export interface BoltSessionAccounts {
  worldPda: PublicKey;
  entityPda: PublicKey;
  sessionStatePda: PublicKey;
  hiddenStatePda: PublicKey;
  inputBufferPda: PublicKey;
  frameLogPda: PublicKey;
}

// ── Session client ──────────────────────────────────────────────────────────

export class SessionClient {
  private connection: Connection;
  private accounts?: BoltSessionAccounts;
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

  /** Get the session entity PDA (available after create or join). */
  get entityPda(): PublicKey | undefined {
    return this.accounts?.entityPda;
  }

  /** Get the session state component PDA. */
  get sessionStatePda(): PublicKey | undefined {
    return this.accounts?.sessionStatePda;
  }

  /** Get all BOLT session accounts. */
  get boltAccounts(): BoltSessionAccounts | undefined {
    return this.accounts;
  }

  /** Attach to an existing session's BOLT accounts. */
  attachSession(accounts: BoltSessionAccounts, playerNumber: 1 | 2 = 1) {
    this.accounts = accounts;
    this.playerNumber = playerNumber;
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

  // ── BOLT ECS setup ─────────────────────────────────────────────────

  /**
   * Create a new session via BOLT ECS.
   *
   * 1. InitializeNewWorld → worldPda
   * 2. AddEntity → entityPda
   * 3. InitializeComponent × 4 (session_state, hidden_state, input_buffer, frame_log)
   * 4. ApplySystem(session_lifecycle, CREATE args)
   */
  async createSession(): Promise<PublicKey> {
    this.emitStatus("Creating BOLT world...");
    this.playerNumber = 1;

    // 1. Initialize World
    const initWorld = await InitializeNewWorld({
      payer: this.player.publicKey,
      connection: this.connection,
    });
    await sendAndConfirmTransaction(
      this.connection,
      initWorld.transaction,
      [this.player],
    );
    const worldPda = initWorld.worldPda;

    this.emitStatus("Adding entity...");

    // 2. Add Entity
    const addEntity = await AddEntity({
      payer: this.player.publicKey,
      world: worldPda,
      connection: this.connection,
    });
    await sendAndConfirmTransaction(
      this.connection,
      addEntity.transaction,
      [this.player],
    );
    const entityPda = addEntity.entityPda;

    this.emitStatus("Initializing components...");

    // 3. Initialize Components
    // Order matters: must match #[system_input] struct order for systems
    const componentIds = [
      SESSION_STATE_PROGRAM_ID,
      HIDDEN_STATE_PROGRAM_ID,
      INPUT_BUFFER_PROGRAM_ID,
      FRAME_LOG_PROGRAM_ID,
    ];

    const componentPdas: PublicKey[] = [];
    for (const componentId of componentIds) {
      const initComp = await InitializeComponent({
        payer: this.player.publicKey,
        entity: entityPda,
        componentId,
      });
      await sendAndConfirmTransaction(
        this.connection,
        initComp.transaction,
        [this.player],
      );
      componentPdas.push(initComp.componentPda);
    }

    this.accounts = {
      worldPda,
      entityPda,
      sessionStatePda: componentPdas[0],
      hiddenStatePda: componentPdas[1],
      inputBufferPda: componentPdas[2],
      frameLogPda: componentPdas[3],
    };

    this.emitStatus("Calling session_lifecycle CREATE...");

    // 4. ApplySystem — CREATE
    const createResult = await ApplySystem({
      authority: this.player.publicKey,
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
        action: ACTION_CREATE,
        player: this.player.publicKey.toBase58(),
        character: this.config.character,
        stage: this.config.stage,
        model: (this.config.modelManifest).toBase58(),
        max_frames: this.config.maxFrames ?? 0,
        seed: Date.now(),
        d_inner: this.config.dInner ?? 768,
        d_state: this.config.dState ?? 64,
        num_layers: this.config.numLayers ?? 4,
      },
    });
    await sendAndConfirmTransaction(
      this.connection,
      createResult.transaction,
      [this.player],
    );

    this.emitStatus(`Session created: entity=${entityPda.toBase58().slice(0, 8)}...`);
    this.emitStatus("Waiting for player 2...");

    return entityPda;
  }

  /**
   * Join an existing session as player 2.
   */
  async joinSession(accounts: BoltSessionAccounts): Promise<void> {
    this.accounts = accounts;
    this.playerNumber = 2;
    this.emitStatus(`Joining session...`);

    const joinResult = await ApplySystem({
      authority: this.player.publicKey,
      systemId: SESSION_LIFECYCLE_PROGRAM_ID,
      world: accounts.worldPda,
      entities: [{
        entity: accounts.entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: HIDDEN_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
          { componentId: FRAME_LOG_PROGRAM_ID },
        ],
      }],
      args: {
        action: ACTION_JOIN,
        player: this.player.publicKey.toBase58(),
        character: this.config.character,
        stage: 0,
        model: PublicKey.default.toBase58(),
        max_frames: 0,
        seed: 0,
        d_inner: 0,
        d_state: 0,
        num_layers: 0,
      },
    });
    await sendAndConfirmTransaction(
      this.connection,
      joinResult.transaction,
      [this.player],
    );

    this.emitStatus("Joined! Session active.");
  }

  /**
   * Send a single frame's controller input via submit_input system.
   */
  async sendInput(input: ControllerInput): Promise<void> {
    if (!this.accounts) return;

    const inputResult = await ApplySystem({
      authority: this.player.publicKey,
      systemId: SUBMIT_INPUT_PROGRAM_ID,
      world: this.accounts.worldPda,
      entities: [{
        entity: this.accounts.entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
        ],
      }],
      args: {
        player: this.player.publicKey.toBase58(),
        stick_x: input.stickX,
        stick_y: input.stickY,
        c_stick_x: input.cStickX,
        c_stick_y: input.cStickY,
        trigger_l: input.triggerL,
        trigger_r: input.triggerR,
        buttons: input.buttons,
        buttons_ext: input.buttonsExt,
      },
    });
    await sendAndConfirmTransaction(
      this.connection,
      inputResult.transaction,
      [this.player],
    );
  }

  /**
   * Start the play loop: subscribe to state changes, send inputs at 60fps.
   */
  async startPlaying(getInput: () => ControllerInput): Promise<void> {
    if (!this.accounts) throw new Error("No active session");

    this.emitStatus("Starting play loop...");

    // Subscribe to SessionState component account changes
    this.subscriptionId = this.connection.onAccountChange(
      this.accounts.sessionStatePda,
      (accountInfo) => {
        try {
          const session = deserializeSessionState(accountInfo.data);
          const frame = sessionToVizFrame(session);
          for (const cb of this.frameCallbacks) cb(frame);
        } catch (e) {
          console.warn("Failed to deserialize frame:", e);
        }
      },
      "processed"
    );

    // Send inputs at 60fps
    this.inputInterval = setInterval(() => {
      this.currentInput = getInput();
      this.sendInput(this.currentInput).catch((e) => {
        console.warn("Failed to send input:", e);
      });
    }, 16);

    this.emitStatus("Playing!");
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
   * End the session.
   */
  async endSession(): Promise<void> {
    this.stopPlaying();
    if (!this.accounts) return;

    this.emitStatus("Ending session...");

    const endResult = await ApplySystem({
      authority: this.player.publicKey,
      systemId: SESSION_LIFECYCLE_PROGRAM_ID,
      world: this.accounts.worldPda,
      entities: [{
        entity: this.accounts.entityPda,
        components: [
          { componentId: SESSION_STATE_PROGRAM_ID },
          { componentId: HIDDEN_STATE_PROGRAM_ID },
          { componentId: INPUT_BUFFER_PROGRAM_ID },
          { componentId: FRAME_LOG_PROGRAM_ID },
        ],
      }],
      args: {
        action: ACTION_END,
        player: this.player.publicKey.toBase58(),
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
    await sendAndConfirmTransaction(
      this.connection,
      endResult.transaction,
      [this.player],
    );

    this.emitStatus("Session ended.");
    this.accounts = undefined;
  }

  /**
   * Fetch and deserialize the current SessionState from the component PDA.
   */
  async fetchSessionState(): Promise<SessionState> {
    if (!this.accounts) throw new Error("No active session");
    const account = await this.connection.getAccountInfo(
      this.accounts.sessionStatePda,
      "confirmed",
    );
    if (!account) throw new Error("SessionState account not found");
    return deserializeSessionState(account.data);
  }
}

// ── Deserialization ───────────────────────────────────────────────────────────

// BOLT #[component] may add a bolt_metadata field (Pubkey, 32 bytes) before
// the Anchor discriminator. The exact offset depends on the BOLT version.
// We probe for the correct offset by checking known patterns.

const PLAYER_STATE_SIZE = 32; // 4+4+2+2+2*5+2+1*2+1*2+2+1*2 = 32 bytes

/**
 * Deserialize raw account data into SessionState.
 *
 * BOLT components have layout: [8-byte anchor discriminator][bolt_metadata?][fields]
 * We try the standard 8-byte offset first, then 8+32 if bolt_metadata is present.
 */
export function deserializeSessionState(data: Buffer): SessionState {
  // Standard Anchor: 8-byte discriminator
  // BOLT may add 32-byte bolt_metadata (Pubkey) after discriminator
  // Try 8 first — if status looks wrong, try 8+32
  let offset = 8;

  const status = data.readUInt8(offset);
  if (status > 3) {
    // Probably hit bolt_metadata — skip 32 more bytes
    offset = 8 + 32;
  }

  return parseSessionStateAt(data, offset);
}

function parseSessionStateAt(data: Buffer, offset: number): SessionState {
  const status = data.readUInt8(offset); offset += 1;
  const frame = data.readUInt32LE(offset); offset += 4;
  const maxFrames = data.readUInt32LE(offset); offset += 4;

  const player1 = new PublicKey(data.subarray(offset, offset + 32)).toBase58();
  offset += 32;
  const player2 = new PublicKey(data.subarray(offset, offset + 32)).toBase58();
  offset += 32;

  const stage = data.readUInt8(offset); offset += 1;

  const players: [import("./state").PlayerState, import("./state").PlayerState] = [
    deserializePlayerState(data, offset),
    deserializePlayerState(data, offset + PLAYER_STATE_SIZE),
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

function deserializePlayerState(data: Buffer, offset: number): import("./state").PlayerState {
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

export async function listModels(
  connection: Connection,
  programId: PublicKey
): Promise<ModelInfo[]> {
  // In production: use getProgramAccounts with memcmp filter
  return [];
}
