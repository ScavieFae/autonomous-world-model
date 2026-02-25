/**
 * Session management — create, join, play, and end sessions.
 *
 * Handles the full session lifecycle:
 *   1. Browse available models (read ModelManifest accounts)
 *   2. Create session (allocate accounts, delegate to ephemeral rollup)
 *   3. Join session (player 2 connects)
 *   4. Play (send inputs at 60fps, receive state via subscription)
 *   5. End session (undelegate, commit to mainnet)
 */

import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  TransactionInstruction,
} from "@solana/web3.js";
import { SessionState, SessionStatus, VizFrame, sessionToVizFrame } from "./state";
import { ControllerInput, defaultInput } from "./input";

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
}

// ── Session client ──────────────────────────────────────────────────────────

export class SessionClient {
  private connection: Connection;
  private sessionKey?: PublicKey;
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

  // ── Session lifecycle ─────────────────────────────────────────────────

  /**
   * Create a new session. Returns the session public key.
   *
   * Creates SessionState, HiddenState, InputBuffer, and FrameLog accounts,
   * then delegates them to the ephemeral rollup.
   */
  async createSession(): Promise<PublicKey> {
    this.emitStatus("Creating session...");

    const sessionKeypair = Keypair.generate();
    this.sessionKey = sessionKeypair.publicKey;
    this.playerNumber = 1;

    // In production:
    // 1. Create all 4 session accounts via SystemProgram.createAccount
    // 2. Call session_lifecycle::execute with ACTION_CREATE
    // 3. Delegate accounts to ephemeral rollup via MagicBlock delegation program

    this.emitStatus(`Session created: ${this.sessionKey.toBase58().slice(0, 8)}...`);
    this.emitStatus("Waiting for player 2...");

    return this.sessionKey;
  }

  /**
   * Join an existing session as player 2.
   */
  async joinSession(sessionKey: PublicKey): Promise<void> {
    this.sessionKey = sessionKey;
    this.playerNumber = 2;
    this.emitStatus(`Joining session ${sessionKey.toBase58().slice(0, 8)}...`);

    // In production:
    // 1. Call session_lifecycle::execute with ACTION_JOIN
    // 2. Session transitions to Active

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
        // Deserialize account data to SessionState
        // In production, use Anchor's account deserializer
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
   * Send a single frame's controller input.
   */
  private async sendInput(input: ControllerInput): Promise<void> {
    if (!this.sessionKey) return;

    // In production:
    // Call submit_input::execute with controller state
    // Transaction sent via WebSocket for minimum latency
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
   * End the session. Undelegates accounts back to mainnet.
   */
  async endSession(): Promise<void> {
    this.stopPlaying();

    if (!this.sessionKey) return;

    this.emitStatus("Ending session...");

    // In production:
    // 1. Call session_lifecycle::execute with ACTION_END
    // 2. Undelegate accounts from ephemeral rollup
    // 3. Close session accounts to reclaim rent

    this.emitStatus("Session ended.");
    this.sessionKey = undefined;
  }

  // ── Deserialization ───────────────────────────────────────────────────

  /**
   * Deserialize raw account data into SessionState.
   * In production, use Anchor's generated deserializer.
   */
  private deserializeSessionState(data: Buffer): SessionState {
    // Skip Anchor discriminator (8 bytes)
    let offset = 8;

    const status = data.readUInt8(offset); offset += 1;
    const frame = data.readUInt32LE(offset); offset += 4;
    const maxFrames = data.readUInt32LE(offset); offset += 4;

    // Skip pubkeys (32 bytes each) and other fields for now
    // In production, use the generated Anchor type

    return {
      status,
      frame,
      maxFrames,
      player1: "",
      player2: "",
      stage: 0,
      players: [
        { x: 0, y: 0, percent: 0, shieldStrength: 0, speedAirX: 0, speedY: 0,
          speedGroundX: 0, speedAttackX: 0, speedAttackY: 0, stateAge: 0,
          hitlag: 0, stocks: 4, facing: 1, onGround: 1, actionState: 0,
          jumpsLeft: 2, character: 0 },
        { x: 0, y: 0, percent: 0, shieldStrength: 0, speedAirX: 0, speedY: 0,
          speedGroundX: 0, speedAttackX: 0, speedAttackY: 0, stateAge: 0,
          hitlag: 0, stocks: 4, facing: 0, onGround: 1, actionState: 0,
          jumpsLeft: 2, character: 0 },
      ],
      model: "",
      createdAt: 0,
      lastUpdate: 0,
      seed: 0,
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
