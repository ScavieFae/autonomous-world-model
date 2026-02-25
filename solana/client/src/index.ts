/**
 * Autonomous World Model — Client SDK
 *
 * TypeScript SDK for interacting with the onchain world model.
 * Provides session management, input handling, and state deserialization.
 *
 * Usage:
 *   import { SessionClient, InputManager, P1_KEYMAP } from "@awm/client";
 *
 *   const input = new InputManager(P1_KEYMAP);
 *   const stopInput = input.start();
 *
 *   const client = new SessionClient(wallet, {
 *     cluster: "https://devnet.magicblock.app",
 *     ephemeralWs: "wss://devnet.magicblock.app",
 *     modelManifest: new PublicKey("..."),
 *     stage: 31,
 *     character: 18,
 *   });
 *
 *   client.onFrame((frame) => {
 *     // frame matches viz/visualizer-juicy.html JSON format
 *     visualizer.renderFrame(frame);
 *   });
 *
 *   const sessionKey = await client.createSession();
 *   // Share sessionKey with player 2
 *   await client.startPlaying(() => input.getInput());
 *
 *   // When done:
 *   await client.endSession();
 *   stopInput();
 */

// State types and conversion
export {
  type PlayerState,
  type SessionState,
  type VizPlayerFrame,
  type VizFrame,
  SessionStatus,
  playerStateToViz,
  sessionToVizFrame,
  validateVizFrame,
} from "./state";

// Session management
export {
  type SessionConfig,
  type ModelInfo,
  SessionClient,
  listModels,
} from "./session";

// Input handling
export {
  type ControllerInput,
  InputManager,
  defaultInput,
  readGamepad,
  P1_KEYMAP,
  P2_KEYMAP,
  GCC_A,
  GCC_B,
  GCC_X,
  GCC_Y,
  GCC_Z,
  GCC_START,
} from "./input";
