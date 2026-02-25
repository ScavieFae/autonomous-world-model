/**
 * State types and deserialization for the autonomous world model.
 *
 * Converts onchain SessionState to the JSON format consumed by
 * viz/visualizer-juicy.html.
 */

// ── Onchain types (matching Rust component structs) ─────────────────────────

export interface PlayerState {
  x: number; // Fixed-point: actual = x / 256.0
  y: number;
  percent: number;
  shieldStrength: number;
  speedAirX: number;
  speedY: number;
  speedGroundX: number;
  speedAttackX: number;
  speedAttackY: number;
  stateAge: number;
  hitlag: number;
  stocks: number;
  facing: number;
  onGround: number;
  actionState: number;
  jumpsLeft: number;
  character: number;
}

export interface SessionState {
  status: number;
  frame: number;
  maxFrames: number;
  player1: string; // Pubkey as base58
  player2: string;
  stage: number;
  players: [PlayerState, PlayerState];
  model: string;
  createdAt: number;
  lastUpdate: number;
  seed: number;
}

export const SessionStatus = {
  Created: 0,
  WaitingPlayers: 1,
  Active: 2,
  Ended: 3,
} as const;

// ── Visualizer JSON format ──────────────────────────────────────────────────

export interface VizPlayerFrame {
  x: number;
  y: number;
  percent: number;
  shield_strength: number;
  speed_air_x: number;
  speed_y: number;
  speed_ground_x: number;
  speed_attack_x: number;
  speed_attack_y: number;
  state_age: number;
  hitlag: number;
  stocks: number;
  facing: number;
  on_ground: number;
  action_state: number;
  jumps_left: number;
  character: number;
}

export interface VizFrame {
  players: [VizPlayerFrame, VizPlayerFrame];
  stage: number;
}

// ── Conversion ──────────────────────────────────────────────────────────────

/**
 * Convert onchain PlayerState to visualizer JSON format.
 * Handles fixed-point → float conversion and field name mapping.
 */
export function playerStateToViz(p: PlayerState): VizPlayerFrame {
  return {
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
  };
}

/**
 * Convert onchain SessionState to a single visualizer frame.
 */
export function sessionToVizFrame(session: SessionState): VizFrame {
  return {
    players: [
      playerStateToViz(session.players[0]),
      playerStateToViz(session.players[1]),
    ],
    stage: session.stage,
  };
}

/**
 * Validate a VizFrame has all required fields with correct types.
 */
export function validateVizFrame(frame: VizFrame): string[] {
  const errors: string[] = [];

  if (!frame.players || frame.players.length !== 2) {
    errors.push("Frame must have exactly 2 players");
    return errors;
  }

  const requiredFields: (keyof VizPlayerFrame)[] = [
    "x", "y", "percent", "shield_strength", "speed_air_x", "speed_y",
    "speed_ground_x", "speed_attack_x", "speed_attack_y", "state_age",
    "hitlag", "stocks", "facing", "on_ground", "action_state",
    "jumps_left", "character",
  ];

  for (let i = 0; i < 2; i++) {
    const p = frame.players[i];
    for (const field of requiredFields) {
      if (typeof p[field] !== "number") {
        errors.push(`Player ${i + 1}.${field} must be a number, got ${typeof p[field]}`);
      }
    }
  }

  if (typeof frame.stage !== "number") {
    errors.push(`stage must be a number, got ${typeof frame.stage}`);
  }

  return errors;
}
