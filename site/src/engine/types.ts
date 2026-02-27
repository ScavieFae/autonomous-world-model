/**
 * Types for the autonomous world model visualizer.
 * Copied from solana/client/src/state.ts — kept in sync manually.
 */

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

export type RenderMode = 'wire' | 'capsule' | 'data' | 'xray';

export const CHARACTER_FILL_MODES = [
  'stroke', 'scanline', 'hatch', 'grid', 'ghost', 'gradient', 'stipple', 'hologram',
] as const;
export type CharacterFillMode = (typeof CHARACTER_FILL_MODES)[number];

export interface PlayerColors {
  main: string;
  glow: string;
  dim: string;
  r: number;
  g: number;
  b: number;
}

export const PLAYER_COLORS: [PlayerColors, PlayerColors] = [
  { main: '#22c55e', glow: 'rgba(34,197,94,0.35)', dim: 'rgba(34,197,94,0.15)', r: 34, g: 197, b: 94 },
  { main: '#f59e0b', glow: 'rgba(245,158,11,0.35)', dim: 'rgba(245,158,11,0.15)', r: 245, g: 158, b: 11 },
];

/** Shared engine interface — both PlaybackEngine and LiveEngine implement this. */
export interface Engine {
  frames: VizFrame[];
  currentFrame: number;
  playing: boolean;
  playSpeed: number;
  speedIdx: number;
  readonly totalFrames: number;
  readonly currentVizFrame: VizFrame | null;
  setOnFrame(cb: (frame: VizFrame, index: number) => void): void;
  loadFrames(frames: VizFrame[]): void;
  play(): void;
  pause(): void;
  togglePlay(): void;
  stepFrame(delta: number): void;
  seekTo(frameIdx: number): void;
  cycleSpeed(direction?: 1 | -1): void;
  destroy(): void;
}
