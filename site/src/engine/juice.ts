/**
 * Juice state — trails, shake, hits, squash/stretch, combos, dynamic camera.
 * Ported from viz/visualizer-juicy.html L1322-1710.
 */

import { VizFrame, VizPlayerFrame } from './types';
import { STAGE, actionCategory, attackDisplayName } from './constants';
import { emitParticle, updateParticles, clearAllParticles } from './particles';

// Momentum trails
const TRAIL_LENGTH = 12;
export const playerTrails: Array<Array<{ x: number; y: number }>> = [[], []];

// Screen shake
export let shakeX = 0;
export let shakeY = 0;
let shakeMagnitude = 0;

// Hit freeze
export let hitFreezeFrames = 0;

// Impact flashes
export interface ImpactFlash {
  x: number; y: number; life: number; maxLife: number; ring?: boolean;
}
export const impactFlashes: ImpactFlash[] = [];

// Squash/stretch
export const squashState = [
  { scaleX: 1, scaleY: 1, timer: 0, landing: false },
  { scaleX: 1, scaleY: 1, timer: 0, landing: false },
];

// Dynamic camera
export let camCenterX = 0;
export let camCenterY = 30;
export let camZoom = 1;
let camTargetCenterX = 0;
let camTargetCenterY = 30;
let camTargetZoom = 1;

// Combo counter
export interface ComboState {
  count: number; timer: number; displayScale: number; x: number; y: number;
}
export const comboState: ComboState[] = [
  { count: 0, timer: 0, displayScale: 1, x: 0, y: 0 },
  { count: 0, timer: 0, displayScale: 1, x: 0, y: 0 },
];

// Hit labels
export interface HitLabel {
  text: string; x: number; y: number; life: number; maxLife: number;
}
export const hitLabels: HitLabel[] = [];

// Jump lines
export interface JumpLine {
  x: number; y: number; life: number; maxLife: number; offsets: number[];
}
export const jumpLines: JumpLine[] = [];

// Blast zone flash
export let blastFlashDir = '';
export let blastFlashLife = 0;

// Previous frame for transition detection
let prevFrame: VizFrame | null = null;

// Camera listener — parallax background subscribes to camera updates
type CameraListener = (cx: number, cy: number, zoom: number) => void;
let _cameraListener: CameraListener | null = null;

export function setCameraListener(cb: CameraListener | null) {
  _cameraListener = cb;
}

// Event callback — arena wires this up to push events to the store
export interface JuiceEvent {
  type: 'hit' | 'ko' | 'combo';
  attackerIdx: number;
  defenderIdx: number;
  attackName: string;
  damage?: number;
  newPercent?: number;
  comboCount?: number;
}
let _juiceEventCb: ((e: JuiceEvent) => void) | null = null;

export function setJuiceEventCallback(cb: ((e: JuiceEvent) => void) | null) {
  _juiceEventCb = cb;
}

// Coordinate conversion — set by the renderer each frame
let _gameToScreen: (gx: number, gy: number) => [number, number] = (gx, gy) => [gx, gy];

export function setGameToScreen(fn: (gx: number, gy: number) => [number, number]) {
  _gameToScreen = fn;
}

export function resetJuiceState() {
  playerTrails[0] = [];
  playerTrails[1] = [];
  shakeX = 0; shakeY = 0; shakeMagnitude = 0;
  hitFreezeFrames = 0;
  impactFlashes.length = 0;
  squashState[0] = { scaleX: 1, scaleY: 1, timer: 0, landing: false };
  squashState[1] = { scaleX: 1, scaleY: 1, timer: 0, landing: false };
  comboState[0] = { count: 0, timer: 0, displayScale: 1, x: 0, y: 0 };
  comboState[1] = { count: 0, timer: 0, displayScale: 1, x: 0, y: 0 };
  hitLabels.length = 0;
  jumpLines.length = 0;
  prevFrame = null;
  blastFlashDir = '';
  blastFlashLife = 0;
  clearAllParticles();
  camCenterX = 0; camCenterY = 30;
  camZoom = 1;
  camTargetCenterX = 0; camTargetCenterY = 30;
  camTargetZoom = 1;
}

export function detectTransitionsAndTriggerJuice(frame: VizFrame) {
  if (!prevFrame) { prevFrame = frame; return; }

  const prev = prevFrame;
  const cur = frame;

  for (let idx = 0; idx < 2; idx++) {
    const cp = cur.players[idx];
    const pp = prev.players[idx];
    const other = cur.players[1 - idx] as VizPlayerFrame;
    const otherPrev = prev.players[1 - idx] as VizPlayerFrame;

    // Hit detection
    const prevKB = Math.sqrt(pp.speed_attack_x ** 2 + pp.speed_attack_y ** 2);
    const curKB = Math.sqrt(cp.speed_attack_x ** 2 + cp.speed_attack_y ** 2);
    const hitlagStarted = cp.hitlag > 0 && pp.hitlag === 0;
    const kbHit = curKB > 0.5 && prevKB < 0.2;

    const pctDelta = cp.percent - pp.percent;
    const inDamage = cp.action_state >= 75 && cp.action_state <= 93;
    const wasNotDamage = pp.action_state < 75 || pp.action_state > 93;
    const pctHit = pctDelta > 0 && inDamage && wasNotDamage;

    if (kbHit || hitlagStarted || pctHit) {
      const impX = (cp.x + other.x) / 2;
      const impY = (cp.y + other.y) / 2;

      const kb = curKB > 0.5 ? curKB : Math.max(pctDelta * 0.3, 1);
      shakeMagnitude = Math.min(kb * 3, 15);

      impactFlashes.push({ x: impX, y: impY, life: 6, maxLife: 6 });

      const sparkCount = 8 + Math.floor(Math.random() * 5);
      const [impSX, impSY] = _gameToScreen(impX, impY);
      for (let s = 0; s < sparkCount; s++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 1 + Math.random() * 4;
        emitParticle(impSX, impSY, Math.cos(angle) * speed, Math.sin(angle) * speed,
          10 + Math.floor(Math.random() * 10), 1 + Math.random() * 2, 255, 240, 180, 0.9, 0.1);
      }

      const attackName = attackDisplayName(otherPrev.action_state);
      hitLabels.push({ text: attackName, x: impX, y: impY, life: 30, maxLife: 30 });

      const attackerIdx = 1 - idx;
      const combo = comboState[attackerIdx];
      combo.count++;
      combo.timer = 120;
      combo.displayScale = 1.5;
      combo.x = other.x;
      combo.y = other.y;

      if (_juiceEventCb) {
        _juiceEventCb({
          type: combo.count > 1 ? 'combo' : 'hit',
          attackerIdx,
          defenderIdx: idx,
          attackName,
          damage: pctDelta > 0 ? pctDelta : undefined,
          newPercent: cp.percent,
          comboCount: combo.count > 1 ? combo.count : undefined,
        });
      }
    }

    if (hitlagStarted || pctHit) {
      hitFreezeFrames = 3;
    }

    // Landing detection
    const landed = cp.on_ground && !pp.on_ground;
    if (landed) {
      const [footX, footY] = _gameToScreen(cp.x, STAGE.ground.y);
      const dustCount = 4 + Math.floor(Math.random() * 3);
      for (let d = 0; d < dustCount; d++) {
        const dir = (d % 2 === 0) ? 1 : -1;
        emitParticle(footX + (Math.random() - 0.5) * 6, footY,
          dir * (0.5 + Math.random() * 1.5), -(0.2 + Math.random() * 0.5),
          12 + Math.floor(Math.random() * 6), 1.5 + Math.random(), 160, 160, 170, 0.4, 0.02);
      }
      squashState[idx].landing = true;
      squashState[idx].timer = 4;
      squashState[idx].scaleX = 1.25;
      squashState[idx].scaleY = 0.7;
    }

    // Jump detection
    const jumpAction = cp.action_state >= 24 && cp.action_state <= 26;
    const jumped = !cp.on_ground && pp.on_ground && (cp.speed_y > 0 || jumpAction);
    if (jumped) {
      const lineCount = 2 + Math.floor(Math.random() * 2);
      const offsets: number[] = [];
      for (let l = 0; l < lineCount; l++) offsets.push((Math.random() - 0.5) * 10);
      jumpLines.push({ x: cp.x, y: STAGE.ground.y, life: 4, maxLife: 4, offsets });
    }

    // Death detection
    if (cp.stocks < pp.stocks) {
      const [deathSX, deathSY] = _gameToScreen(pp.x, pp.y);
      const burstCount = 30 + Math.floor(Math.random() * 11);
      for (let b = 0; b < burstCount; b++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 1 + Math.random() * 5;
        const isWhite = Math.random() > 0.4;
        emitParticle(deathSX, deathSY, Math.cos(angle) * speed, Math.sin(angle) * speed,
          15 + Math.floor(Math.random() * 20), 2 + Math.random() * 3,
          255, isWhite ? 255 : 220, isWhite ? 255 : 100, 1.0, 0.05);
      }
      impactFlashes.push({ x: pp.x, y: pp.y, life: 12, maxLife: 12, ring: true });
      shakeMagnitude = 12;

      if (pp.x < STAGE.blastzone.left + 10) blastFlashDir = 'left';
      else if (pp.x > STAGE.blastzone.right - 10) blastFlashDir = 'right';
      else if (pp.y > STAGE.blastzone.top - 10) blastFlashDir = 'top';
      else blastFlashDir = 'bottom';
      blastFlashLife = 10;

      if (_juiceEventCb) {
        _juiceEventCb({
          type: 'ko',
          attackerIdx: 1 - idx,
          defenderIdx: idx,
          attackName: 'KO',
          newPercent: pp.percent,
        });
      }
    }
  }

  prevFrame = frame;
}

export function updateJuice(frame: VizFrame) {
  // Decrement hit freeze counter
  if (hitFreezeFrames > 0) hitFreezeFrames--;

  updateParticles();

  // Screen shake decay
  if (shakeMagnitude > 0.1) {
    shakeX = (Math.random() - 0.5) * shakeMagnitude * 2;
    shakeY = (Math.random() - 0.5) * shakeMagnitude * 2;
    shakeMagnitude *= 0.7;
  } else {
    shakeX = 0; shakeY = 0; shakeMagnitude = 0;
  }

  // Impact flashes
  for (let i = impactFlashes.length - 1; i >= 0; i--) {
    impactFlashes[i].life--;
    if (impactFlashes[i].life <= 0) impactFlashes.splice(i, 1);
  }

  // Hit labels
  for (let i = hitLabels.length - 1; i >= 0; i--) {
    hitLabels[i].life--;
    if (hitLabels[i].life <= 0) hitLabels.splice(i, 1);
  }

  // Jump lines
  for (let i = jumpLines.length - 1; i >= 0; i--) {
    jumpLines[i].life--;
    if (jumpLines[i].life <= 0) jumpLines.splice(i, 1);
  }

  // Blast zone flash
  if (blastFlashLife > 0) blastFlashLife--;

  // Squash/stretch
  for (let idx = 0; idx < 2; idx++) {
    const ss = squashState[idx];
    if (ss.timer > 0) {
      ss.timer--;
      if (ss.timer <= 0) { ss.scaleX = 1; ss.scaleY = 1; ss.landing = false; }
    }
    if (!ss.landing && frame.players[idx]) {
      const p = frame.players[idx];
      const vx = p.on_ground ? p.speed_ground_x : p.speed_air_x;
      const vy = p.speed_y;
      const speed = Math.sqrt(vx * vx + vy * vy);
      if (speed > 1.0) {
        const stretchFactor = Math.min(speed / 6, 0.3);
        ss.scaleX = 1 - stretchFactor * 0.3;
        ss.scaleY = 1 + stretchFactor;
      } else if (ss.timer <= 0) {
        ss.scaleX = 1; ss.scaleY = 1;
      }
    }
  }

  // Combo counters
  for (let idx = 0; idx < 2; idx++) {
    const combo = comboState[idx];
    if (combo.timer > 0) {
      combo.timer--;
      combo.displayScale += (1 - combo.displayScale) * 0.2;
      if (combo.timer <= 0) combo.count = 0;
    }
  }

  // Momentum trails
  if (frame.players) {
    for (let idx = 0; idx < 2; idx++) {
      const p = frame.players[idx];
      playerTrails[idx].push({ x: p.x, y: p.y });
      if (playerTrails[idx].length > TRAIL_LENGTH) playerTrails[idx].shift();
    }
  }

  // Dash dust
  if (frame.players) {
    for (let idx = 0; idx < 2; idx++) {
      const p = frame.players[idx];
      if (p.on_ground && Math.abs(p.speed_ground_x) > 1.5 && Math.random() < 0.35) {
        const [footX, footY] = _gameToScreen(p.x, STAGE.ground.y);
        emitParticle(footX + (Math.random() - 0.5) * 4, footY,
          -p.speed_ground_x * 0.2 + (Math.random() - 0.5) * 0.5, -(0.1 + Math.random() * 0.3),
          8 + Math.floor(Math.random() * 5), 1 + Math.random() * 0.8, 140, 140, 150, 0.25, 0.01);
      }
    }
  }

  // Dynamic camera
  if (frame.players) {
    const p0 = frame.players[0];
    const p1 = frame.players[1];
    camTargetCenterX = (p0.x + p1.x) / 2;
    camTargetCenterY = (p0.y + p1.y) / 2 + 20;

    const dx = Math.abs(p0.x - p1.x);
    const dy = Math.abs(p0.y - p1.y);
    const stageW = STAGE.camera.right - STAGE.camera.left;
    const needed = Math.max(dx + 80, dy + 60, 120);
    camTargetZoom = Math.min(2.0, Math.max(1.0, stageW / needed * 0.75));

    camCenterX += (camTargetCenterX - camCenterX) * 0.08;
    camCenterY += (camTargetCenterY - camCenterY) * 0.08;
    camZoom += (camTargetZoom - camZoom) * 0.06;
  }

  if (_cameraListener) _cameraListener(camCenterX, camCenterY, camZoom);
}

export function setCameraImmediate(frame: VizFrame) {
  const p0 = frame.players[0];
  const p1 = frame.players[1];
  camCenterX = (p0.x + p1.x) / 2;
  camCenterY = (p0.y + p1.y) / 2 + 20;
  camTargetCenterX = camCenterX;
  camTargetCenterY = camCenterY;
  const dx = Math.abs(p0.x - p1.x);
  const dy = Math.abs(p0.y - p1.y);
  const stageW = STAGE.camera.right - STAGE.camera.left;
  const needed = Math.max(dx + 80, dy + 60, 120);
  camZoom = Math.min(2.0, Math.max(1.0, stageW / needed * 0.75));
  camTargetZoom = camZoom;

  if (_cameraListener) _cameraListener(camCenterX, camCenterY, camZoom);
}
