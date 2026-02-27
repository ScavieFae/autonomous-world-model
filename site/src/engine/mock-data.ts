/**
 * Mock data generators — neutral game, combo, edgeguard scenarios.
 * Ported from viz/visualizer-juicy.html L800-1215.
 */

import { VizFrame, VizPlayerFrame } from './types';
import { STAGE } from './constants';

function makePlayer(x: number, y: number, char: number, facing: number): VizPlayerFrame {
  return {
    x, y, percent: 0, shield_strength: 60,
    speed_air_x: 0, speed_y: 0, speed_ground_x: 0,
    speed_attack_x: 0, speed_attack_y: 0,
    state_age: 0, hitlag: 0, stocks: 4,
    facing, on_ground: y <= 1 ? 1 : 0,
    action_state: 14, jumps_left: 2, character: char,
  };
}

function clonePlayer(p: VizPlayerFrame): VizPlayerFrame { return { ...p }; }

function applyGravity(p: VizPlayerFrame) {
  if (!p.on_ground) {
    p.speed_y -= 0.13;
    if (p.speed_y < -3.5) p.speed_y = -3.5;
  }
}

function applyMovement(p: VizPlayerFrame) {
  if (p.on_ground) {
    p.x += p.speed_ground_x;
    p.speed_ground_x *= 0.9;
  } else {
    p.x += p.speed_air_x;
    p.y += p.speed_y;
  }
  p.x += p.speed_attack_x;
  p.y += p.speed_attack_y;
  p.speed_attack_x *= 0.95;
  p.speed_attack_y *= 0.95;
  if (Math.abs(p.speed_attack_x) < 0.05) p.speed_attack_x = 0;
  if (Math.abs(p.speed_attack_y) < 0.05) p.speed_attack_y = 0;
}

function applyStageCollision(p: VizPlayerFrame) {
  if (p.y <= STAGE.ground.y && p.x >= STAGE.ground.x1 && p.x <= STAGE.ground.x2) {
    if (!p.on_ground) {
      p.y = STAGE.ground.y;
      p.on_ground = 1;
      p.speed_y = 0;
      p.speed_air_x = 0;
      p.jumps_left = p.character === 4 ? 6 : 2;
    }
  }
  if (p.on_ground && (p.x < STAGE.ground.x1 || p.x > STAGE.ground.x2)) {
    p.on_ground = 0;
  }
}

function syncFrame(f: VizPlayerFrame, p: VizPlayerFrame) {
  f.x = p.x; f.y = p.y;
  f.speed_air_x = p.speed_air_x; f.speed_y = p.speed_y;
  f.speed_ground_x = p.speed_ground_x;
  f.speed_attack_x = p.speed_attack_x; f.speed_attack_y = p.speed_attack_y;
  f.on_ground = p.on_ground; f.percent = p.percent;
  f.shield_strength = p.shield_strength;
  f.jumps_left = p.jumps_left;
  f.stocks = p.stocks;
  f.hitlag = Math.max(0, (f.hitlag || 0) - 1);
}

function computeStateAge(frames: VizFrame[], playerIdx: number, curAction: number): number {
  if (frames.length === 0) return 0;
  const prev = frames[frames.length - 1].players[playerIdx];
  return prev.action_state === curAction ? prev.state_age + 1 : 0;
}

export function generateNeutral(): VizFrame[] {
  const frames: VizFrame[] = [];
  const p0 = makePlayer(-30, 0, 18, 1);
  const p1 = makePlayer(30, 0, 1, 0);
  const N = 360;

  for (let i = 0; i < N; i++) {
    const f0 = clonePlayer(p0);
    const f1 = clonePlayer(p1);
    const phase = Math.floor(i / 60);
    const t = (i % 60) / 60;

    if (phase === 0) {
      f0.action_state = 14; f1.action_state = 14;
      p0.speed_ground_x = Math.sin(i * 0.05) * 0.3;
      p1.speed_ground_x = -Math.sin(i * 0.07) * 0.2;
    } else if (phase === 1) {
      if (t < 0.3) { f0.action_state = 20; p0.speed_ground_x = 2.0; }
      else if (t < 0.5) {
        f0.action_state = 44; p0.speed_ground_x *= 0.3;
        if (t > 0.4 && t < 0.5 && Math.abs(p0.x - p1.x) < 20) {
          f1.action_state = 88;
          p1.speed_attack_x = 2.5; p1.speed_attack_y = 1.5;
          p1.on_ground = 0; p1.percent += 12;
          f0.hitlag = 4; f1.hitlag = 4;
        }
      } else { f0.action_state = 14; p0.speed_ground_x *= 0.8; }
      if (f1.action_state >= 75 && f1.action_state <= 93) {
        f1.state_age = Math.floor((t - 0.4) * 60);
      } else if (!p1.on_ground && p1.y > 0) { f1.action_state = 29; }
      else { f1.action_state = 14; }
    } else if (phase === 2) {
      if (t < 0.15) { f1.action_state = 24; f1.facing = 0; }
      else if (t < 0.2) {
        f1.action_state = 25; p1.speed_y = 3.2; p1.speed_air_x = -1.5;
        p1.on_ground = 0; p1.jumps_left = 1;
      } else if (t < 0.45) { f1.action_state = 57; f1.on_ground = 0; }
      else if (t < 0.6) { f1.action_state = 29; f1.on_ground = 0; }
      else { f1.action_state = 233; if (t > 0.7) f1.action_state = 14; }
      if (t > 0.3 && t < 0.6) {
        f0.action_state = 179; p0.shield_strength = Math.max(30, p0.shield_strength - 0.3);
      } else { f0.action_state = 14; }
    } else if (phase === 3) {
      const dd0 = Math.sin(i * 0.15) > 0;
      const dd1 = Math.sin(i * 0.12 + 1) > 0;
      f0.action_state = 20; f1.action_state = 20;
      f0.facing = dd0 ? 1 : 0; f1.facing = dd1 ? 1 : 0;
      p0.speed_ground_x = dd0 ? 1.8 : -1.8;
      p1.speed_ground_x = dd1 ? 1.6 : -1.6;
    } else if (phase === 4) {
      if (t < 0.2) { f0.action_state = 20; p0.speed_ground_x = 2.0; }
      else if (t < 0.35) { f0.action_state = 212; p0.speed_ground_x = 0; }
      else if (t < 0.7) {
        f0.action_state = 214; f1.action_state = 224;
        p0.speed_ground_x = 0; p1.x = p0.x + (f0.facing ? 8 : -8);
      } else {
        f0.action_state = 14; f1.action_state = 76;
        p1.speed_attack_x = f0.facing ? 3 : -3; p1.speed_attack_y = 2;
        p1.on_ground = 0; p1.percent += 8;
        f0.hitlag = 3; f1.hitlag = 3;
      }
    } else {
      f0.action_state = 14; f1.action_state = 14;
      p0.speed_ground_x += (-30 - p0.x) * 0.02;
      p1.speed_ground_x += (30 - p1.x) * 0.02;
    }

    f0.state_age = computeStateAge(frames, 0, f0.action_state);
    f1.state_age = computeStateAge(frames, 1, f1.action_state);

    applyGravity(p0); applyGravity(p1);
    applyMovement(p0); applyMovement(p1);
    applyStageCollision(p0); applyStageCollision(p1);

    syncFrame(f0, p0); syncFrame(f1, p1);
    frames.push({ players: [f0, f1], stage: 31 });
  }
  return frames;
}

export function generateCombo(): VizFrame[] {
  const frames: VizFrame[] = [];
  const p0 = makePlayer(-20, 0, 2, 1);
  const p1 = makePlayer(20, 0, 22, 0);
  const N = 300;

  interface ScriptPhase {
    dur: number; p0a: number; p1a?: number; p0vx?: number;
    setup?: (i: number, f0: VizPlayerFrame, f1: VizPlayerFrame, pp0: VizPlayerFrame, pp1: VizPlayerFrame) => void;
  }

  const script: ScriptPhase[] = [
    { dur: 30, p0a: 20, p1a: 14, p0vx: 2.2, setup: (i, f0) => {
      if (i > 20) { f0.action_state = 212; p0.speed_ground_x = 0; }
    }},
    { dur: 30, p0a: 214, p1a: 224, setup: (_i, _f0, _f1, pp0, pp1) => { pp1.x = pp0.x + 8; }},
    { dur: 15, p0a: 14, p1a: 78, setup: (i, f0, f1, _pp0, pp1) => {
      if (i === 0) {
        pp1.speed_attack_y = 4; pp1.speed_attack_x = 0.3;
        pp1.on_ground = 0; pp1.percent += 7;
        f0.hitlag = 3; f1.hitlag = 3;
      }
    }},
    { dur: 20, p0a: 24, setup: (i, f0, f1, pp0, pp1) => {
      if (i === 5) { pp0.speed_y = 4.5; pp0.speed_air_x = 1.2; pp0.on_ground = 0; pp0.jumps_left = 1; f0.action_state = 27; }
      if (i > 5) f0.action_state = 27;
      f1.action_state = pp1.speed_attack_y > 0.3 ? 78 : 29;
    }},
    { dur: 20, p0a: 56, setup: (i, f0, f1, pp0, pp1) => {
      f0.on_ground = 0;
      if (i === 8 && Math.abs(pp0.y - pp1.y) < 25) {
        f1.action_state = 76; pp1.speed_attack_x = 1.5; pp1.speed_attack_y = 2;
        pp1.percent += 14; f0.hitlag = 4; f1.hitlag = 4;
      } else { f1.action_state = pp1.speed_attack_y > 0.3 ? 76 : 29; }
    }},
    { dur: 25, p0a: 29, setup: (i, f0, f1, pp0, pp1) => {
      f0.on_ground = 0;
      f1.action_state = pp1.speed_attack_y > 0.3 ? 76 : 29; f1.on_ground = 0;
      if (i === 10 && pp0.jumps_left > 0) { pp0.speed_y = 3.8; pp0.jumps_left = 0; f0.action_state = 28; }
    }},
    { dur: 20, p0a: 59, setup: (i, f0, f1, _pp0, pp1) => {
      f0.on_ground = 0;
      if (i === 6) {
        f1.action_state = 75; pp1.speed_attack_y = 3.5; pp1.speed_attack_x = 0.5;
        pp1.percent += 13; f0.hitlag = 5; f1.hitlag = 5;
      } else { f1.action_state = pp1.speed_attack_y > 0.3 ? 75 : 29; }
      f1.on_ground = 0;
    }},
    { dur: 60, p0a: 29, setup: (i, f0, f1, pp0, pp1) => {
      f1.action_state = pp1.speed_attack_y > 0.3 ? 75 : 29;
      f0.on_ground = pp0.on_ground; f1.on_ground = pp1.on_ground;
      if (pp0.on_ground) f0.action_state = 233;
      if (pp1.on_ground) f1.action_state = 233;
      if (pp0.on_ground && i > 10) f0.action_state = 14;
      if (pp1.on_ground && i > 15) f1.action_state = 14;
    }},
  ];

  let phaseIdx = 0, phaseFrame = 0;

  for (let i = 0; i < N; i++) {
    const f0 = clonePlayer(p0);
    const f1 = clonePlayer(p1);

    if (phaseIdx < script.length) {
      const phase = script[phaseIdx];
      f0.action_state = phase.p0a || f0.action_state;
      if (phase.p1a !== undefined) f1.action_state = phase.p1a;
      if (phase.p0vx !== undefined && phaseFrame === 0) p0.speed_ground_x = phase.p0vx;
      if (phase.setup) phase.setup(phaseFrame, f0, f1, p0, p1);
      phaseFrame++;
      if (phaseFrame >= phase.dur) { phaseIdx++; phaseFrame = 0; }
    } else {
      f0.action_state = p0.on_ground ? 14 : 29;
      f1.action_state = p1.on_ground ? 14 : 29;
      p0.speed_ground_x += (-20 - p0.x) * 0.02;
      p1.speed_ground_x += (20 - p1.x) * 0.02;
    }

    f0.state_age = computeStateAge(frames, 0, f0.action_state);
    f1.state_age = computeStateAge(frames, 1, f1.action_state);

    applyGravity(p0); applyGravity(p1);
    applyMovement(p0); applyMovement(p1);
    applyStageCollision(p0); applyStageCollision(p1);

    syncFrame(f0, p0); syncFrame(f1, p1);
    frames.push({ players: [f0, f1], stage: 31 });
  }
  return frames;
}

export function generateEdgeguard(): VizFrame[] {
  const frames: VizFrame[] = [];
  const p0 = makePlayer(40, 0, 18, 0);
  const p1 = makePlayer(-120, -30, 1, 1);
  p1.on_ground = 0; p1.percent = 80; p1.jumps_left = 0;
  p1.speed_air_x = 2.5; p1.speed_y = 1.5;
  const N = 300;

  for (let i = 0; i < N; i++) {
    const f0 = clonePlayer(p0);
    const f1 = clonePlayer(p1);
    const t = i / 60;

    if (t < 1.5) {
      f1.action_state = 35;
      if (t < 0.5) { p1.speed_air_x = 2.8; p1.speed_y = 2.0; }
      else if (t < 1.0) { p1.speed_air_x = 1.5; p1.speed_y = 0.5; }
      f0.action_state = 14;
      if (t > 0.8 && t < 1.2) { f0.action_state = 20; p0.speed_ground_x = -1.5; }
      if (t > 1.2) {
        f0.action_state = 57;
        if (t > 1.0) { p0.on_ground = 0; p0.speed_y = 1; p0.speed_air_x = -1; }
        if (t > 1.3 && t < 1.4 && Math.abs(p0.x - p1.x) < 25) {
          f1.action_state = 77;
          p1.speed_attack_x = -3; p1.speed_attack_y = -2;
          p1.percent += 14; f0.hitlag = 6; f1.hitlag = 6;
        }
      }
    } else if (t < 3.0) {
      f1.action_state = (p1.speed_attack_y < -0.2 || p1.speed_attack_x < -0.2) ? 79 : 29;
      f1.on_ground = 0;
      f0.action_state = p0.on_ground ? 14 : 29;
      if (!p0.on_ground) { p0.speed_air_x += (40 - p0.x) * 0.01; }
      else { p0.speed_ground_x += (40 - p0.x) * 0.03; f0.action_state = 14; }
    } else {
      if (p1.y < STAGE.blastzone.bottom || p1.x < STAGE.blastzone.left) {
        p1.stocks = Math.max(0, p1.stocks - 1);
        p1.x = 0; p1.y = 50; p1.on_ground = 0;
        p1.speed_air_x = 0; p1.speed_y = 0;
        p1.speed_attack_x = 0; p1.speed_attack_y = 0;
        p1.percent = 0; p1.jumps_left = 2;
        f1.action_state = 29;
      }
      f0.action_state = p0.on_ground ? 14 : 29;
      f1.action_state = p1.on_ground ? 14 : 29;
      p0.speed_ground_x += (40 - p0.x) * 0.02;
      p1.speed_ground_x += (-20 - p1.x) * 0.01;
    }

    f0.state_age = computeStateAge(frames, 0, f0.action_state);
    f1.state_age = computeStateAge(frames, 1, f1.action_state);

    applyGravity(p0); applyGravity(p1);
    applyMovement(p0); applyMovement(p1);
    applyStageCollision(p0); applyStageCollision(p1);

    syncFrame(f0, p0); syncFrame(f1, p1);
    frames.push({ players: [f0, f1], stage: 31 });
  }
  return frames;
}

export type MockScenario = 'neutral' | 'combo' | 'edgeguard';

export function generateMockData(scenario: MockScenario): VizFrame[] {
  const gen = { neutral: generateNeutral, combo: generateCombo, edgeguard: generateEdgeguard };
  return (gen[scenario] || gen.neutral)();
}
