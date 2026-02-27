/**
 * Wireframe fighter renderer for "The Wire" arena.
 * Draws a ~15-segment humanoid skeleton-mesh per player with per-action-state
 * poses, wireframe mesh topology, geometric face, hit scatter, and KO
 * disintegration.
 */

import type { VizPlayerFrame, PlayerColors } from '../types';
import type { ActionCategory } from '../constants';
import { actionCategory } from '../constants';
import { emitParticle } from '../particles';

// ---------------------------------------------------------------------------
// Joint definitions — offsets from player position (feet) in game units
// ---------------------------------------------------------------------------

interface Vec2 { x: number; y: number }

interface Skeleton {
  head: Vec2; neck: Vec2; chest: Vec2; hip: Vec2;
  l_shoulder: Vec2; r_shoulder: Vec2;
  l_elbow: Vec2; r_elbow: Vec2;
  l_hand: Vec2; r_hand: Vec2;
  l_hip: Vec2; r_hip: Vec2;
  l_knee: Vec2; r_knee: Vec2;
  l_foot: Vec2; r_foot: Vec2;
}

const BASE_SKELETON: Skeleton = {
  head:       { x:  0,    y: 10.5 },
  neck:       { x:  0,    y:  9.5 },
  chest:      { x:  0,    y:  8   },
  hip:        { x:  0,    y:  5.5 },
  l_shoulder: { x: -2,    y:  9   },
  r_shoulder: { x:  2,    y:  9   },
  l_elbow:    { x: -3.5,  y:  7.5 },
  r_elbow:    { x:  3.5,  y:  7.5 },
  l_hand:     { x: -4.5,  y:  6   },
  r_hand:     { x:  4.5,  y:  6   },
  l_hip:      { x: -1.2,  y:  5.5 },
  r_hip:      { x:  1.2,  y:  5.5 },
  l_knee:     { x: -1.5,  y:  3   },
  r_knee:     { x:  1.5,  y:  3   },
  l_foot:     { x: -1.8,  y:  0   },
  r_foot:     { x:  1.8,  y:  0   },
};

const JOINT_KEYS = Object.keys(BASE_SKELETON) as (keyof Skeleton)[];

// ---------------------------------------------------------------------------
// Module-level state for hit/KO detection
// ---------------------------------------------------------------------------

const prevHitlag: [number, number] = [0, 0];
const prevCategory: [ActionCategory | null, ActionCategory | null] = [null, null];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cloneSkeleton(): Skeleton {
  const s = {} as Record<string, Vec2>;
  for (const k of JOINT_KEYS) {
    s[k] = { x: BASE_SKELETON[k].x, y: BASE_SKELETON[k].y };
  }
  return s as unknown as Skeleton;
}

function lerpColor(a: string, b: string, t: number): string {
  const ar = parseInt(a.slice(1, 3), 16);
  const ag = parseInt(a.slice(3, 5), 16);
  const ab = parseInt(a.slice(5, 7), 16);
  const br = parseInt(b.slice(1, 3), 16);
  const bg = parseInt(b.slice(3, 5), 16);
  const bb = parseInt(b.slice(5, 7), 16);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bv = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${bv})`;
}

/** One-shot easing: 0→1 over `duration` frames. */
function ease(stateAge: number, duration: number): number {
  return Math.min(1, stateAge / duration);
}

/** Three-phase attack swing: wind-up → strike → settle.
 *  Returns roughly -0.2 → 1.1 → 1.0 over windup+strike+settle frames. */
function swing(age: number, windup = 2, strike = 4, settle = 4): number {
  if (age < windup) return -0.2 * (age / windup);
  const t1 = age - windup;
  if (t1 < strike) return -0.2 + 1.3 * Math.min(1, t1 / strike);
  const t2 = t1 - strike;
  return 1.1 - 0.1 * Math.min(1, t2 / settle);
}

// ---------------------------------------------------------------------------
// Pose mapping — mutate skeleton based on action state + category
// ---------------------------------------------------------------------------

function applyPose(
  sk: Skeleton,
  p: VizPlayerFrame,
  category: ActionCategory,
  facingDir: number,
): void {
  const age = p.state_age;
  const as = p.action_state;

  switch (category) {
    case 'neutral': {
      // Breathing bounce + subtle arm sway
      const bounce = Math.sin(age * 0.1) * 0.3;
      const armSway = Math.sin(age * 0.07) * 0.4;
      for (const k of JOINT_KEYS) sk[k].y += bounce;
      sk.l_hand.x += armSway;
      sk.r_hand.x -= armSway;
      sk.l_elbow.x += armSway * 0.5;
      sk.r_elbow.x -= armSway * 0.5;
      break;
    }

    case 'movement': {
      if (as === 20) {
        // Dash — big forward lunge
        const lean = 3 * facingDir;
        sk.head.x += lean;
        sk.neck.x += lean * 0.9;
        sk.chest.x += lean * 0.7;
        sk.l_shoulder.x += lean * 0.8;
        sk.r_shoulder.x += lean * 0.8;
        // Back arm swings behind
        sk.l_hand.x -= facingDir * 2;
        sk.l_hand.y += 1;
        sk.r_hand.x += facingDir * 3;
        sk.r_hand.y -= 0.5;
        // Front leg extended
        if (facingDir > 0) {
          sk.r_foot.x += 3; sk.r_knee.x += 1.5;
          sk.l_foot.x -= 0.5; sk.l_knee.y += 1;
        } else {
          sk.l_foot.x -= 3; sk.l_knee.x -= 1.5;
          sk.r_foot.x += 0.5; sk.r_knee.y += 1;
        }
      } else if (as === 21 || as === 22) {
        // Run cycle — alternating legs + arm pump
        const cycle = Math.sin(age * 0.4);
        const lean = 2.5 * facingDir;
        sk.head.x += lean * 0.8;
        sk.neck.x += lean * 0.7;
        sk.chest.x += lean * 0.5;
        sk.l_shoulder.x += lean * 0.6;
        sk.r_shoulder.x += lean * 0.6;
        // Legs alternate
        sk.l_foot.x += cycle * 2;
        sk.l_foot.y += Math.abs(cycle) * 1.5;
        sk.l_knee.x += cycle * 0.8;
        sk.l_knee.y += Math.abs(cycle) * 0.8;
        sk.r_foot.x -= cycle * 2;
        sk.r_foot.y += Math.abs(-cycle) * 1.5;
        sk.r_knee.x -= cycle * 0.8;
        sk.r_knee.y += Math.abs(-cycle) * 0.8;
        // Arms pump opposite to legs
        sk.l_hand.x -= cycle * 1.5 * facingDir;
        sk.l_hand.y += cycle * 0.8;
        sk.r_hand.x += cycle * 1.5 * facingDir;
        sk.r_hand.y -= cycle * 0.8;
      } else if (as === 23) {
        // RunBrake — legs forward, lean back, arms out for balance
        sk.head.x -= facingDir * 1.5;
        sk.neck.x -= facingDir * 1;
        sk.chest.x -= facingDir * 0.5;
        sk.l_hand.x -= 2; sk.l_hand.y += 1.5;
        sk.r_hand.x += 2; sk.r_hand.y += 1.5;
        sk.l_foot.x += facingDir * 1.5;
        sk.r_foot.x += facingDir * 1.5;
        sk.l_knee.x += facingDir * 0.8;
        sk.r_knee.x += facingDir * 0.8;
      } else if (as >= 39 && as <= 41) {
        // Crouch — compressed, arms out, wide stance
        for (const k of JOINT_KEYS) sk[k].y *= 0.65;
        sk.l_hand.x -= 1.5; sk.l_hand.y += 0.5;
        sk.r_hand.x += 1.5; sk.r_hand.y += 0.5;
        sk.l_foot.x -= 1; sk.r_foot.x += 1;
      } else if (as >= 15 && as <= 17) {
        // Walk — alternating leg cycle, lean proportional to walk speed
        const cycle = Math.sin(age * 0.3);
        const speed = as === 15 ? 0.6 : as === 16 ? 0.8 : 1;
        const lean = 1.5 * facingDir * speed;
        sk.head.x += lean * 0.6;
        sk.neck.x += lean * 0.5;
        sk.chest.x += lean * 0.3;
        sk.l_shoulder.x += lean * 0.4;
        sk.r_shoulder.x += lean * 0.4;
        // Legs alternate
        sk.l_foot.x += cycle * 1.5 * speed;
        sk.l_foot.y += Math.abs(cycle) * 0.8 * speed;
        sk.l_knee.x += cycle * 0.5 * speed;
        sk.r_foot.x -= cycle * 1.5 * speed;
        sk.r_foot.y += Math.abs(-cycle) * 0.8 * speed;
        sk.r_knee.x -= cycle * 0.5 * speed;
        // Gentle arm sway
        sk.l_hand.x -= cycle * 0.8 * speed;
        sk.r_hand.x += cycle * 0.8 * speed;
      } else {
        // Fallback movement lean
        const lean = 2 * facingDir;
        sk.head.x += lean;
        sk.neck.x += lean;
        sk.chest.x += lean;
        sk.l_shoulder.x += lean;
        sk.r_shoulder.x += lean;
        sk.l_elbow.x += lean * 0.7;
        sk.r_elbow.x += lean * 0.7;
        sk.l_hand.x += lean * 0.5;
        sk.r_hand.x += lean * 0.5;
        sk.l_foot.x -= 1; sk.r_foot.x += 1;
        sk.l_knee.x -= 0.5; sk.r_knee.x += 0.5;
      }
      break;
    }

    case 'aerial': {
      if (as === 24) {
        // KneeBend — pre-jump squat, compressed, legs coiled
        for (const k of JOINT_KEYS) sk[k].y *= 0.75;
        sk.l_knee.x -= 0.8; sk.r_knee.x += 0.8;
        sk.l_foot.x -= 0.5; sk.r_foot.x += 0.5;
      } else if (as === 25 || as === 26) {
        // Jump — upward extension, arms rise, legs tuck
        const t = ease(age, 10);
        for (const k of JOINT_KEYS) sk[k].y += t * 1;
        sk.l_hand.y += 2 + t * 1;
        sk.r_hand.y += 2 + t * 1;
        sk.l_elbow.y += 1 + t * 0.5;
        sk.r_elbow.y += 1 + t * 0.5;
        sk.l_foot.y += 2 + t * 1;
        sk.r_foot.y += 2 + t * 1;
        sk.l_knee.y += 1;
        sk.r_knee.y += 1;
        sk.l_knee.x -= 0.5;
        sk.r_knee.x += 0.5;
      } else if (as >= 35 && as <= 37) {
        // FallSpecial — limp: arms dangle, legs loose
        sk.l_hand.y -= 2;
        sk.r_hand.y -= 2;
        sk.l_elbow.y -= 1;
        sk.r_elbow.y -= 1;
        sk.l_hand.x -= 1;
        sk.r_hand.x += 1;
        sk.l_foot.y += 1;
        sk.r_foot.y += 1;
        sk.l_foot.x -= 0.5;
        sk.r_foot.x += 0.5;
        // Slight sway
        const sway = Math.sin(age * 0.15) * 0.5;
        sk.l_hand.x += sway;
        sk.r_hand.x += sway;
      } else {
        // Fall (29-34) and double jumps (27-28) — spread limbs
        sk.l_hand.x -= 2;
        sk.r_hand.x += 2;
        sk.l_hand.y += 1;
        sk.r_hand.y += 1;
        sk.l_elbow.x -= 1;
        sk.r_elbow.x += 1;
        sk.l_elbow.y += 0.5;
        sk.r_elbow.y += 0.5;
        sk.l_foot.x -= 0.8;
        sk.r_foot.x += 0.8;
        sk.l_foot.y += 1;
        sk.r_foot.y += 1;
      }
      break;
    }

    case 'attack': {
      if (as >= 42 && as <= 46) {
        // Ftilt — snappy forward strike
        const t = swing(age, 2, 3, 4);
        const at = Math.max(0, t); // clamp for position (wind-up uses body lean only)
        // Striking arm extends forward
        sk.l_hand.x += facingDir * 4.5 * at;
        sk.l_hand.y += 1.5 * at;
        sk.r_hand.x += facingDir * 4.5 * at;
        sk.r_hand.y += 1.5 * at;
        sk.l_elbow.x += facingDir * 2.5 * at;
        sk.r_elbow.x += facingDir * 2.5 * at;
        // Back arm cocked during wind-up, trails during strike
        if (facingDir > 0) {
          sk.l_hand.x -= 2.5 * (1 - at); sk.l_hand.y += 0.5;
        } else {
          sk.r_hand.x += 2.5 * (1 - at); sk.r_hand.y += 0.5;
        }
        // Body lean follows the swing
        sk.chest.x += facingDir * 1.5 * t;
        sk.head.x += facingDir * 2 * t;
        sk.neck.x += facingDir * 1 * t;
        // Weight shifts to front foot
        if (facingDir > 0) {
          sk.r_foot.x += 1 * at; sk.l_foot.x -= 0.5 * at;
        } else {
          sk.l_foot.x -= 1 * at; sk.r_foot.x += 0.5 * at;
        }
      } else if (as === 47) {
        // Utilt — sweeping arm arc overhead
        const t = swing(age, 2, 4, 3);
        const at = Math.max(0, t);
        // Arms sweep: start forward-low, arc to overhead
        const arcAngle = at * Math.PI * 0.6; // 0 → ~108°
        sk.l_hand.x += facingDir * 2 * Math.cos(arcAngle);
        sk.l_hand.y += 2 + 3 * Math.sin(arcAngle);
        sk.r_hand.x -= facingDir * 1;
        sk.r_hand.y += 2 + 3 * Math.sin(arcAngle);
        sk.l_elbow.y += 2 * at;
        sk.r_elbow.y += 2 * at;
        sk.chest.y += 0.5 * at;
        sk.head.y += 0.8 * at;
      } else if (as === 48) {
        // Dtilt — low leg sweep
        for (const k of JOINT_KEYS) sk[k].y *= 0.7;
        const t = swing(age, 1, 3, 3);
        const at = Math.max(0, t);
        if (facingDir > 0) {
          sk.r_foot.x += 4 * at; sk.r_knee.x += 2 * at;
          sk.r_hand.x += 2 * at; sk.r_hand.y -= 1;
          sk.l_hand.x -= 1.5 * (1 - at); // balance arm
        } else {
          sk.l_foot.x -= 4 * at; sk.l_knee.x -= 2 * at;
          sk.l_hand.x -= 2 * at; sk.l_hand.y -= 1;
          sk.r_hand.x += 1.5 * (1 - at);
        }
        sk.chest.x += facingDir * 0.8 * at;
      } else if (as >= 49 && as <= 53) {
        // Fsmash — big wind-up then explosive release
        if (age < 8) {
          const t = ease(age, 8);
          sk.l_hand.x -= facingDir * 3.5 * t;
          sk.r_hand.x -= facingDir * 3.5 * t;
          sk.l_hand.y += 1.5 * t;
          sk.r_hand.y += 1.5 * t;
          sk.l_elbow.x -= facingDir * 2 * t;
          sk.r_elbow.x -= facingDir * 2 * t;
          sk.chest.x -= facingDir * 1 * t;
          sk.head.x -= facingDir * 0.5 * t;
          // Coil legs + compress
          sk.l_knee.y += 0.8 * t; sk.r_knee.y += 0.8 * t;
          for (const k of JOINT_KEYS) sk[k].y *= (1 - 0.12 * t);
        } else {
          // Explosive release with overshoot
          const t = swing(age - 8, 0, 3, 5);
          sk.l_hand.x += facingDir * 6 * t;
          sk.r_hand.x += facingDir * 6 * t;
          sk.l_elbow.x += facingDir * 3.5 * t;
          sk.r_elbow.x += facingDir * 3.5 * t;
          sk.chest.x += facingDir * 2 * t;
          sk.head.x += facingDir * 2.5 * t;
          sk.neck.x += facingDir * 2 * t;
          sk.l_foot.x -= 2; sk.r_foot.x += 2;
        }
      } else if (as === 54) {
        // Usmash — arms drive upward with swing arc
        const t = swing(age, 2, 4, 4);
        const at = Math.max(0, t);
        // Compress during wind-up
        if (t < 0) for (const k of JOINT_KEYS) sk[k].y *= 0.92;
        sk.l_hand.y += 5.5 * at; sk.l_hand.x = -1;
        sk.r_hand.y += 5.5 * at; sk.r_hand.x = 1;
        sk.l_elbow.y += 3.5 * at; sk.l_elbow.x = -1.5;
        sk.r_elbow.y += 3.5 * at; sk.r_elbow.x = 1.5;
        sk.chest.y += 0.8 * at;
        sk.head.y += 1.2 * at;
        sk.neck.y += 0.8 * at;
      } else if (as === 55) {
        // Dsmash — explosive split both sides
        const t = swing(age, 2, 3, 4);
        const at = Math.max(0, t);
        for (const k of JOINT_KEYS) sk[k].y *= 0.75;
        sk.l_hand.x -= 3.5 * at; sk.l_hand.y -= 2.5 * at;
        sk.r_hand.x += 3.5 * at; sk.r_hand.y -= 2.5 * at;
        sk.l_elbow.x -= 2.5 * at; sk.l_elbow.y -= 1.5 * at;
        sk.r_elbow.x += 2.5 * at; sk.r_elbow.y -= 1.5 * at;
        sk.l_foot.x -= 2.5 * at; sk.r_foot.x += 2.5 * at;
        sk.l_knee.x -= 1.5 * at; sk.r_knee.x += 1.5 * at;
      } else if (as === 56) {
        // Nair — tuck then explode into star pose, slowly retract
        const t = swing(age, 2, 3, 6);
        const at = Math.max(0, t);
        // During wind-up, tuck in
        if (t < 0) {
          const tuck = -t / 0.2;
          sk.l_hand.x += 1 * tuck; sk.r_hand.x -= 1 * tuck;
          sk.l_foot.x += 0.5 * tuck; sk.r_foot.x -= 0.5 * tuck;
          sk.l_knee.y += 1 * tuck; sk.r_knee.y += 1 * tuck;
        }
        sk.l_hand.x -= 3.5 * at; sk.l_hand.y += 2.5 * at;
        sk.r_hand.x += 3.5 * at; sk.r_hand.y += 2.5 * at;
        sk.l_elbow.x -= 2.5 * at; sk.l_elbow.y += 1.5 * at;
        sk.r_elbow.x += 2.5 * at; sk.r_elbow.y += 1.5 * at;
        sk.l_foot.x -= 3 * at; sk.l_foot.y += 0.5 * at;
        sk.r_foot.x += 3 * at; sk.r_foot.y += 0.5 * at;
        sk.l_knee.x -= 2 * at; sk.r_knee.x += 2 * at;
      } else if (as === 57) {
        // Fair — arm swings forward in arc, body rotates
        const t = swing(age, 2, 4, 4);
        const at = Math.max(0, t);
        // Swing arc: arm goes from cocked-back to full extension
        const arcT = Math.max(0, Math.min(1, (age - 1) / 5));
        const armAngle = -0.4 + arcT * 1.8; // radians: behind → in front
        sk.l_hand.x += facingDir * 4 * Math.sin(armAngle) * (0.5 + at * 0.5);
        sk.l_hand.y += 2 * Math.cos(armAngle);
        sk.r_hand.x += facingDir * 4 * Math.sin(armAngle) * (0.5 + at * 0.5);
        sk.r_hand.y += 2 * Math.cos(armAngle);
        sk.l_elbow.x += facingDir * 2.5 * at;
        sk.r_elbow.x += facingDir * 2.5 * at;
        // Back leg trails
        if (facingDir > 0) {
          sk.l_foot.x -= 2.5 * at; sk.l_foot.y += 1.5 * at;
          sk.l_knee.x -= 1.5 * at;
        } else {
          sk.r_foot.x += 2.5 * at; sk.r_foot.y += 1.5 * at;
          sk.r_knee.x += 1.5 * at;
        }
        sk.chest.x += facingDir * 1.5 * t;
        sk.head.x += facingDir * 1 * t;
      } else if (as === 58) {
        // Bair — back kick swings out
        const t = swing(age, 2, 4, 4);
        const at = Math.max(0, t);
        // Leg swings from coiled to extended backward
        if (facingDir > 0) {
          sk.l_foot.x -= 5 * at; sk.l_foot.y += 2 * at;
          sk.l_knee.x -= 3 * at; sk.l_knee.y += 1.5 * at;
        } else {
          sk.r_foot.x += 5 * at; sk.r_foot.y += 2 * at;
          sk.r_knee.x += 3 * at; sk.r_knee.y += 1.5 * at;
        }
        // Arms swing forward for counterbalance
        sk.l_hand.x += facingDir * 2.5 * at;
        sk.r_hand.x += facingDir * 2.5 * at;
        sk.l_hand.y += 1.5 * at;
        sk.r_hand.y += 1.5 * at;
        // Body tilts slightly forward
        sk.chest.x += facingDir * 0.8 * at;
      } else if (as === 59) {
        // Uair — leg sweeps upward in arc
        const t = swing(age, 2, 3, 4);
        const at = Math.max(0, t);
        // Kicking leg sweeps from below to above
        const legAngle = at * Math.PI * 0.45; // arc upward
        if (facingDir > 0) {
          sk.r_foot.x += 2 * Math.cos(legAngle);
          sk.r_foot.y += 1 + 5 * Math.sin(legAngle);
          sk.r_knee.y += 3 * at; sk.r_knee.x += 0.5;
        } else {
          sk.l_foot.x -= 2 * Math.cos(legAngle);
          sk.l_foot.y += 1 + 5 * Math.sin(legAngle);
          sk.l_knee.y += 3 * at; sk.l_knee.x -= 0.5;
        }
        sk.chest.y -= 0.5 * at;
        sk.l_hand.y -= 1.5 * at;
        sk.r_hand.y -= 1.5 * at;
        sk.l_hand.x -= 0.5; sk.r_hand.x += 0.5;
      } else if (as === 60) {
        // Dair — legs drive downward with force
        const t = swing(age, 2, 3, 4);
        const at = Math.max(0, t);
        // Legs tuck during wind-up then stomp down
        if (t < 0) {
          const tuck = -t / 0.2;
          sk.l_knee.y += 2 * tuck; sk.r_knee.y += 2 * tuck;
          sk.l_foot.y += 1.5 * tuck; sk.r_foot.y += 1.5 * tuck;
        }
        sk.l_foot.y -= 2 * at; sk.l_foot.x = -0.5;
        sk.r_foot.y -= 2 * at; sk.r_foot.x = 0.5;
        sk.l_knee.y -= 1 * at; sk.l_knee.x = -0.5;
        sk.r_knee.y -= 1 * at; sk.r_knee.x = 0.5;
        // Arms go up as legs go down
        sk.l_hand.y += 3 * at; sk.l_hand.x -= 2;
        sk.r_hand.y += 3 * at; sk.r_hand.x += 2;
        sk.l_elbow.y += 2 * at;
        sk.r_elbow.y += 2 * at;
      } else if (as >= 61 && as <= 65) {
        // Landing lag — compressed, slight forward lean, arms absorbing
        const t = 1 - ease(age, 8);
        const compress = 0.8 + 0.2 * (1 - t);
        for (const k of JOINT_KEYS) sk[k].y *= compress;
        sk.chest.x += facingDir * 0.8 * t;
        sk.head.x += facingDir * 1 * t;
        sk.l_hand.y -= 1 * t;
        sk.r_hand.y -= 1 * t;
        sk.l_hand.x -= 0.5 * t;
        sk.r_hand.x += 0.5 * t;
      } else {
        // Fallback attack pose — use swing for generic strike
        const t = swing(age, 2, 3, 4);
        const at = Math.max(0, t);
        const ext = 3.5 * facingDir * at;
        sk.l_hand.x += ext;
        sk.r_hand.x += ext;
        sk.l_elbow.x += ext * 0.5;
        sk.r_elbow.x += ext * 0.5;
        sk.chest.x += facingDir * 0.5 * t;
        if (facingDir > 0) {
          sk.r_foot.x += 1.5 * at; sk.r_knee.x += 0.8 * at;
        } else {
          sk.l_foot.x -= 1.5 * at; sk.l_knee.x -= 0.8 * at;
        }
      }
      break;
    }

    case 'damage': {
      if (as >= 75 && as <= 77) {
        // DamageFly — full extension in knockback direction
        const kbx = p.speed_attack_x;
        const kby = p.speed_attack_y;
        const mag = Math.sqrt(kbx * kbx + kby * kby) || 1;
        const nx = kbx / mag;
        const ny = kby / mag;
        // Limbs trail behind (opposite knockback direction)
        sk.l_hand.x -= nx * 3; sk.l_hand.y -= ny * 3;
        sk.r_hand.x -= nx * 3; sk.r_hand.y -= ny * 3;
        sk.l_foot.x -= nx * 2; sk.l_foot.y -= ny * 2;
        sk.r_foot.x -= nx * 2; sk.r_foot.y -= ny * 2;
        sk.l_elbow.x -= nx * 2; sk.l_elbow.y -= ny * 2;
        sk.r_elbow.x -= nx * 2; sk.r_elbow.y -= ny * 2;
        sk.head.x += nx * 1; sk.head.y += ny * 1;
      } else if (as === 78) {
        // DamageFlyTop — arms down, body arched upward
        sk.l_hand.y -= 2; sk.r_hand.y -= 2;
        sk.l_hand.x -= 1.5; sk.r_hand.x += 1.5;
        sk.l_elbow.y -= 1;
        sk.r_elbow.y -= 1;
        sk.head.y += 1; sk.chest.y += 0.5;
      } else if (as === 79) {
        // DamageFlyRoll — tumble spin using stateAge
        const spin = age * 0.25;
        const cos = Math.cos(spin);
        const sin = Math.sin(spin);
        for (const k of JOINT_KEYS) {
          const ox = sk[k].x;
          const oy = sk[k].y - 5.5; // rotate around hip height
          sk[k].x = ox * cos - oy * sin;
          sk[k].y = (ox * sin + oy * cos) + 5.5;
        }
      } else if (as >= 85 && as <= 87) {
        // DamageHi — head jerks up, body recoils
        const t = 1 - ease(age, 10);
        sk.head.y += 1.5 * t;
        sk.neck.y += 0.8 * t;
        sk.l_shoulder.y -= 0.5 * t;
        sk.r_shoulder.y -= 0.5 * t;
        sk.chest.x += facingDir * -0.8 * t;
      } else if (as >= 88 && as <= 90) {
        // DamageN — horizontal flinch
        const t = 1 - ease(age, 10);
        sk.head.x += facingDir * -2 * t;
        sk.neck.x += facingDir * -1.5 * t;
        sk.chest.x += facingDir * -1 * t;
        sk.l_shoulder.y -= 0.8 * t;
        sk.r_shoulder.y -= 0.8 * t;
        sk.hip.x += facingDir * -0.5 * t;
      } else if (as >= 91 && as <= 93) {
        // DamageLw — doubles over
        const t = 1 - ease(age, 10);
        sk.head.y -= 2 * t;
        sk.neck.y -= 1.5 * t;
        sk.chest.y -= 1 * t;
        sk.l_shoulder.y -= 1 * t;
        sk.r_shoulder.y -= 1 * t;
        sk.chest.x += facingDir * -0.5 * t;
        sk.l_hand.y -= 1 * t;
        sk.r_hand.y -= 1 * t;
      } else {
        // Fallback damage
        sk.l_shoulder.y -= 1;
        sk.r_shoulder.y -= 1;
        sk.head.y -= 0.8;
        sk.neck.y -= 0.5;
        sk.chest.x += facingDir * -1;
        sk.hip.x += facingDir * -0.5;
      }
      break;
    }

    case 'defense': {
      if (as === 183) {
        // EscapeF — rolling forward, compressed, cycling legs
        const cycle = Math.sin(age * 0.5);
        for (const k of JOINT_KEYS) sk[k].y *= 0.75;
        sk.l_hand.x = 1; sk.r_hand.x = -1;
        sk.l_hand.y = 6; sk.r_hand.y = 6;
        sk.l_elbow.x = -0.5; sk.r_elbow.x = 0.5;
        sk.l_elbow.y = 6.5; sk.r_elbow.y = 6.5;
        sk.l_foot.x += cycle * 2 * facingDir;
        sk.r_foot.x -= cycle * 2 * facingDir;
        sk.l_knee.x += cycle * 1;
        sk.r_knee.x -= cycle * 1;
      } else if (as === 184) {
        // EscapeB — rolling backward, mirror
        const cycle = Math.sin(age * 0.5);
        for (const k of JOINT_KEYS) sk[k].y *= 0.75;
        sk.l_hand.x = 1; sk.r_hand.x = -1;
        sk.l_hand.y = 6; sk.r_hand.y = 6;
        sk.l_elbow.x = -0.5; sk.r_elbow.x = 0.5;
        sk.l_elbow.y = 6.5; sk.r_elbow.y = 6.5;
        sk.l_foot.x -= cycle * 2 * facingDir;
        sk.r_foot.x += cycle * 2 * facingDir;
        sk.l_knee.x -= cycle * 1;
        sk.r_knee.x += cycle * 1;
      } else {
        // Guard / shield — arms crossed near chest
        sk.l_hand.x = 1; sk.r_hand.x = -1;
        sk.l_hand.y = 8; sk.r_hand.y = 8;
        sk.l_elbow.x = -0.5; sk.r_elbow.x = 0.5;
        sk.l_elbow.y = 8.5; sk.r_elbow.y = 8.5;
        for (const k of JOINT_KEYS) sk[k].y *= 0.9;
      }
      break;
    }

    case 'grab': {
      const gExt = 3 * facingDir;
      sk.l_hand.x += gExt;
      sk.r_hand.x += gExt;
      sk.l_hand.y = 8;
      sk.r_hand.y = 8;
      sk.l_elbow.x += gExt * 0.5;
      sk.r_elbow.x += gExt * 0.5;
      break;
    }

    case 'ledge': {
      if (as === 255 || as === 256) {
        // CliffClimb — pulling up: arms transition from overhead to pushing down
        const t = ease(age, 20);
        for (const k of JOINT_KEYS) sk[k].y -= 8 * (1 - t);
        // Arms transition: high → low
        sk.l_hand.y += 10 * (1 - t) - 2 * t;
        sk.r_hand.y += 10 * (1 - t) - 2 * t;
        sk.l_elbow.y += 6 * (1 - t);
        sk.r_elbow.y += 6 * (1 - t);
        sk.l_shoulder.y += 3 * (1 - t);
        sk.r_shoulder.y += 3 * (1 - t);
        sk.head.y += 2 * (1 - t);
        sk.neck.y += 2 * (1 - t);
      } else if (as === 263 || as === 264) {
        // CliffJump — spring upward from ledge
        const t = ease(age, 12);
        for (const k of JOINT_KEYS) sk[k].y -= 8 * (1 - t);
        // Arms sweep upward
        sk.l_hand.y += 8 * (1 - t) + 3 * t;
        sk.r_hand.y += 8 * (1 - t) + 3 * t;
        sk.l_hand.x -= 2 * t;
        sk.r_hand.x += 2 * t;
        sk.l_elbow.y += 5 * (1 - t) + 1.5 * t;
        sk.r_elbow.y += 5 * (1 - t) + 1.5 * t;
        sk.l_shoulder.y += 3 * (1 - t);
        sk.r_shoulder.y += 3 * (1 - t);
        sk.head.y += 2 * (1 - t) + 1 * t;
        sk.neck.y += 2 * (1 - t) + 0.5 * t;
      } else {
        // Hanging — body below feet position, arms up
        for (const k of JOINT_KEYS) sk[k].y -= 8;
        sk.l_hand.y += 10;
        sk.r_hand.y += 10;
        sk.l_elbow.y += 6;
        sk.r_elbow.y += 6;
        sk.l_shoulder.y += 3;
        sk.r_shoulder.y += 3;
        sk.head.y += 2;
        sk.neck.y += 2;
      }
      break;
    }
    // 'dead' is handled separately (disintegration or skip)
    // 'other' uses base skeleton as-is
  }
}

// ---------------------------------------------------------------------------
// Wireframe mesh body — internal triangulation edges
// ---------------------------------------------------------------------------

const MESH_EDGES: [string, string][] = [
  ['head', 'l_hip'],
  ['head', 'r_hip'],
  ['l_shoulder', 'hip'],
  ['r_shoulder', 'hip'],
  ['l_shoulder', 'r_hip'],
  ['r_shoulder', 'l_hip'],
  ['chest', 'l_knee'],
  ['chest', 'r_knee'],
];

function drawWireMesh(
  ctx: CanvasRenderingContext2D,
  j: Record<string, [number, number]>,
  col: PlayerColors,
): void {
  ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},0.13)`;
  ctx.lineWidth = 0.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of MESH_EDGES) {
    ctx.moveTo(j[a][0], j[a][1]);
    ctx.lineTo(j[b][0], j[b][1]);
  }
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Wireframe geometric face
// ---------------------------------------------------------------------------

function drawWireFace(
  ctx: CanvasRenderingContext2D,
  hx: number,
  hy: number,
  headR: number,
  strokeColor: string,
): void {
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1;
  ctx.lineCap = 'round';

  // Eyes — two small triangles
  const eyeOffsetX = headR * 0.3;
  const eyeOffsetY = headR * 0.15;
  const eyeW = Math.max(1.5, headR * 0.25);
  const eyeH = Math.max(1, headR * 0.18);

  // Left eye (triangle)
  ctx.beginPath();
  ctx.moveTo(hx - eyeOffsetX - eyeW * 0.5, hy - eyeOffsetY + eyeH * 0.5);
  ctx.lineTo(hx - eyeOffsetX, hy - eyeOffsetY - eyeH * 0.5);
  ctx.lineTo(hx - eyeOffsetX + eyeW * 0.5, hy - eyeOffsetY + eyeH * 0.5);
  ctx.closePath();
  ctx.stroke();

  // Right eye (triangle)
  ctx.beginPath();
  ctx.moveTo(hx + eyeOffsetX - eyeW * 0.5, hy - eyeOffsetY + eyeH * 0.5);
  ctx.lineTo(hx + eyeOffsetX, hy - eyeOffsetY - eyeH * 0.5);
  ctx.lineTo(hx + eyeOffsetX + eyeW * 0.5, hy - eyeOffsetY + eyeH * 0.5);
  ctx.closePath();
  ctx.stroke();

  // Mouth — horizontal line
  const mouthY = hy + headR * 0.3;
  const mouthW = headR * 0.35;
  ctx.beginPath();
  ctx.moveTo(hx - mouthW, mouthY);
  ctx.lineTo(hx + mouthW, mouthY);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// KO disintegration — update + draw
// ---------------------------------------------------------------------------

interface DeathSegment {
  cx: number; cy: number;    // center position (absolute screen)
  dx: number; dy: number;    // half-length vector (rotated)
  vx: number; vy: number;    // velocity
  angVel: number;
  life: number;
}

interface WireDeathState {
  active: boolean;
  segments: DeathSegment[];
  r: number; g: number; b: number;
  glowColor: string;
}

const deathStates: [WireDeathState, WireDeathState] = [
  { active: false, segments: [], r: 0, g: 0, b: 0, glowColor: '' },
  { active: false, segments: [], r: 0, g: 0, b: 0, glowColor: '' },
];

function triggerDeath(
  idx: number,
  segments: [string, string][],
  j: Record<string, [number, number]>,
  col: PlayerColors,
): void {
  const ds = deathStates[idx];
  ds.active = true;
  ds.r = col.r;
  ds.g = col.g;
  ds.b = col.b;
  ds.glowColor = col.glow;
  ds.segments = [];

  const chestX = j['chest'][0];
  const chestY = j['chest'][1];

  for (const [a, b] of segments) {
    const x1 = j[a][0], y1 = j[a][1];
    const x2 = j[b][0], y2 = j[b][1];
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const dx = (x2 - x1) / 2;
    const dy = (y2 - y1) / 2;
    const angle = Math.atan2(cy - chestY, cx - chestX) + (Math.random() - 0.5) * 1;
    const speed = 1 + Math.random() * 3;
    ds.segments.push({
      cx, cy, dx, dy,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      angVel: (Math.random() - 0.5) * 0.2,
      life: 45,
    });
  }
}

function updateAndDrawDeath(
  ctx: CanvasRenderingContext2D,
  idx: number,
): boolean {
  const ds = deathStates[idx];
  if (!ds.active) return false;

  let anyAlive = false;
  ctx.save();
  ctx.lineCap = 'round';

  for (const seg of ds.segments) {
    if (seg.life <= 0) continue;
    anyAlive = true;

    // Update physics
    seg.cx += seg.vx;
    seg.cy += seg.vy;
    seg.vy += 0.12;
    seg.vx *= 0.98;
    seg.vy *= 0.98;

    // Rotate half-delta
    const cos = Math.cos(seg.angVel);
    const sin = Math.sin(seg.angVel);
    const ndx = seg.dx * cos - seg.dy * sin;
    const ndy = seg.dx * sin + seg.dy * cos;
    seg.dx = ndx;
    seg.dy = ndy;

    seg.life--;
    const alpha = seg.life / 45;

    // Draw
    ctx.strokeStyle = `rgba(${ds.r},${ds.g},${ds.b},${alpha})`;
    ctx.lineWidth = 2;
    ctx.shadowColor = `rgba(${ds.r},${ds.g},${ds.b},${alpha * 0.5})`;
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(seg.cx - seg.dx, seg.cy - seg.dy);
    ctx.lineTo(seg.cx + seg.dx, seg.cy + seg.dy);
    ctx.stroke();
  }

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.restore();

  if (!anyAlive) ds.active = false;
  return true;
}

// ---------------------------------------------------------------------------
// Hit scatter — emit particles from joints on hitlag start
// ---------------------------------------------------------------------------

function emitHitScatter(
  j: Record<string, [number, number]>,
  col: PlayerColors,
): void {
  for (const k of JOINT_KEYS) {
    const [sx, sy] = j[k];
    const count = 1 + (Math.random() > 0.5 ? 1 : 0); // 1-2 particles
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1 + Math.random() * 2;
      const life = 8 + Math.floor(Math.random() * 7);
      const size = 1 + Math.random();
      emitParticle(
        sx, sy,
        Math.cos(angle) * speed,
        Math.sin(angle) * speed,
        life, size,
        col.r, col.g, col.b,
        0.8,
        0.08,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Skeleton segments (shared between draw and disintegration)
// ---------------------------------------------------------------------------

const SKELETON_SEGMENTS: [string, string][] = [
  ['neck', 'chest'],
  ['chest', 'hip'],
  ['neck', 'l_shoulder'],
  ['l_shoulder', 'l_elbow'],
  ['l_elbow', 'l_hand'],
  ['neck', 'r_shoulder'],
  ['r_shoulder', 'r_elbow'],
  ['r_elbow', 'r_hand'],
  ['hip', 'l_hip'],
  ['l_hip', 'l_knee'],
  ['l_knee', 'l_foot'],
  ['hip', 'r_hip'],
  ['r_hip', 'r_knee'],
  ['r_knee', 'r_foot'],
];

// ---------------------------------------------------------------------------
// Main draw function
// ---------------------------------------------------------------------------

export function drawWirePlayer(
  ctx: CanvasRenderingContext2D,
  p: VizPlayerFrame,
  idx: number,
  px: number,
  py: number,
  col: PlayerColors,
  gameScaleX: (u: number) => number,
  gameScaleY: (u: number) => number,
  gameToScreen: (gx: number, gy: number) => [number, number],
): void {
  const category = actionCategory(p.action_state);
  const pidx = idx as 0 | 1;

  // --- KO disintegration: trigger on transition to dead ---
  if (category === 'dead' && prevCategory[pidx] !== null && prevCategory[pidx] !== 'dead') {
    // We need screen joints from *last frame* — but we don't have them stored.
    // Use current position with base skeleton as approximate final pose.
    const sk = cloneSkeleton();
    const facingDir = p.facing === 1 ? 1 : -1;
    if (facingDir === -1) {
      for (const k of JOINT_KEYS) sk[k].x = -sk[k].x;
    }
    const j: Record<string, [number, number]> = {};
    for (const k of JOINT_KEYS) {
      j[k] = gameToScreen(px + sk[k].x, py + sk[k].y);
    }
    triggerDeath(pidx, SKELETON_SEGMENTS, j, col);
  }

  // Clear death state on respawn
  if (category !== 'dead' && prevCategory[pidx] === 'dead') {
    deathStates[pidx].active = false;
  }

  // Update prev state
  prevCategory[pidx] = category;
  const currentHitlag = p.hitlag;
  const wasHitlag = prevHitlag[pidx];
  prevHitlag[pidx] = currentHitlag;

  // --- Draw disintegration if active ---
  if (category === 'dead') {
    updateAndDrawDeath(ctx, pidx);
    return;
  }

  ctx.save();

  // --- Build skeleton and apply pose ---
  const sk = cloneSkeleton();
  const facingDir = p.facing === 1 ? 1 : -1;
  applyPose(sk, p, category, facingDir);

  // --- Apply facing (flip X if facing left) ---
  if (facingDir === -1) {
    for (const k of JOINT_KEYS) {
      sk[k].x = -sk[k].x;
    }
  }

  // --- Damage jitter ---
  const isDamage = p.action_state >= 75 && p.action_state <= 93;
  const jitterAmount = isDamage ? 1.5 : 0;

  // --- Convert all joints to screen coords ---
  const screenJoints: Record<string, [number, number]> = {};
  for (const k of JOINT_KEYS) {
    const jx = sk[k].x;
    const jy = sk[k].y;
    const [sx, sy] = gameToScreen(px + jx, py + jy);
    if (jitterAmount > 0) {
      screenJoints[k] = [
        sx + (Math.random() - 0.5) * 2 * jitterAmount,
        sy + (Math.random() - 0.5) * 2 * jitterAmount,
      ];
    } else {
      screenJoints[k] = [sx, sy];
    }
  }

  const j = screenJoints;

  // --- Hit scatter: emit particles on hitlag start ---
  if (currentHitlag > 0 && wasHitlag === 0) {
    emitHitScatter(j, col);
  }

  // --- Hit effect state ---
  const inHitlag = p.hitlag > 0;
  const strokeColor = inHitlag ? '#ffffff' : col.main;
  const lineW = inHitlag ? 3 : 2;

  // --- Semi-transparent body fill (behind everything) ---
  ctx.beginPath();
  const fillOrder = [
    'l_hand', 'l_elbow', 'l_shoulder',
    'head',
    'r_shoulder', 'r_elbow', 'r_hand',
    'r_hip', 'r_knee', 'r_foot',
    'l_foot', 'l_knee', 'l_hip',
  ];
  ctx.moveTo(j[fillOrder[0]][0], j[fillOrder[0]][1]);
  for (let i = 1; i < fillOrder.length; i++) {
    ctx.lineTo(j[fillOrder[i]][0], j[fillOrder[i]][1]);
  }
  ctx.closePath();
  ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},0.05)`;
  ctx.fill();

  // --- Wireframe mesh (internal edges) ---
  drawWireMesh(ctx, j, col);

  // --- Draw glow pass (lines with shadow) ---
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lineW;
  ctx.lineCap = 'round';
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 6;

  // Draw line segments
  ctx.beginPath();
  for (const [a, b] of SKELETON_SEGMENTS) {
    ctx.moveTo(j[a][0], j[a][1]);
    ctx.lineTo(j[b][0], j[b][1]);
  }
  ctx.stroke();

  // Draw head circle + face
  const headR = gameScaleX(1.2);
  ctx.beginPath();
  ctx.arc(j['head'][0], j['head'][1], headR, 0, Math.PI * 2);
  ctx.stroke();

  // --- Clear shadow for face + joints ---
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;

  // --- Geometric face ---
  drawWireFace(ctx, j['head'][0], j['head'][1], headR, strokeColor);

  // --- Joint dots ---
  ctx.fillStyle = strokeColor;
  for (const k of JOINT_KEYS) {
    if (k === 'head') continue;
    ctx.beginPath();
    ctx.arc(j[k][0], j[k][1], 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // --- Pulsing heart at chest ---
  const heartBeatFreq = 0.006 * (1 + p.percent / 100);
  const heartPulse = Math.sin(Date.now() * heartBeatFreq);
  const heartBaseR = gameScaleX(0.6);
  const heartR = heartBaseR * (0.8 + 0.4 * (heartPulse * 0.5 + 0.5));

  const heartMix = Math.min(1, p.percent / 150);
  const heartColor = lerpColor(col.main, '#ec4899', heartMix);

  ctx.beginPath();
  ctx.arc(j['chest'][0], j['chest'][1], heartR, 0, Math.PI * 2);
  ctx.fillStyle = heartColor;
  ctx.shadowColor = heartColor;
  ctx.shadowBlur = 8 + heartPulse * 4;
  ctx.fill();

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;

  ctx.restore();
}
