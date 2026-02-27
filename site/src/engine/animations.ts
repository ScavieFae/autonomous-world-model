/**
 * Animation cache for Melee character SVG animations.
 * Loads ZIP files from /zips/ containing per-action SVG path strings.
 * Source: SlippiLab (MIT) — https://github.com/frankborden/slippilab
 */

import { unzipSync } from 'fflate';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Animation name → array of SVG path `d` strings (one per frame). */
export type CharacterAnimations = Record<string, string[]>;

// ---------------------------------------------------------------------------
// Internal → External character ID mapping
// ---------------------------------------------------------------------------

const externalIdByInternalId: Record<number, number> = {
  0: 8, 1: 2, 2: 0, 3: 1, 4: 4, 5: 5, 6: 6, 7: 19, 8: 11, 9: 12,
  10: 14, 11: 14, 12: 13, 13: 16, 14: 17, 15: 15, 16: 10, 17: 7,
  18: 9, 19: 18, 20: 21, 21: 22, 22: 20, 23: 24, 24: 3, 25: 25, 26: 23,
};

const characterZipByExternalId: Record<number, string> = {
  0: 'captainFalcon', 1: 'donkeyKong', 2: 'fox', 3: 'mrGameAndWatch', 4: 'kirby',
  5: 'bowser', 6: 'link', 7: 'luigi', 8: 'mario', 9: 'marth', 10: 'mewtwo',
  11: 'ness', 12: 'peach', 13: 'pikachu', 14: 'iceClimbers', 15: 'jigglypuff',
  16: 'samus', 17: 'yoshi', 18: 'zelda', 19: 'sheik', 20: 'falco',
  21: 'youngLink', 22: 'doctorMario', 23: 'roy', 24: 'pichu', 25: 'ganondorf',
};

// Character scale factors (indexed by external ID)
export const characterScaleByExternalId: Record<number, number> = {
  0: 1.12, 1: 1.30, 2: 0.96, 3: 0.85, 4: 0.82, 5: 0.90, 6: 1.05,
  7: 0.95, 8: 0.95, 9: 1.05, 10: 1.30, 11: 0.90, 12: 0.95, 13: 0.85,
  14: 0.90, 15: 0.70, 16: 1.10, 17: 1.00, 18: 1.00, 19: 0.96, 20: 0.98,
  21: 1.05, 22: 0.95, 23: 1.05, 24: 0.55, 25: 1.20,
};

// ---------------------------------------------------------------------------
// Action state → animation name mapping
// ---------------------------------------------------------------------------

const actionNameById = [
  'DeadDown', 'DeadLeft', 'DeadRight', 'DeadUp', 'DeadUpStar', 'DeadUpStarIce',
  'DeadUpFall', 'DeadUpFallHitCamera', 'DeadUpFallHitCameraFlat', 'DeadUpFallIce',
  'DeadUpFallHitCameraIce', 'Sleep', 'Rebirth', 'RebirthWait', 'Wait', 'WalkSlow',
  'WalkMiddle', 'WalkFast', 'Turn', 'TurnRun', 'Dash', 'Run', 'RunDirect', 'RunBrake',
  'KneeBend', 'JumpF', 'JumpB', 'JumpAerialF', 'JumpAerialB', 'Fall', 'FallF', 'FallB',
  'FallAerial', 'FallAerialF', 'FallAerialB', 'FallSpecial', 'FallSpecialF', 'FallSpecialB',
  'DamageFall', 'Squat', 'SquatWait', 'SquatRv', 'Landing', 'LandingFallSpecial',
  'Attack11', 'Attack12', 'Attack13', 'Attack100Start', 'Attack100Loop', 'Attack100End',
  'AttackDash', 'AttackS3Hi', 'AttackS3HiS', 'AttackS3S', 'AttackS3LwS', 'AttackS3Lw',
  'AttackHi3', 'AttackLw3', 'AttackS4Hi', 'AttackS4HiS', 'AttackS4S', 'AttackS4LwS',
  'AttackS4Lw', 'AttackHi4', 'AttackLw4', 'AttackAirN', 'AttackAirF', 'AttackAirB',
  'AttackAirHi', 'AttackAirLw', 'LandingAirN', 'LandingAirF', 'LandingAirB', 'LandingAirHi',
  'LandingAirLw', 'DamageHi1', 'DamageHi2', 'DamageHi3', 'DamageN1', 'DamageN2', 'DamageN3',
  'DamageLw1', 'DamageLw2', 'DamageLw3', 'DamageAir1', 'DamageAir2', 'DamageAir3',
  'DamageFlyHi', 'DamageFlyN', 'DamageFlyLw', 'DamageFlyTop', 'DamageFlyRoll',
  'LightGet', 'HeavyGet', 'LightThrowF', 'LightThrowB', 'LightThrowHi', 'LightThrowLw',
  'LightThrowDash', 'LightThrowDrop', 'LightThrowAirF', 'LightThrowAirB', 'LightThrowAirHi',
  'LightThrowAirLw', 'HeavyThrowF', 'HeavyThrowB', 'HeavyThrowHi', 'HeavyThrowLw',
  'LightThrowF4', 'LightThrowB4', 'LightThrowHi4', 'LightThrowLw4', 'LightThrowAirF4',
  'LightThrowAirB4', 'LightThrowAirHi4', 'LightThrowAirLw4', 'HeavyThrowF4', 'HeavyThrowB4',
  'HeavyThrowHi4', 'HeavyThrowLw4', 'SwordSwing1', 'SwordSwing3', 'SwordSwing4',
  'SwordSwingDash', 'BatSwing1', 'BatSwing3', 'BatSwing4', 'BatSwingDash', 'ParasolSwing1',
  'ParasolSwing3', 'ParasolSwing4', 'ParasolSwingDash', 'HarisenSwing1', 'HarisenSwing3',
  'HarisenSwing4', 'HarisenSwingDash', 'StarRodSwing1', 'StarRodSwing3', 'StarRodSwing4',
  'StarRodSwingDash', 'LipStickSwing1', 'LipStickSwing3', 'LipStickSwing4', 'LipStickSwingDash',
  'ItemParasolOpen', 'ItemParasolFall', 'ItemParasolFallSpecial', 'ItemParasolDamageFall',
  'LGunShoot', 'LGunShootAir', 'LGunShootEmpty', 'LGunShootAirEmpty', 'FireFlowerShoot',
  'FireFlowerShootAir', 'ItemScrew', 'ItemScrewAir', 'DamageScrew', 'DamageScrewAir',
  'ItemScopeStart', 'ItemScopeRapid', 'ItemScopeFire', 'ItemScopeEnd', 'ItemScopeAirStart',
  'ItemScopeAirRapid', 'ItemScopeAirFire', 'ItemScopeAirEnd', 'ItemScopeStartEmpty',
  'ItemScopeRapidEmpty', 'ItemScopeFireEmpty', 'ItemScopeEndEmpty', 'ItemScopeAirStartEmpty',
  'ItemScopeAirRapidEmpty', 'ItemScopeAirFireEmpty', 'ItemScopeAirEndEmpty', 'LiftWait',
  'LiftWalk1', 'LiftWalk2', 'LiftTurn', 'GuardOn', 'Guard', 'GuardOff', 'GuardSetOff',
  'GuardReflect', 'DownBoundU', 'DownWaitU', 'DownDamageU', 'DownStandU', 'DownAttackU',
  'DownFowardU', 'DownBackU', 'DownSpotU', 'DownBoundD', 'DownWaitD', 'DownDamageD',
  'DownStandD', 'DownAttackD', 'DownFowardD', 'DownBackD', 'DownSpotD', 'Passive',
  'PassiveStandF', 'PassiveStandB', 'PassiveWall', 'PassiveWallJump', 'PassiveCeil',
  'ShieldBreakFly', 'ShieldBreakFall', 'ShieldBreakDownU', 'ShieldBreakDownD',
  'ShieldBreakStandU', 'ShieldBreakStandD', 'FuraFura', 'Catch', 'CatchPull', 'CatchDash',
  'CatchDashPull', 'CatchWait', 'CatchAttack', 'CatchCut', 'ThrowF', 'ThrowB', 'ThrowHi',
  'ThrowLw', 'CapturePulledHi', 'CaptureWaitHi', 'CaptureDamageHi', 'CapturePulledLw',
  'CaptureWaitLw', 'CaptureDamageLw', 'CaptureCut', 'CaptureJump', 'CaptureNeck',
  'CaptureFoot', 'EscapeF', 'EscapeB', 'Escape', 'EscapeAir', 'ReboundStop', 'Rebound',
  'ThrownF', 'ThrownB', 'ThrownHi', 'ThrownLw', 'ThrownLwWomen', 'Pass', 'Ottotto',
  'OttottoWait', 'FlyReflectWall', 'FlyReflectCeil', 'StopWall', 'StopCeil', 'MissFoot',
  'CliffCatch', 'CliffWait', 'CliffClimbSlow', 'CliffClimbQuick', 'CliffAttackSlow',
  'CliffAttackQuick', 'CliffEscapeSlow', 'CliffEscapeQuick', 'CliffJumpSlow1',
  'CliffJumpSlow2', 'CliffJumpQuick1', 'CliffJumpQuick2', 'AppealR', 'AppealL',
] as const;

const animationRemaps: Record<string, string> = {
  'AppealL': 'Appeal', 'AppealR': 'Appeal',
  'Escape': 'EscapeN',
  'GuardReflect': 'Guard', 'GuardSetOff': 'GuardDamage',
  'KneeBend': 'Landing', 'LandingFallSpecial': 'Landing',
  'CatchDashPull': 'CatchWait', 'CatchPull': 'CatchWait',
  'EntryEnd': 'Entry', 'EntryStart': 'Entry',
  'Wait': 'Wait1',
};

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

const cache = new Map<number, CharacterAnimations>();
const loading = new Map<number, Promise<CharacterAnimations | null>>();

// Path2D cache: externalId → animName → frameIdx → Path2D
const path2dCache = new Map<number, Map<string, Map<number, Path2D>>>();

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Get the external character ID for an internal ID. */
export function getExternalId(internalCharId: number): number {
  return externalIdByInternalId[internalCharId] ?? 8; // fallback to Mario
}

/** Fetch and cache animations for a character by internal ID. */
export async function fetchAnimations(internalCharId: number): Promise<CharacterAnimations | null> {
  const extId = getExternalId(internalCharId);
  const cached = cache.get(extId);
  if (cached) return cached;

  const inFlight = loading.get(extId);
  if (inFlight) return inFlight;

  const zipName = characterZipByExternalId[extId];
  if (!zipName) return null;

  const promise = (async () => {
    try {
      const response = await fetch(`/zips/${zipName}.zip`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const arrayBuffer = await response.arrayBuffer();
      const unzipped = unzipSync(new Uint8Array(arrayBuffer));

      const animations: CharacterAnimations = {};
      for (const [filename, data] of Object.entries(unzipped)) {
        if (!filename.endsWith('.json')) continue;
        const animName = filename.replace('.json', '');
        const text = new TextDecoder().decode(data);
        animations[animName] = JSON.parse(text);
      }

      cache.set(extId, animations);
      console.log(`[anim] Loaded ${zipName}: ${Object.keys(animations).length} animations`);
      return animations;
    } catch (err) {
      console.error(`[anim] Failed to load ${zipName}.zip:`, err);
      return null;
    } finally {
      loading.delete(extId);
    }
  })();

  loading.set(extId, promise);
  return promise;
}

/** Get cached animations (synchronous — returns undefined if not loaded). */
export function getAnimations(internalCharId: number): CharacterAnimations | undefined {
  return cache.get(getExternalId(internalCharId));
}

/** Resolve the SVG path string for a given action state + frame age. */
export function resolveFrame(
  anims: CharacterAnimations,
  actionState: number,
  stateAge: number,
): string | null {
  const rawName = actionNameById[actionState] ?? 'Wait';
  const animName = animationRemaps[rawName] ?? rawName;

  const frames = anims[animName] ?? anims['Wait1'];
  if (!frames || frames.length === 0) return null;

  const frameIdx = Math.floor(Math.max(0, stateAge)) % frames.length;
  let path = frames[frameIdx];

  // Handle frame references like "frame20"
  if (path?.startsWith('frame')) {
    const refIdx = parseInt(path.slice(5), 10);
    path = frames[refIdx];
  }

  return path ?? null;
}

/** Get or create a cached Path2D for the given SVG path string. */
export function getPath2D(
  internalCharId: number,
  animName: string,
  frameIdx: number,
  svgPath: string,
): Path2D {
  const extId = getExternalId(internalCharId);
  let charCache = path2dCache.get(extId);
  if (!charCache) {
    charCache = new Map();
    path2dCache.set(extId, charCache);
  }
  let animCache = charCache.get(animName);
  if (!animCache) {
    animCache = new Map();
    charCache.set(animName, animCache);
  }
  let p2d = animCache.get(frameIdx);
  if (!p2d) {
    p2d = new Path2D(svgPath);
    animCache.set(frameIdx, p2d);
  }
  return p2d;
}

/** Preload animations for characters found in a set of frames. */
export function preloadFromFrames(frames: Array<{ players: [{ character: number }, { character: number }] }>): void {
  if (frames.length === 0) return;
  const charIds = new Set<number>();
  charIds.add(frames[0].players[0].character);
  charIds.add(frames[0].players[1].character);
  for (const id of charIds) {
    fetchAnimations(id);
  }
}
