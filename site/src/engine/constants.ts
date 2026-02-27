/**
 * Game constants — action states, characters, stage geometry, colors.
 * Ported from viz/visualizer-juicy.html L463-662.
 */

export const ACTION_NAMES: Record<number, string> = {
  0: 'DeadDown', 1: 'DeadLeft', 2: 'DeadRight', 3: 'DeadUp',
  14: 'Wait', 15: 'WalkSlow', 16: 'WalkMiddle', 17: 'WalkFast',
  20: 'Dash', 21: 'Run', 22: 'RunDirect', 23: 'RunBrake',
  24: 'KneeBend', 25: 'JumpF', 26: 'JumpB', 27: 'JumpAerialF',
  28: 'JumpAerialB', 29: 'Fall', 30: 'FallF', 31: 'FallB',
  32: 'FallAerial', 33: 'FallAerialF', 34: 'FallAerialB',
  35: 'FallSpecial', 36: 'FallSpecialF', 37: 'FallSpecialB',
  39: 'SquatWait', 40: 'Squat', 41: 'SquatRv',
  42: 'AttackS3Hi', 43: 'AttackS3HiS', 44: 'AttackS3S',
  45: 'AttackS3LwS', 46: 'AttackS3Lw',
  47: 'AttackHi3', 48: 'AttackLw3',
  49: 'AttackS4Hi', 50: 'AttackS4HiS', 51: 'AttackS4S',
  52: 'AttackS4LwS', 53: 'AttackS4Lw',
  54: 'AttackHi4', 55: 'AttackLw4',
  56: 'AttackAirN', 57: 'AttackAirF', 58: 'AttackAirB',
  59: 'AttackAirHi', 60: 'AttackAirLw',
  61: 'LandingAirN', 62: 'LandingAirF', 63: 'LandingAirB',
  64: 'LandingAirHi', 65: 'LandingAirLw',
  75: 'DamageFlyHi', 76: 'DamageFlyN', 77: 'DamageFlyLw',
  78: 'DamageFlyTop', 79: 'DamageFlyRoll',
  85: 'DamageHi1', 86: 'DamageHi2', 87: 'DamageHi3',
  88: 'DamageN1', 89: 'DamageN2', 90: 'DamageN3',
  91: 'DamageLw1', 92: 'DamageLw2', 93: 'DamageLw3',
  178: 'GuardOn', 179: 'Guard', 180: 'GuardOff',
  181: 'GuardSetOff', 182: 'GuardReflect',
  183: 'EscapeF', 184: 'EscapeB',
  199: 'ReboundStop',
  212: 'Catch', 214: 'CatchWait',
  233: 'Landing', 234: 'LandingFallSpecial',
  253: 'CliffCatch', 254: 'CliffWait',
  255: 'CliffClimbSlow', 256: 'CliffClimbQuick',
  263: 'CliffJumpSlow1', 264: 'CliffJumpQuick1',
};

export const ATTACK_DISPLAY_NAMES: Record<number, string> = {
  42: 'Ftilt!', 43: 'Ftilt!', 44: 'Ftilt!', 45: 'Ftilt!', 46: 'Ftilt!',
  47: 'Utilt!', 48: 'Dtilt!',
  49: 'Fsmash!', 50: 'Fsmash!', 51: 'Fsmash!', 52: 'Fsmash!', 53: 'Fsmash!',
  54: 'Upsmash!', 55: 'Dsmash!',
  56: 'Nair!', 57: 'Fair!', 58: 'Bair!', 59: 'Uair!', 60: 'Dair!',
  212: 'Grab!',
};

export function actionName(id: number): string {
  return ACTION_NAMES[id] || `State_${id}`;
}

export function attackDisplayName(id: number): string {
  return ATTACK_DISPLAY_NAMES[id] || actionName(id);
}

export type ActionCategory = 'neutral' | 'movement' | 'aerial' | 'attack' | 'damage' | 'defense' | 'grab' | 'ledge' | 'dead' | 'other';

export function actionCategory(id: number): ActionCategory {
  if (id <= 10) return 'dead';
  if (id === 14) return 'neutral';
  if (id >= 15 && id <= 23) return 'movement';
  if (id >= 24 && id <= 37) return 'aerial';
  if (id >= 39 && id <= 41) return 'movement';
  if (id >= 42 && id <= 65) return 'attack';
  if (id >= 75 && id <= 93) return 'damage';
  if (id >= 178 && id <= 184) return 'defense';
  if (id >= 212 && id <= 214) return 'grab';
  if (id >= 253 && id <= 264) return 'ledge';
  return 'other';
}

export const CATEGORY_COLORS: Record<ActionCategory, string> = {
  neutral:  'rgba(255,255,255,0.5)',
  movement: 'rgba(6,182,212,0.8)',
  aerial:   'rgba(167,139,250,0.8)',
  attack:   'rgba(244,63,94,0.9)',
  damage:   'rgba(245,158,11,0.9)',
  defense:  'rgba(34,197,94,0.7)',
  grab:     'rgba(236,72,153,0.8)',
  ledge:    'rgba(99,102,241,0.8)',
  dead:     'rgba(107,114,128,0.5)',
  other:    'rgba(148,163,184,0.5)',
};

export const CHARACTERS: Record<number, string> = {
  0: 'Mario', 1: 'Fox', 2: 'Cpt. Falcon', 3: 'DK', 4: 'Kirby',
  5: 'Bowser', 6: 'Link', 7: 'Sheik', 8: 'Ness', 9: 'Peach',
  10: 'Popo', 12: 'Pikachu', 13: 'Samus', 17: 'Luigi',
  18: 'Marth', 19: 'Zelda', 20: 'Young Link', 21: 'Doc',
  22: 'Falco', 23: 'Pichu', 24: 'G&W', 25: 'Ganondorf',
  26: 'Roy', 31: 'Mewtwo', 32: 'Jigglypuff',
};

export const STAGE = {
  name: 'Final Destination',
  ground: { x1: -85.5606, x2: 85.5606, y: 0 },
  blastzone: { left: -246, right: 246, top: 188, bottom: -140 },
  camera: { left: -170, right: 170, top: 120, bottom: -60 },
} as const;

export const CAPSULE_GAME_H = 10;
export const CAPSULE_GAME_W = 3.5;
