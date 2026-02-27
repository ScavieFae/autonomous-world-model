/**
 * Particle system — object-pooled spark/dust particles + ambient background.
 * Ported from viz/visualizer-juicy.html L1219-1319.
 */

// ---------------------------------------------------------------------------
// Pool particle types
// ---------------------------------------------------------------------------

export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
  r: number;
  g: number;
  b: number;
  alpha: number;
  active: boolean;
  gravity: number;
}

export interface AmbientParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  alpha: number;
  size: number;
}

// ---------------------------------------------------------------------------
// Object-pooled particle system
// ---------------------------------------------------------------------------

export const PARTICLE_POOL_SIZE = 600;

const particlePool: Particle[] = new Array(PARTICLE_POOL_SIZE);

for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
  particlePool[i] = {
    x: 0, y: 0, vx: 0, vy: 0,
    life: 0, maxLife: 1, size: 2,
    r: 255, g: 255, b: 255,
    alpha: 1, active: false, gravity: 0,
  };
}

/**
 * Activate a dead particle from the pool with the given parameters.
 * Returns the particle, or null if the pool is exhausted.
 */
export function emitParticle(
  x: number, y: number,
  vx: number, vy: number,
  life: number, size: number,
  r: number, g: number, b: number,
  alpha: number,
  gravity: number = 0,
): Particle | null {
  for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
    if (!particlePool[i].active) {
      const p = particlePool[i];
      p.x = x; p.y = y; p.vx = vx; p.vy = vy;
      p.life = life; p.maxLife = life;
      p.size = size; p.r = r; p.g = g; p.b = b;
      p.alpha = alpha; p.active = true;
      p.gravity = gravity;
      return p;
    }
  }
  return null;
}

/** Tick every active particle: decrement life, apply velocity + gravity + damping. */
export function updateParticles(): void {
  for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
    const p = particlePool[i];
    if (!p.active) continue;
    p.life--;
    if (p.life <= 0) { p.active = false; continue; }
    p.x += p.vx;
    p.y += p.vy;
    p.vy += p.gravity;
    p.vx *= 0.98;
    p.vy *= 0.98;
  }
}

/** Draw every active particle as a filled circle with fade based on remaining life. */
export function drawParticles(ctx: CanvasRenderingContext2D): void {
  for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
    const p = particlePool[i];
    if (!p.active) continue;
    const t = p.life / p.maxLife;
    const a = p.alpha * t;
    if (a < 0.01) continue;
    ctx.globalAlpha = a;
    ctx.fillStyle = `rgb(${p.r},${p.g},${p.b})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * t, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

/** Deactivate every particle in the pool. */
export function clearAllParticles(): void {
  for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
    particlePool[i].active = false;
  }
}

// ---------------------------------------------------------------------------
// Ambient background particles (normalized 0-1 coordinates)
// ---------------------------------------------------------------------------

export const AMBIENT_COUNT = 30;

export const ambientParticles: AmbientParticle[] = [];

for (let i = 0; i < AMBIENT_COUNT; i++) {
  ambientParticles.push({
    x: Math.random(),
    y: Math.random(),
    vx: (Math.random() - 0.5) * 0.0003,
    vy: (Math.random() - 0.5) * 0.0003,
    alpha: 0.08 + Math.random() * 0.12,
    size: 1 + Math.random() * 1.5,
  });
}

/** Move ambient particles by their velocity, wrapping around edges. */
export function updateAmbientParticles(): void {
  for (const p of ambientParticles) {
    p.x += p.vx;
    p.y += p.vy;
    if (p.x < 0) p.x += 1;
    if (p.x > 1) p.x -= 1;
    if (p.y < 0) p.y += 1;
    if (p.y > 1) p.y -= 1;
  }
}

/** Draw ambient particles as white circles at normalized position scaled to canvas size. */
export function drawAmbientParticles(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  for (const p of ambientParticles) {
    ctx.globalAlpha = p.alpha;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, p.size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
