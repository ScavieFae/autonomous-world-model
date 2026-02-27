/**
 * Character SVG renderer — draws actual Melee character silhouettes.
 * Supports solid fill and 7 X-ray fill modes cycled with the X key:
 *   stroke, scanline, hatch, grid, ghost, gradient, stipple
 *
 * SVG paths are stored in ~1000×1000 pixel space with the character
 * centered around (500, 500). The transform chain converts:
 *   SVG path space → game units → screen pixels
 */

import type { VizPlayerFrame, PlayerColors, CharacterFillMode } from '../types';
import {
  getAnimations,
  resolveFrame,
  getPath2D,
  getExternalId,
  characterScaleByExternalId,
} from '../animations';

/**
 * Draw a character using SVG animation data.
 * Returns true if drawn, false if animations aren't loaded (caller should fall back).
 *
 * fillMode: 'solid' for capsule mode, or one of the 7 CharacterFillMode values for xray.
 */
export function drawCharacterPlayer(
  ctx: CanvasRenderingContext2D,
  p: VizPlayerFrame,
  idx: number,
  px: number,
  py: number,
  col: PlayerColors,
  gameScaleX: (u: number) => number,
  gameScaleY: (u: number) => number,
  gameToScreen: (gx: number, gy: number) => [number, number],
  fillMode: CharacterFillMode | 'solid',
): boolean {
  const anims = getAnimations(p.character);
  if (!anims) return false;

  const svgPath = resolveFrame(anims, p.action_state, p.state_age);
  if (!svgPath) return false;

  const extId = getExternalId(p.character);
  const charScale = characterScaleByExternalId[extId] ?? 1.0;
  const facingDir = p.facing === 1 ? 1 : -1;
  const inHitlag = p.hitlag > 0;

  // Get screen position and scale
  const [sx, sy] = gameToScreen(px, py);
  const pxPerUnit = gameScaleX(1);

  // Build Path2D (cached)
  const rawName = String(p.action_state);
  const frameIdx = Math.floor(Math.max(0, p.state_age)) % 1000;
  const path2d = getPath2D(p.character, rawName, frameIdx, svgPath);

  ctx.save();

  // Transform chain: screen ← game ← SVG path
  // SVG paths are in ~1000px space, Y+ down (same as screen).
  // gameToScreen already flipped game Y → screen Y, so no Y flip needed here.
  ctx.translate(sx, sy);
  const s = pxPerUnit * charScale * 0.1;
  ctx.scale(s * facingDir, s);
  // Anchor at foot level
  ctx.translate(-500, -510);

  if (fillMode === 'solid') {
    drawSolid(ctx, path2d, col, s, inHitlag);
  } else {
    drawXrayFill(ctx, path2d, col, s, inHitlag, fillMode);
  }

  ctx.restore();
  return true;
}

// ---------------------------------------------------------------------------
// Solid fill (capsule mode)
// ---------------------------------------------------------------------------

function drawSolid(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  inHitlag: boolean,
): void {
  const fillColor = inHitlag ? 'rgba(255,255,255,0.9)' : col.main;
  ctx.fillStyle = fillColor;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 6;
  ctx.fill(path);

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(0,0,0,0.3)';
  ctx.lineWidth = 2 / s;
  ctx.stroke(path);
}

// ---------------------------------------------------------------------------
// X-ray fill modes
// ---------------------------------------------------------------------------

function drawXrayFill(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  inHitlag: boolean,
  mode: CharacterFillMode,
): void {
  const strokeColor = inHitlag ? '#ffffff' : col.main;

  switch (mode) {
    case 'stroke':
      drawStroke(ctx, path, col, s, strokeColor);
      break;
    case 'scanline':
      drawScanline(ctx, path, col, s, strokeColor);
      break;
    case 'hatch':
      drawHatch(ctx, path, col, s, strokeColor);
      break;
    case 'grid':
      drawGrid(ctx, path, col, s, strokeColor);
      break;
    case 'ghost':
      drawGhost(ctx, path, col, s, strokeColor, inHitlag);
      break;
    case 'gradient':
      drawGradient(ctx, path, col, s, strokeColor);
      break;
    case 'stipple':
      drawStipple(ctx, path, col, s, strokeColor);
      break;
    case 'hologram':
      drawHologram(ctx, path, col, s, strokeColor, inHitlag);
      break;
  }
}

// ---------------------------------------------------------------------------
// 1. Stroke — wireframe outline with glow
// ---------------------------------------------------------------------------

function drawStroke(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
): void {
  // Glow pass
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.5 / s;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 10;
  ctx.stroke(path);

  // Sharp pass on top
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.2 / s;
  ctx.stroke(path);
}

// ---------------------------------------------------------------------------
// 2. Scanline — horizontal lines clipped to silhouette
// ---------------------------------------------------------------------------

function drawScanline(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
): void {
  // Clip to silhouette, draw horizontal scanlines
  ctx.save();
  ctx.clip(path);

  const spacing = 6 / s; // tight ~6px spacing for dense CRT look
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.5 / s;
  ctx.globalAlpha = 0.7;

  for (let y = 0; y < 1000; y += spacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1000, y);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.restore();

  // Soft glow outline on top
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1 / s;
  ctx.globalAlpha = 0.45;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 12;
  ctx.stroke(path);
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// 3. Hatch — diagonal cross-hatch clipped to silhouette
// ---------------------------------------------------------------------------

function drawHatch(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
): void {
  // Subtle fill backdrop so the hatch reads against dark backgrounds
  ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},0.06)`;
  ctx.fill(path);

  ctx.save();
  ctx.clip(path);

  const spacing = 20 / s; // wide spacing, bolder strokes
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.5 / s;
  ctx.globalAlpha = 0.8;

  // Forward diagonals (\)
  for (let d = -1000; d < 2000; d += spacing) {
    ctx.beginPath();
    ctx.moveTo(d, 0);
    ctx.lineTo(d + 1000, 1000);
    ctx.stroke();
  }

  // Back diagonals (/)
  ctx.globalAlpha = 0.6; // lighter second pass for depth
  for (let d = -1000; d < 2000; d += spacing) {
    ctx.beginPath();
    ctx.moveTo(d + 1000, 0);
    ctx.lineTo(d, 1000);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.restore();

  // Soft glow outline
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.2 / s;
  ctx.globalAlpha = 0.4;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 14;
  ctx.stroke(path);
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// 4. Grid — horizontal + vertical grid clipped to silhouette
// ---------------------------------------------------------------------------

function drawGrid(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
): void {
  ctx.save();
  ctx.clip(path);

  const spacing = 10 / s; // tighter grid
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1 / s;
  ctx.globalAlpha = 0.6;

  // Horizontal
  for (let y = 0; y < 1000; y += spacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1000, y);
    ctx.stroke();
  }

  // Vertical
  for (let x = 0; x < 1000; x += spacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 1000);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.restore();

  // Soft outline
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1 / s;
  ctx.globalAlpha = 0.4;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 12;
  ctx.stroke(path);
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// 5. Ghost — low-alpha translucent fill with strong glow
// ---------------------------------------------------------------------------

function drawGhost(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
  inHitlag: boolean,
): void {
  // Translucent fill — visible but see-through
  const fillAlpha = inHitlag ? 0.5 : 0.25;
  ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${fillAlpha})`;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 20;
  ctx.fill(path);

  // Second glow layer for bloom
  ctx.shadowBlur = 35;
  ctx.fill(path);

  // Third pass — big soft halo
  ctx.globalAlpha = 0.3;
  ctx.shadowBlur = 50;
  ctx.fill(path);
  ctx.globalAlpha = 1;

  // Soft outline
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 0.8 / s;
  ctx.globalAlpha = 0.4;
  ctx.stroke(path);
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// 6. Gradient — radial gradient fill from center
// ---------------------------------------------------------------------------

function drawGradient(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
): void {
  // Hot radial gradient — bright core fading to edges
  const grad = ctx.createRadialGradient(500, 400, 0, 500, 400, 450);
  grad.addColorStop(0, `rgba(255,255,255,0.9)`);
  grad.addColorStop(0.15, `rgba(${col.r},${col.g},${col.b},0.8)`);
  grad.addColorStop(0.5, `rgba(${col.r},${col.g},${col.b},0.4)`);
  grad.addColorStop(0.85, `rgba(${col.r},${col.g},${col.b},0.12)`);
  grad.addColorStop(1, `rgba(${col.r},${col.g},${col.b},0.03)`);

  ctx.fillStyle = grad;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 12;
  ctx.fill(path);

  // Soft outline
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 0.8 / s;
  ctx.globalAlpha = 0.4;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 12;
  ctx.stroke(path);
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// 7. Stipple — dot pattern clipped to silhouette
// ---------------------------------------------------------------------------

function drawStipple(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
): void {
  ctx.save();
  ctx.clip(path);

  const spacing = 7 / s; // dense dot field
  const dotRadius = 2 / s;
  ctx.fillStyle = strokeColor;
  ctx.globalAlpha = 0.7;

  // Staggered dot grid (hexagonal packing)
  let row = 0;
  for (let y = 0; y < 1000; y += spacing) {
    const offset = (row % 2) * (spacing / 2);
    for (let x = offset; x < 1000; x += spacing) {
      ctx.beginPath();
      ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
      ctx.fill();
    }
    row++;
  }

  ctx.globalAlpha = 1;
  ctx.restore();

  // Soft outline
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1 / s;
  ctx.globalAlpha = 0.4;
  ctx.shadowColor = col.glow;
  ctx.shadowBlur = 12;
  ctx.stroke(path);
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// 8. Hologram — ghost + grid + scanlines layered together
// ---------------------------------------------------------------------------

function drawHologram(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  col: PlayerColors,
  s: number,
  strokeColor: string,
  inHitlag: boolean,
): void {
  // Layer 1: Ghost fill (translucent body + heavy bloom)
  // Use full-strength glow color (col.glow is only 0.35 alpha)
  const glowBright = `rgba(${col.r},${col.g},${col.b},0.8)`;
  const fillAlpha = inHitlag ? 0.5 : 0.22;
  ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${fillAlpha})`;
  ctx.shadowColor = glowBright;
  ctx.shadowBlur = 25;
  ctx.fill(path);
  ctx.shadowBlur = 45;
  ctx.fill(path);
  ctx.globalAlpha = 0.5;
  ctx.shadowBlur = 70;
  ctx.fill(path);
  ctx.globalAlpha = 1;
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;

  // Layer 2: Grid (clipped)
  ctx.save();
  ctx.clip(path);

  const gridSpacing = 12 / s;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 0.8 / s;
  ctx.globalAlpha = 0.35;

  for (let y = 0; y < 1000; y += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1000, y);
    ctx.stroke();
  }
  for (let x = 0; x < 1000; x += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 1000);
    ctx.stroke();
  }

  // Layer 3: Scanlines on top (tighter, brighter)
  const scanSpacing = 5 / s;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.2 / s;
  ctx.globalAlpha = 0.45;

  for (let y = 0; y < 1000; y += scanSpacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1000, y);
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
  ctx.restore();

  // Outline — jittered static passes
  const jitterAmt = 3 / s;
  const passes = 4;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.2 / s;
  ctx.shadowColor = glowBright;
  ctx.shadowBlur = 10;

  for (let i = 0; i < passes; i++) {
    const jx = (Math.random() - 0.5) * jitterAmt;
    const jy = (Math.random() - 0.5) * jitterAmt;
    ctx.save();
    ctx.translate(jx, jy);
    ctx.globalAlpha = 0.35;
    ctx.stroke(path);
    ctx.restore();
  }

  // One clean pass on top for readable shape
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 0.8 / s;
  ctx.stroke(path);
  ctx.globalAlpha = 1;
}
