/**
 * Capsule renderer — draws the rounded-rect "pill" body + facing triangle.
 * Ported from viz/visualizer-juicy.html drawCapsule (L1761-1801).
 */

import type { VizPlayerFrame, PlayerColors } from '../types';
import { CAPSULE_GAME_H, CAPSULE_GAME_W } from '../constants';
import { squashState } from '../juice';

export function drawCapsule(
  ctx: CanvasRenderingContext2D,
  p: VizPlayerFrame,
  idx: number,
  px: number,
  py: number,
  col: PlayerColors,
  glowColor: string,
  glowRadius: number,
  capsuleColor: string,
  gameScaleX: (u: number) => number,
  gameScaleY: (u: number) => number,
): { capsuleH: number; capsuleW: number; bodyTop: number; bodyBot: number } {
  const capsuleH = gameScaleY(CAPSULE_GAME_H);
  const capsuleW = gameScaleX(CAPSULE_GAME_W);
  const ss = squashState[idx];
  const bodyTop = py - capsuleH;
  const bodyBot = py;

  // --- Squash/stretch transform around capsule center ---
  ctx.save();
  ctx.translate(px, py - capsuleH / 2);
  ctx.scale(ss.scaleX, ss.scaleY);
  ctx.translate(-px, -(py - capsuleH / 2));

  // --- Glow shadow ---
  ctx.shadowColor = glowColor;
  ctx.shadowBlur = glowRadius;

  // --- Capsule body: two semicircular arcs joined by straight sides ---
  ctx.fillStyle = capsuleColor;
  ctx.beginPath();
  ctx.moveTo(px - capsuleW, bodyTop + capsuleW);
  ctx.arc(px, bodyTop + capsuleW, capsuleW, Math.PI, 0);
  ctx.lineTo(px + capsuleW, bodyBot - capsuleW);
  ctx.arc(px, bodyBot - capsuleW, capsuleW, 0, Math.PI);
  ctx.closePath();
  ctx.fill();

  // --- Clear shadow, restore transform ---
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.restore();

  // --- Facing triangle indicator ---
  const fDir = p.facing ? 1 : -1;
  const triOff = gameScaleX(1);
  const triLen = gameScaleX(2);
  const triHalf = gameScaleY(1.2);
  ctx.fillStyle = capsuleColor;
  ctx.beginPath();
  ctx.moveTo(px + fDir * (capsuleW + triOff), py - capsuleH / 2);
  ctx.lineTo(px + fDir * (capsuleW + triOff + triLen), py - capsuleH / 2 - triHalf);
  ctx.lineTo(px + fDir * (capsuleW + triOff + triLen), py - capsuleH / 2 + triHalf);
  ctx.closePath();
  ctx.fill();

  return { capsuleH, capsuleW, bodyTop, bodyBot };
}
