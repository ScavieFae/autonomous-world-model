/**
 * Shared frame renderer — draws everything from blast zones to particles.
 * Ported from viz/visualizer-juicy.html drawFrame (L1804-2222).
 *
 * The capsule/data render modes use drawCapsule from ./capsule.
 * Wire mode delegates player body drawing to an optional callback.
 */

import type { VizFrame, VizPlayerFrame, RenderMode, PlayerColors, CharacterFillMode } from '../types';
import { PLAYER_COLORS } from '../types';
import {
  STAGE,
  CAPSULE_GAME_H,
  CAPSULE_GAME_W,
  actionCategory,
  actionName,
  CATEGORY_COLORS,
} from '../constants';
import {
  shakeX,
  shakeY,
  playerTrails,
  impactFlashes,
  jumpLines,
  hitLabels,
  comboState,
  blastFlashLife,
  blastFlashDir,
  camCenterX,
  camCenterY,
  camZoom,
} from '../juice';
import { drawAmbientParticles, drawParticles } from '../particles';
import { drawCapsule } from './capsule';

// ---------------------------------------------------------------------------
// Module-level view bounds — updated each frame by updateCameraView()
// ---------------------------------------------------------------------------

let viewLeft = STAGE.camera.left;
let viewRight = STAGE.camera.right;
let viewTop = STAGE.camera.top;
let viewBottom = STAGE.camera.bottom;

// ---------------------------------------------------------------------------
// Coordinate helpers (use module view bounds)
// ---------------------------------------------------------------------------

/** Convert game-space coordinates to screen-space pixel coordinates. */
export function gameToScreen(gx: number, gy: number): [number, number] {
  const sx = ((gx - viewLeft) / (viewRight - viewLeft)) * _canvasW;
  const sy = ((viewTop - gy) / (viewTop - viewBottom)) * _canvasH;
  return [sx, sy];
}

/** Scale a horizontal distance in game units to screen pixels. */
export function gameScaleX(units: number): number {
  return (units / (viewRight - viewLeft)) * _canvasW;
}

/** Scale a vertical distance in game units to screen pixels. */
export function gameScaleY(units: number): number {
  return (units / (viewTop - viewBottom)) * _canvasH;
}

// Cached canvas dimensions — set at the start of each drawFrame call
let _canvasW = 1;
let _canvasH = 1;

// ---------------------------------------------------------------------------
// Camera view update
// ---------------------------------------------------------------------------

function updateCameraView(): void {
  const baseHalfW = (STAGE.camera.right - STAGE.camera.left) / 2;
  const baseHalfH = (STAGE.camera.top - STAGE.camera.bottom) / 2;
  const halfW = baseHalfW / camZoom;
  const halfH = baseHalfH / camZoom;
  viewLeft = camCenterX - halfW;
  viewRight = camCenterX + halfW;
  viewTop = camCenterY + halfH;
  viewBottom = camCenterY - halfH;
}

// ---------------------------------------------------------------------------
// drawFrame — the main per-frame render entry point
// ---------------------------------------------------------------------------

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  frame: VizFrame,
  canvasW: number,
  canvasH: number,
  renderMode: RenderMode,
  characterFill: CharacterFillMode,
  drawWirePlayer?: (
    ctx: CanvasRenderingContext2D,
    p: VizPlayerFrame,
    idx: number,
    px: number,
    py: number,
    col: PlayerColors,
    gameScaleX: (u: number) => number,
    gameScaleY: (u: number) => number,
    gameToScreen: (gx: number, gy: number) => [number, number],
  ) => void,
  drawCharPlayer?: (
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
  ) => boolean,
): void {
  _canvasW = canvasW;
  _canvasH = canvasH;

  // 1. Update camera view bounds
  updateCameraView();

  ctx.save();

  // 2. Apply screen shake translation
  ctx.translate(shakeX, shakeY);

  // 3. Clear canvas
  ctx.clearRect(-20, -20, canvasW + 40, canvasH + 40);

  // 4. Ambient background particles
  drawAmbientParticles(ctx, canvasW, canvasH);

  // -----------------------------------------------------------------------
  // 5. Blast zone (dashed rect)
  // -----------------------------------------------------------------------
  const [bzl, bzt] = gameToScreen(STAGE.blastzone.left, STAGE.blastzone.top);
  const [bzr, bzb] = gameToScreen(STAGE.blastzone.right, STAGE.blastzone.bottom);
  ctx.strokeStyle = 'rgba(244,63,94,0.1)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(bzl, bzt, bzr - bzl, bzb - bzt);
  ctx.setLineDash([]);

  // 6. Blast zone flash on death
  if (blastFlashLife > 0) {
    const flashAlpha = (blastFlashLife / 10) * 0.4;
    ctx.globalAlpha = flashAlpha;
    ctx.fillStyle = '#f43f5e';
    const thickness = 8;
    if (blastFlashDir === 'left') {
      ctx.fillRect(bzl - thickness, bzt, thickness * 3, bzb - bzt);
    } else if (blastFlashDir === 'right') {
      ctx.fillRect(bzr - thickness * 2, bzt, thickness * 3, bzb - bzt);
    } else if (blastFlashDir === 'top') {
      ctx.fillRect(bzl, bzt - thickness, bzr - bzl, thickness * 3);
    } else if (blastFlashDir === 'bottom') {
      ctx.fillRect(bzl, bzb - thickness * 2, bzr - bzl, thickness * 3);
    }
    ctx.globalAlpha = 1;
  }

  // -----------------------------------------------------------------------
  // 7. Stage platform (gradient + line + edge lines)
  // -----------------------------------------------------------------------
  const [sl, sy] = gameToScreen(STAGE.ground.x1, STAGE.ground.y);
  const [sr] = gameToScreen(STAGE.ground.x2, STAGE.ground.y);

  const grd = ctx.createLinearGradient(sl, sy - 20, sl, sy + 8);
  grd.addColorStop(0, 'transparent');
  grd.addColorStop(0.7, 'rgba(255,255,255,0.03)');
  grd.addColorStop(1, 'rgba(255,255,255,0.06)');
  ctx.fillStyle = grd;
  ctx.fillRect(sl, sy - 20, sr - sl, 28);

  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(sl, sy);
  ctx.lineTo(sr, sy);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(sl, sy); ctx.lineTo(sl, sy + 60);
  ctx.moveTo(sr, sy); ctx.lineTo(sr, sy + 60);
  ctx.stroke();

  // -----------------------------------------------------------------------
  // 8. Subtle grid lines
  // -----------------------------------------------------------------------
  ctx.strokeStyle = 'rgba(255,255,255,0.02)';
  ctx.lineWidth = 1;
  for (let gx = -150; gx <= 150; gx += 50) {
    const [sx] = gameToScreen(gx, 0);
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, canvasH);
    ctx.stroke();
  }
  for (let gy = -100; gy <= 100; gy += 50) {
    const [, sy2] = gameToScreen(0, gy);
    ctx.beginPath();
    ctx.moveTo(0, sy2);
    ctx.lineTo(canvasW, sy2);
    ctx.stroke();
  }

  // 9. Center marker
  const [cx, cy] = gameToScreen(0, 0);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.beginPath();
  ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy);
  ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4);
  ctx.stroke();

  // -----------------------------------------------------------------------
  // 10. Jump lines
  // -----------------------------------------------------------------------
  for (const jl of jumpLines) {
    const t = jl.life / jl.maxLife;
    ctx.globalAlpha = t * 0.6;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    for (const off of jl.offsets) {
      const [jx, jy] = gameToScreen(jl.x + off, jl.y);
      ctx.beginPath();
      ctx.moveTo(jx, jy);
      ctx.lineTo(jx, jy + 12 * t);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // -----------------------------------------------------------------------
  // 11. Draw players
  // -----------------------------------------------------------------------
  frame.players.forEach((p: VizPlayerFrame, idx: number) => {
    const col = PLAYER_COLORS[idx as 0 | 1];
    const [px, py] = gameToScreen(p.x, p.y);
    const cat = actionCategory(p.action_state);
    const catColor = CATEGORY_COLORS[cat];

    // === Momentum trails ===
    const trail = playerTrails[idx];
    if (trail.length > 1) {
      for (let ti = 0; ti < trail.length - 1; ti++) {
        const frac = ti / trail.length;
        const alpha = frac * 0.25;
        const tp = trail[ti];
        const [tx, ty] = gameToScreen(tp.x, tp.y);

        ctx.globalAlpha = alpha;
        ctx.fillStyle = col.main;
        const cH = gameScaleY(CAPSULE_GAME_H * (0.5 + frac * 0.5));
        const cW = gameScaleX(CAPSULE_GAME_W * (0.5 + frac * 0.5));
        ctx.beginPath();
        ctx.moveTo(tx - cW, ty - cH + cW);
        ctx.arc(tx, ty - cH + cW, cW, Math.PI, 0);
        ctx.lineTo(tx + cW, ty - cW);
        ctx.arc(tx, ty - cW, cW, 0, Math.PI);
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // === Shadow / ground indicator ===
    if (!p.on_ground && p.x >= STAGE.ground.x1 && p.x <= STAGE.ground.x2) {
      const [, groundY] = gameToScreen(p.x, STAGE.ground.y);
      ctx.strokeStyle = col.dim;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(px, py); ctx.lineTo(px, groundY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col.dim;
      const shadowW = gameScaleX(3);
      const shadowH = gameScaleY(0.8);
      ctx.beginPath();
      ctx.ellipse(px, groundY, shadowW, shadowH, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // === Velocity arrow (cyan) ===
    const vx = p.on_ground ? p.speed_ground_x : p.speed_air_x;
    const vy = p.speed_y;
    const vMag = Math.sqrt(vx * vx + vy * vy);
    if (vMag > 0.3) {
      const ax = px + gameScaleX(vx * 3);
      const ay = py - gameScaleY(vy * 3);
      ctx.strokeStyle = 'rgba(6,182,212,0.6)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(ax, ay);
      ctx.stroke();
      const angle = Math.atan2(ay - py, ax - px);
      const arrowHead = Math.max(4, gameScaleX(1.5));
      ctx.fillStyle = 'rgba(6,182,212,0.6)';
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - arrowHead * Math.cos(angle - 0.4), ay - arrowHead * Math.sin(angle - 0.4));
      ctx.lineTo(ax - arrowHead * Math.cos(angle + 0.4), ay - arrowHead * Math.sin(angle + 0.4));
      ctx.fill();
    }

    // === Knockback arrow (red) ===
    const kMag = Math.sqrt(p.speed_attack_x ** 2 + p.speed_attack_y ** 2);
    if (kMag > 0.3) {
      const kx = px + gameScaleX(p.speed_attack_x * 3);
      const ky = py - gameScaleY(p.speed_attack_y * 3);
      ctx.strokeStyle = 'rgba(244,63,94,0.7)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(kx, ky);
      ctx.stroke();
      const angle = Math.atan2(ky - py, kx - px);
      const arrowHead = Math.max(4, gameScaleX(1.5));
      ctx.fillStyle = 'rgba(244,63,94,0.7)';
      ctx.beginPath();
      ctx.moveTo(kx, ky);
      ctx.lineTo(kx - arrowHead * Math.cos(angle - 0.4), ky - arrowHead * Math.sin(angle - 0.4));
      ctx.lineTo(kx - arrowHead * Math.cos(angle + 0.4), ky - arrowHead * Math.sin(angle + 0.4));
      ctx.fill();
    }

    // === Shield bubble ===
    if (p.action_state >= 178 && p.action_state <= 182) {
      const shieldRad = gameScaleY(5 + (p.shield_strength / 60) * 4);
      ctx.strokeStyle = col.glow;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.4 + (p.shield_strength / 60) * 0.4;
      ctx.beginPath();
      ctx.arc(px, py - gameScaleY(2), shieldRad, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = col.dim;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // === Percent heat glow (color shift green->amber->magenta->white) ===
    const pct = p.percent;
    let glowColor = col.glow;
    let glowRadius = cat === 'attack' ? 16 : (cat === 'damage' ? 12 : 8);
    let capsuleColor = col.main;

    if (pct > 120) {
      const pulse = 1 + Math.sin(performance.now() * 0.008) * 0.3;
      glowColor = `rgba(244, 63, 94, ${0.6 * pulse})`;
      glowRadius = Math.max(glowRadius, 20) * pulse;
      const brightness = 1 + (pct - 120) / 200;
      capsuleColor = `rgb(${Math.min(255, Math.round(col.r * brightness))}, ${Math.min(255, Math.round(col.g * 0.6))}, ${Math.min(255, Math.round(col.b * 0.5))})`;
    } else if (pct > 80) {
      const pulse = 1 + Math.sin(performance.now() * 0.006) * 0.15;
      const t = (pct - 80) / 40;
      glowColor = `rgba(${Math.round(245 * t + col.r * (1 - t))}, ${Math.round(120 * t + col.g * (1 - t))}, ${Math.round(30 * t + col.b * (1 - t))}, ${0.45 * pulse})`;
      glowRadius = Math.max(glowRadius, 14) * pulse;
    } else if (pct > 30) {
      const t = (pct - 30) / 50;
      glowColor = `rgba(${Math.round(col.r + (50 * t))}, ${Math.round(col.g * (1 - t * 0.2))}, ${Math.round(col.b * (1 - t * 0.3))}, 0.4)`;
      glowRadius = Math.max(glowRadius, 10 + t * 4);
    }

    // === Draw player body ===
    let drewWire = false;
    let drewChar = false;
    let capsuleGeom: { capsuleH: number; capsuleW: number; bodyTop: number; bodyBot: number } | null = null;

    if (renderMode === 'wire' && drawWirePlayer) {
      drawWirePlayer(ctx, p, idx, p.x, p.y, col, gameScaleX, gameScaleY, gameToScreen);
      drewWire = true;
    }

    if (!drewWire && (renderMode === 'capsule' || renderMode === 'xray') && drawCharPlayer) {
      const fillMode = renderMode === 'xray' ? characterFill : 'solid' as const;
      drewChar = drawCharPlayer(ctx, p, idx, p.x, p.y, col, gameScaleX, gameScaleY, gameToScreen, fillMode);
    }

    if (!drewWire && !drewChar) {
      capsuleGeom = drawCapsule(
        ctx, p, idx, px, py, col,
        glowColor, glowRadius, capsuleColor,
        gameScaleX, gameScaleY,
      );
    }

    // Fallback geometry values for wire mode (approximated from capsule dimensions)
    const pxPerUnit = canvasW / (viewRight - viewLeft);
    const capsuleH = capsuleGeom ? capsuleGeom.capsuleH : gameScaleY(CAPSULE_GAME_H);
    const capsuleW = capsuleGeom ? capsuleGeom.capsuleW : gameScaleX(CAPSULE_GAME_W);
    const bodyTop = capsuleGeom ? capsuleGeom.bodyTop : py - capsuleH;
    const bodyBot = capsuleGeom ? capsuleGeom.bodyBot : py;

    // === Hitlag flash ===
    if (p.hitlag > 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 2;
      const flashR = drewWire
        ? (28 + p.hitlag * 2)
        : (capsuleH * 0.8 + p.hitlag * gameScaleY(1));
      const flashCenterY = drewWire ? (py - pxPerUnit * 5) : (py - capsuleH / 2);
      ctx.beginPath();
      ctx.arc(px, flashCenterY, flashR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // === Damage sparks ===
    if (cat === 'damage') {
      ctx.strokeStyle = 'rgba(245,158,11,0.4)';
      ctx.lineWidth = 1;
      const sparkCenterY = drewWire ? (py - pxPerUnit * 5) : (py - capsuleH / 2);
      const sparkSpreadX = drewWire ? Math.max(18, pxPerUnit * 8) : (capsuleW * 4);
      const sparkSpreadY = drewWire ? Math.max(10, pxPerUnit * 4) : capsuleH;
      for (let s = 0; s < 3; s++) {
        const sparkX = px + (Math.random() - 0.5) * sparkSpreadX;
        const sparkY = sparkCenterY + (Math.random() - 0.5) * sparkSpreadY;
        const sparkLen = Math.max(2, gameScaleX(1));
        ctx.beginPath();
        ctx.moveTo(sparkX - sparkLen, sparkY);
        ctx.lineTo(sparkX + sparkLen, sparkY);
        ctx.stroke();
      }
    }

    // === Player label (P1/P2) ===
    ctx.fillStyle = col.main;
    ctx.font = '600 10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`P${idx + 1}`, px, drewWire ? (py - pxPerUnit * 12) : (bodyTop - 4));

    // === Action state label (color-coded by category) ===
    const actionLabel = actionName(p.action_state);
    ctx.fillStyle = catColor;
    ctx.font = '500 9px monospace';
    ctx.fillText(actionLabel, px, (drewWire ? py : bodyBot) + 12);

    // === Percent near player ===
    ctx.fillStyle = p.percent > 100 ? '#f43f5e' : (p.percent > 60 ? '#f59e0b' : 'rgba(255,255,255,0.7)');
    ctx.font = '700 11px monospace';
    ctx.fillText(`${Math.round(p.percent)}%`, px, (drewWire ? py : bodyBot) + 24);
  });

  // -----------------------------------------------------------------------
  // 12. Impact flashes (starburst + ring for deaths)
  // -----------------------------------------------------------------------
  for (const flash of impactFlashes) {
    const t = flash.life / flash.maxLife;
    const [fx, fy] = gameToScreen(flash.x, flash.y);

    if (flash.ring) {
      // Death ring: expanding circle
      const radius = (1 - t) * 80;
      ctx.globalAlpha = t * 0.7;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3 * t;
      ctx.beginPath();
      ctx.arc(fx, fy, radius, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      // Hit starburst
      const radius = (1 - t) * 30 + 5;
      ctx.globalAlpha = t * 0.9;

      // White center radial gradient
      const starGrad = ctx.createRadialGradient(fx, fy, 0, fx, fy, radius);
      starGrad.addColorStop(0, 'rgba(255,255,255,0.9)');
      starGrad.addColorStop(0.3, 'rgba(255,255,200,0.6)');
      starGrad.addColorStop(1, 'rgba(255,255,200,0)');
      ctx.fillStyle = starGrad;
      ctx.beginPath();
      ctx.arc(fx, fy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Star spikes
      ctx.strokeStyle = `rgba(255, 255, 220, ${t * 0.7})`;
      ctx.lineWidth = 2 * t;
      const spikeCount = 6;
      for (let s = 0; s < spikeCount; s++) {
        const angle = (s / spikeCount) * Math.PI * 2 + (1 - t) * 0.5;
        const spikeLen = radius * 1.5;
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.lineTo(fx + Math.cos(angle) * spikeLen, fy + Math.sin(angle) * spikeLen);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  // -----------------------------------------------------------------------
  // 13. Hit labels (floating text)
  // -----------------------------------------------------------------------
  for (const hl of hitLabels) {
    const t = hl.life / hl.maxLife;
    const [hx, hy] = gameToScreen(hl.x, hl.y);
    const drift = (1 - t) * 30;
    ctx.globalAlpha = t;
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(hl.text, hx, hy - 20 - drift);
    ctx.globalAlpha = 1;
  }

  // -----------------------------------------------------------------------
  // 14. Combo counters (pill with "N HIT")
  // -----------------------------------------------------------------------
  for (let idx = 0; idx < 2; idx++) {
    const combo = comboState[idx];
    if (combo.count >= 2 && combo.timer > 0) {
      const col = PLAYER_COLORS[idx as 0 | 1];
      const p = frame.players[idx];
      const [combX, combY] = gameToScreen(p.x, p.y);
      const fadeAlpha = Math.min(1, combo.timer / 20);
      const scale = combo.displayScale;

      ctx.save();
      ctx.translate(combX, combY - 50);
      ctx.scale(scale, scale);
      ctx.globalAlpha = fadeAlpha;

      // Background pill
      const text = `${combo.count} HIT`;
      ctx.font = '800 14px monospace';
      const textW = ctx.measureText(text).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      const pillW = textW + 16;
      const pillH = 22;
      ctx.beginPath();
      ctx.roundRect(-pillW / 2, -pillH / 2, pillW, pillH, 4);
      ctx.fill();

      // Text
      ctx.fillStyle = col.main;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 0, 0);

      ctx.restore();
      ctx.globalAlpha = 1;
    }
  }

  // -----------------------------------------------------------------------
  // 15. Particles (sparks, dust, death burst)
  // -----------------------------------------------------------------------
  drawParticles(ctx);

  // -----------------------------------------------------------------------
  // 16. Legend text
  // -----------------------------------------------------------------------
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('cyan arrow = velocity', 10, canvasH - 30);
  ctx.fillText('red arrow = knockback', 10, canvasH - 16);

  // X-ray fill mode label (top-right)
  if (renderMode === 'xray') {
    const FILL_LABELS: Record<string, string> = {
      stroke: 'STROKE', scanline: 'SCANLINE', hatch: 'HATCH',
      grid: 'GRID', ghost: 'GHOST', gradient: 'GRADIENT', stipple: 'STIPPLE',
      hologram: 'HOLOGRAM',
    };
    const label = FILL_LABELS[characterFill] ?? characterFill;
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '600 11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`X-RAY: ${label}`, canvasW - 10, 18);
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '9px monospace';
    ctx.fillText('press X to cycle', canvasW - 10, 30);
  }

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  const [l0] = gameToScreen(-100, 0);
  const [l1] = gameToScreen(0, 0);
  const [l2] = gameToScreen(100, 0);
  ctx.fillText('-100', l0, canvasH - 4);
  ctx.fillText('0', l1, canvasH - 4);
  ctx.fillText('100', l2, canvasH - 4);

  // 17. Restore shake translation
  ctx.restore();
}
