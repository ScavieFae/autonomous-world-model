/**
 * PlaybackEngine — manages frame playback, speed, stepping.
 * Ported from viz/visualizer-juicy.html L2330-2614.
 */

import { VizFrame } from './types';
import {
  resetJuiceState, detectTransitionsAndTriggerJuice, updateJuice,
  setCameraImmediate, playerTrails, hitFreezeFrames,
} from './juice';

export const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4] as const;
const TRAIL_LENGTH = 12;

export class PlaybackEngine {
  frames: VizFrame[] = [];
  currentFrame = 0;
  playing = false;
  speedIdx = 3;
  playSpeed = 1;

  private lastFrameTime = 0;
  private animId: number | null = null;
  private onFrame: ((frame: VizFrame, index: number) => void) | null = null;

  setOnFrame(cb: (frame: VizFrame, index: number) => void) {
    this.onFrame = cb;
  }

  loadFrames(newFrames: VizFrame[]) {
    this.frames = newFrames;
    this.currentFrame = 0;
    resetJuiceState();
    this.emitFrame();
  }

  get totalFrames() {
    return this.frames.length;
  }

  get currentVizFrame(): VizFrame | null {
    return this.frames[this.currentFrame] ?? null;
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this.lastFrameTime = performance.now();
    this.scheduleTick();
  }

  pause() {
    this.playing = false;
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  togglePlay() {
    if (this.playing) this.pause();
    else this.play();
  }

  stepFrame(delta: number) {
    if (this.playing) this.pause();
    const newFrame = Math.max(0, Math.min(this.frames.length - 1, this.currentFrame + delta));

    if (delta > 0) {
      for (let f = this.currentFrame + 1; f <= newFrame; f++) {
        detectTransitionsAndTriggerJuice(this.frames[f]);
        updateJuice(this.frames[f]);
      }
    } else if (delta < 0) {
      playerTrails[0] = [];
      playerTrails[1] = [];
    }

    this.currentFrame = newFrame;
    this.emitFrame();
  }

  seekTo(frameIdx: number) {
    resetJuiceState();
    this.currentFrame = Math.max(0, Math.min(this.frames.length - 1, frameIdx));

    // Seed trails
    const trailStart = Math.max(0, this.currentFrame - TRAIL_LENGTH);
    for (let f = trailStart; f <= this.currentFrame; f++) {
      const fr = this.frames[f];
      if (fr) {
        for (let idx = 0; idx < 2; idx++) {
          playerTrails[idx].push({ x: fr.players[idx].x, y: fr.players[idx].y });
          if (playerTrails[idx].length > TRAIL_LENGTH) playerTrails[idx].shift();
        }
      }
    }

    // Snap camera
    if (this.frames[this.currentFrame]) {
      setCameraImmediate(this.frames[this.currentFrame]);
    }

    this.emitFrame();
  }

  cycleSpeed(direction: 1 | -1 = 1) {
    this.speedIdx = (this.speedIdx + direction + SPEEDS.length) % SPEEDS.length;
    this.playSpeed = SPEEDS[this.speedIdx];
  }

  destroy() {
    this.pause();
    this.onFrame = null;
  }

  private scheduleTick() {
    this.animId = requestAnimationFrame((ts) => this.tick(ts));
  }

  private tick(ts: number) {
    if (!this.playing) return;

    try {
      // Hit freeze — updateJuice decrements the counter
      if (hitFreezeFrames > 0) {
        if (this.frames.length) {
          updateJuice(this.frames[this.currentFrame]);
        }
        this.emitFrame();
        this.scheduleTick();
        return;
      }

      const dt = ts - this.lastFrameTime;
      const frameInterval = 1000 / (60 * this.playSpeed);
      if (dt >= frameInterval) {
        this.lastFrameTime = ts - (dt % frameInterval);
        this.currentFrame++;
        if (this.currentFrame >= this.frames.length) {
          this.currentFrame = 0;
          resetJuiceState();
        }
        const frame = this.frames[this.currentFrame];
        detectTransitionsAndTriggerJuice(frame);
        updateJuice(frame);
        this.emitFrame();
      }
    } catch (e) {
      console.error('[PlaybackEngine] tick error:', e);
    }
    this.scheduleTick();
  }

  private emitFrame() {
    if (!this.frames.length) return;
    const frame = this.frames[this.currentFrame];
    if (frame && this.onFrame) {
      try {
        this.onFrame(frame, this.currentFrame);
      } catch (e) {
        console.error(`[Engine] render error at frame ${this.currentFrame}:`, e);
      }
    }
  }
}
