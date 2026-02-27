/**
 * LiveEngine — receives VizFrames over WebSocket and drives playback.
 *
 * Drop-in companion to PlaybackEngine. Instead of loading a static array,
 * it connects to a crank WS server and appends frames as they arrive.
 * Maintains a rolling buffer for trail/scrub support.
 *
 * Protocol matches viz/visualizer-juicy.html (lines 2686-2707):
 * each WS message = JSON-encoded VizFrame (or array of frames).
 */

import { VizFrame } from './types';
import {
  resetJuiceState, detectTransitionsAndTriggerJuice, updateJuice,
  setCameraImmediate, playerTrails, hitFreezeFrames,
} from './juice';
import { fetchAnimations } from './animations';

export type LiveConnectionState = 'idle' | 'connecting' | 'live' | 'disconnected' | 'error';

const BUFFER_SIZE = 3600; // Keep ~60s of frames at 60fps
const TRAIL_LENGTH = 12;

export class LiveEngine {
  frames: VizFrame[] = [];
  currentFrame = 0;
  playing = true; // Live mode is always "playing"
  readonly playSpeed = 1;
  readonly speedIdx = 3;

  connectionState: LiveConnectionState = 'idle';
  private ws: WebSocket | null = null;
  private url: string = '';
  private onFrame: ((frame: VizFrame, index: number) => void) | null = null;
  private onConnectionChange: ((state: LiveConnectionState) => void) | null = null;
  private animId: number | null = null;
  private lastRenderFrame = -1;
  private loadedChars = new Set<number>();

  setOnFrame(cb: (frame: VizFrame, index: number) => void) {
    this.onFrame = cb;
  }

  setOnConnectionChange(cb: (state: LiveConnectionState) => void) {
    this.onConnectionChange = cb;
  }

  get totalFrames() {
    return this.frames.length;
  }

  get currentVizFrame(): VizFrame | null {
    return this.frames[this.currentFrame] ?? null;
  }

  connect(url: string) {
    this.url = url;
    this.disconnect();
    this.frames = [];
    this.currentFrame = 0;
    this.lastRenderFrame = -1;
    resetJuiceState();

    this.setConnectionState('connecting');

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.setConnectionState('live');
        console.log('[LiveEngine] Connected to', url);
        this.startRenderLoop();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const newFrames: VizFrame[] = Array.isArray(data) ? data : [data];

          for (const frame of newFrames) {
            if (!frame.players || frame.players.length < 2) continue;
            this.frames.push(frame);
            // Preload character animations on first sight
            for (const p of frame.players) {
              if (p.character != null && !this.loadedChars.has(p.character)) {
                this.loadedChars.add(p.character);
                fetchAnimations(p.character);
              }
            }
          }

          // Trim buffer if too large
          if (this.frames.length > BUFFER_SIZE) {
            const excess = this.frames.length - BUFFER_SIZE;
            this.frames.splice(0, excess);
          }

          // Always track the latest frame
          this.currentFrame = this.frames.length - 1;
        } catch (e) {
          console.warn('[LiveEngine] Failed to parse frame:', e);
        }
      };

      this.ws.onclose = () => {
        this.setConnectionState('disconnected');
        console.log('[LiveEngine] Disconnected');
      };

      this.ws.onerror = () => {
        this.setConnectionState('error');
        console.error('[LiveEngine] WebSocket error');
      };
    } catch (e) {
      console.error('[LiveEngine] Failed to connect:', e);
      this.setConnectionState('error');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stopRenderLoop();
    this.setConnectionState('idle');
  }

  /** PlaybackEngine compat — no-ops for live mode */
  loadFrames(_frames: VizFrame[]) { /* no-op in live mode */ }
  play() { /* always playing */ }
  pause() { this.stopRenderLoop(); }
  togglePlay() { /* no-op */ }
  stepFrame(_delta: number) { /* no-op in live mode */ }
  seekTo(_frameIdx: number) { /* no-op in live mode */ }
  cycleSpeed(_direction: 1 | -1 = 1) { /* no-op — live is always 1x */ }

  destroy() {
    this.disconnect();
    this.onFrame = null;
    this.onConnectionChange = null;
  }

  private setConnectionState(state: LiveConnectionState) {
    this.connectionState = state;
    this.onConnectionChange?.(state);
  }

  private startRenderLoop() {
    this.stopRenderLoop();
    const tick = () => {
      this.animId = requestAnimationFrame(tick);
      this.renderLatest();
    };
    this.animId = requestAnimationFrame(tick);
  }

  private stopRenderLoop() {
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  private renderLatest() {
    if (!this.frames.length) return;

    const idx = this.currentFrame;
    if (idx === this.lastRenderFrame) return; // No new frame yet

    // Process all frames between last render and current (in case WS batched)
    const start = Math.max(0, this.lastRenderFrame + 1);
    for (let f = start; f <= idx; f++) {
      const frame = this.frames[f];
      if (!frame) continue;
      detectTransitionsAndTriggerJuice(frame);
      updateJuice(frame);

      // Maintain trails
      for (let p = 0; p < 2; p++) {
        playerTrails[p].push({ x: frame.players[p].x, y: frame.players[p].y });
        if (playerTrails[p].length > TRAIL_LENGTH) playerTrails[p].shift();
      }
    }

    this.lastRenderFrame = idx;

    // Emit to renderer
    const frame = this.frames[idx];
    if (frame && this.onFrame) {
      try {
        this.onFrame(frame, idx);
      } catch (e) {
        console.error(`[LiveEngine] render error at frame ${idx}:`, e);
      }
    }
  }
}
