'use client';

import { useRef, useEffect, useCallback } from 'react';
import { PlaybackEngine } from '@/engine/playback';
import { generateMockData, type MockScenario } from '@/engine/mock-data';
import { drawFrame } from '@/engine/renderers/common';
import { drawWirePlayer } from '@/engine/renderers/wire';
import { drawCharacterPlayer } from '@/engine/renderers/character';
import { setGameToScreen } from '@/engine/juice';
import { preloadFromFrames } from '@/engine/animations';
import { useArenaStore } from '@/stores/arena';
import type { VizFrame, RenderMode, CharacterFillMode } from '@/engine/types';

interface StageCanvasProps {
  scenario?: MockScenario;
  frames?: VizFrame[];
  engine: PlaybackEngine;
  onFrameUpdate?: (frame: VizFrame, index: number) => void;
  dimmed?: boolean;
}

export default function StageCanvas({ scenario = 'neutral', frames, engine, onFrameUpdate, dimmed }: StageCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const renderMode = useArenaStore((s) => s.renderMode);
  const characterFill = useArenaStore((s) => s.characterFill);
  const renderModeRef = useRef<RenderMode>(renderMode);
  const characterFillRef = useRef<CharacterFillMode>(characterFill);
  renderModeRef.current = renderMode;
  characterFillRef.current = characterFill;

  const render = useCallback((frame: VizFrame, index: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const container = containerRef.current;
    if (!container) return;

    const w = container.clientWidth;
    const h = container.clientHeight;

    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const mode = renderModeRef.current;
    const fill = characterFillRef.current;
    const wireRenderer = mode === 'wire' ? drawWirePlayer : undefined;
    const charRenderer = (mode === 'capsule' || mode === 'xray') ? drawCharacterPlayer : undefined;
    drawFrame(ctx, frame, w, h, mode, fill, wireRenderer, charRenderer);

    onFrameUpdate?.(frame, index);
  }, [onFrameUpdate]);

  // Register render callback + preload character animations
  useEffect(() => {
    engine.setOnFrame(render);
    if (engine.frames.length > 0) {
      preloadFromFrames(engine.frames);
    }
  }, [engine, render]);

  // Resize handler
  useEffect(() => {
    function handleResize() {
      const frame = engine.currentVizFrame;
      if (frame) render(frame, engine.currentFrame);
    }
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [engine, render]);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const { setRenderMode } = useArenaStore.getState();
      switch (e.key) {
        case ' ':
          e.preventDefault();
          engine.togglePlay();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          engine.stepFrame(e.shiftKey ? -10 : -1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          engine.stepFrame(e.shiftKey ? 10 : 1);
          break;
        case '+': case '=':
          engine.cycleSpeed(1);
          break;
        case '-': case '_':
          engine.cycleSpeed(-1);
          break;
        case 'w': case 'W':
          setRenderMode('wire');
          break;
        case 'c': case 'C':
          setRenderMode('capsule');
          break;
        case 'd': case 'D':
          setRenderMode('data');
          break;
        case 'x': case 'X':
          if (useArenaStore.getState().renderMode === 'xray') {
            useArenaStore.getState().cycleCharacterFill();
          } else {
            setRenderMode('xray');
          }
          break;
        case 'f': case 'F':
          containerRef.current?.requestFullscreen?.();
          break;
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [engine]);

  return (
    <div ref={containerRef} className="arena-canvas-area" style={dimmed ? { opacity: 0.3 } : undefined}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%' }}
      />
    </div>
  );
}
