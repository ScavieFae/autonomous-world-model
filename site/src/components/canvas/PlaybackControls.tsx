'use client';

import { useState, useEffect, useCallback } from 'react';
import type { PlaybackEngine } from '@/engine/playback';
import { useArenaStore } from '@/stores/arena';
import type { RenderMode, CharacterFillMode } from '@/engine/types';

interface PlaybackControlsProps {
  engine: PlaybackEngine;
  currentFrame: number;
  totalFrames: number;
  minimal?: boolean;
}

const MODE_LABELS: Record<RenderMode, string> = {
  wire: 'W',
  capsule: 'C',
  data: 'D',
  xray: 'X',
};

const FILL_SHORT: Record<CharacterFillMode, string> = {
  stroke: 'strk',
  scanline: 'scan',
  hatch: 'htch',
  grid: 'grid',
  ghost: 'ghst',
  gradient: 'grad',
  stipple: 'stip',
  hologram: 'holo',
};

export default function PlaybackControls({ engine, currentFrame, totalFrames, minimal }: PlaybackControlsProps) {
  const [playing, setPlaying] = useState(true);
  const renderMode = useArenaStore((s) => s.renderMode);
  const characterFill = useArenaStore((s) => s.characterFill);
  const setRenderMode = useArenaStore((s) => s.setRenderMode);
  const cycleCharacterFill = useArenaStore((s) => s.cycleCharacterFill);

  const togglePlay = useCallback(() => {
    engine.togglePlay();
    setPlaying(engine.playing);
  }, [engine]);

  // Sync playing state
  useEffect(() => {
    const interval = setInterval(() => {
      if (engine.playing !== playing) setPlaying(engine.playing);
    }, 100);
    return () => clearInterval(interval);
  }, [engine, playing]);

  if (minimal) {
    return (
      <div className="arena-controls arena-controls-minimal">
        <div className="btn-row" style={{ justifyContent: 'center' }}>
          {(Object.entries(MODE_LABELS) as [RenderMode, string][]).map(([mode, label]) => (
            <button
              key={mode}
              className={`ctrl-btn ${renderMode === mode ? 'active' : ''}`}
              onClick={() => {
                if (mode === 'xray' && renderMode === 'xray') {
                  cycleCharacterFill();
                } else {
                  setRenderMode(mode);
                }
              }}
              title={mode === 'xray' && renderMode === 'xray' ? `x-ray: ${characterFill}` : `${mode} mode`}
            >
              {mode === 'xray' && renderMode === 'xray' ? FILL_SHORT[characterFill] : label}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="arena-controls">
      <div className="timeline-row">
        <input
          type="range"
          className="timeline"
          min={0}
          max={Math.max(0, totalFrames - 1)}
          value={currentFrame}
          onChange={(e) => engine.seekTo(parseInt(e.target.value))}
        />
      </div>
      <div className="btn-row">
        <button className="ctrl-btn" onClick={() => engine.stepFrame(-1)} title="Previous frame">
          &#x2190;
        </button>
        <button
          className={`ctrl-btn ${playing ? 'active' : ''}`}
          onClick={togglePlay}
          title="Play/Pause"
          style={{ minWidth: 50 }}
        >
          {playing ? '\u23F8' : '\u25B6'}
        </button>
        <button className="ctrl-btn" onClick={() => engine.stepFrame(1)} title="Next frame">
          &#x2192;
        </button>
        <button className="ctrl-btn" onClick={() => engine.cycleSpeed(1)}>
          {engine.playSpeed}x
        </button>
        <span className="speed-label">{Math.round(60 * engine.playSpeed)} fps</span>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          {(Object.entries(MODE_LABELS) as [RenderMode, string][]).map(([mode, label]) => (
            <button
              key={mode}
              className={`ctrl-btn ${renderMode === mode ? 'active' : ''}`}
              onClick={() => {
                if (mode === 'xray' && renderMode === 'xray') {
                  cycleCharacterFill();
                } else {
                  setRenderMode(mode);
                }
              }}
              title={mode === 'xray' && renderMode === 'xray' ? `x-ray: ${characterFill}` : `${mode} mode`}
            >
              {mode === 'xray' && renderMode === 'xray' ? FILL_SHORT[characterFill] : label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
