'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import StageCanvas from '@/components/canvas/StageCanvas';
import CRTOverlay from '@/components/canvas/CRTOverlay';
import PlaybackControls from '@/components/canvas/PlaybackControls';
import PlayerPanel from '@/components/player/PlayerPanel';
import MatchTicker from '@/components/arena/MatchTicker';
import { PlaybackEngine } from '@/engine/playback';
import { LiveEngine, type LiveConnectionState } from '@/engine/live';
import { preloadFromFrames } from '@/engine/animations';
import { setJuiceEventCallback, type JuiceEvent } from '@/engine/juice';
import { useArenaStore } from '@/stores/arena';
import type { VizFrame, Engine } from '@/engine/types';

const DEFAULT_WS = 'ws://localhost:8765';

const CONNECTION_LABELS: Record<LiveConnectionState, { text: string; className: string }> = {
  idle: { text: '', className: '' },
  connecting: { text: 'CONNECTING', className: 'live-badge live-badge--connecting' },
  live: { text: 'LIVE', className: 'live-badge live-badge--live' },
  disconnected: { text: 'DISCONNECTED', className: 'live-badge live-badge--disconnected' },
  error: { text: 'ERROR', className: 'live-badge live-badge--error' },
};

type LiveMode = 'ws' | 'replay';

function detectMode(): { mode: LiveMode; wsUrl: string } {
  if (typeof window === 'undefined') return { mode: 'replay', wsUrl: DEFAULT_WS };
  const params = new URLSearchParams(window.location.search);
  const ws = params.get('ws');
  if (ws) return { mode: 'ws', wsUrl: ws };
  return { mode: 'replay', wsUrl: DEFAULT_WS };
}

export default function LiveArenaView() {
  const config = useRef(detectMode());
  const isWs = config.current.mode === 'ws';

  const engineRef = useRef<Engine | null>(null);
  if (!engineRef.current) {
    engineRef.current = isWs ? new LiveEngine() : new PlaybackEngine();
  }
  const engine = engineRef.current;

  const [frame, setFrame] = useState<VizFrame | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [hasData, setHasData] = useState(false);
  const liveConnection = useArenaStore((s) => s.liveConnection);

  const onFrameUpdate = useCallback((f: VizFrame, idx: number) => {
    setFrame(f);
    setFrameIdx(idx);
    setTotalFrames(engine.totalFrames);
  }, [engine]);

  // WS mode: connect to crank server
  useEffect(() => {
    if (!isWs || !(engine instanceof LiveEngine)) return;

    engine.setOnConnectionChange((state) => {
      useArenaStore.getState().setLiveConnection(state);
    });
    engine.connect(config.current.wsUrl);
    setHasData(true);

    return () => { engine.disconnect(); };
  }, [engine, isWs]);

  // Replay mode: load replay.json
  useEffect(() => {
    if (isWs) return;

    let cancelled = false;

    fetch('/replay.json')
      .then((r) => {
        if (!r.ok) throw new Error('no replay');
        return r.json();
      })
      .then((raw: VizFrame[] | { meta?: unknown; frames: VizFrame[] }) => {
        if (cancelled) return;
        const data: VizFrame[] = Array.isArray(raw) ? raw : raw.frames;
        preloadFromFrames(data);
        engine.loadFrames(data);
        engine.play();
        setHasData(true);
      })
      .catch(() => {
        // No data available
      });

    return () => {
      cancelled = true;
      engine.pause();
    };
  }, [engine, isWs]);

  // Juice events
  useEffect(() => {
    setJuiceEventCallback((e: JuiceEvent) => {
      useArenaStore.getState().pushEvent({
        frame: frameIdx,
        type: e.type,
        attacker: e.attackerIdx === 0 ? 'P1' : 'P2',
        defender: e.defenderIdx === 0 ? 'P1' : 'P2',
        attackName: e.attackName,
        damage: e.damage,
        newPercent: e.newPercent,
      });
    });
    return () => setJuiceEventCallback(null);
  }, [frameIdx]);

  useEffect(() => {
    return () => engine.destroy();
  }, [engine]);

  const badge = CONNECTION_LABELS[liveConnection];
  const showEmpty = !hasData && !isWs;

  return (
    <div className="arena-outer">
      <div className="live-header">
        <h1 className="live-title">Live Arena</h1>
        <span className="live-subtitle">
          {isWs ? `Connected to ${config.current.wsUrl}` : 'Replay mode'}
        </span>
      </div>

      <div className="arena-stage-wrapper crt-screen">
        <StageCanvas
          scenario="neutral"
          engine={engine}
          onFrameUpdate={onFrameUpdate}
          dimmed={false}
        />
        <CRTOverlay />

        {/* Connection badge */}
        {isWs && badge.text && (
          <div className={badge.className}>{badge.text}</div>
        )}

        {/* Empty state */}
        {showEmpty && (
          <div className="live-empty">
            <p>No replay data found.</p>
            <p className="live-empty-hint">
              Generate a match: <code>python -m crank.main --world-model ... --output site/public/replay.json</code>
            </p>
            <p className="live-empty-hint">
              Or stream live: <code>/live?ws=ws://localhost:8765</code>
            </p>
          </div>
        )}

        {/* HUD */}
        {frame && (
          <>
            <div className="arena-hud-p1">
              <PlayerPanel
                player={frame.players[0]}
                index={0}
                agentName="P1"
                compact
              />
            </div>
            <div className="arena-hud-p2">
              <PlayerPanel
                player={frame.players[1]}
                index={1}
                agentName="P2"
                compact
              />
            </div>
          </>
        )}
        <MatchTicker />
      </div>

      {/* Playback controls only in replay mode */}
      {!isWs && hasData && (
        <PlaybackControls
          engine={engine}
          currentFrame={frameIdx}
          totalFrames={totalFrames}
          minimal
        />
      )}

      {/* Frame counter */}
      <div className="live-frame-counter">
        Frame {frameIdx} / {totalFrames}
      </div>
    </div>
  );
}
