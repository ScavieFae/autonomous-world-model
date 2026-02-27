'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import StageCanvas from '@/components/canvas/StageCanvas';
import CRTOverlay from '@/components/canvas/CRTOverlay';
import PlaybackControls from '@/components/canvas/PlaybackControls';
import PlayerPanel from '@/components/player/PlayerPanel';
import MatchTicker from '@/components/arena/MatchTicker';
import BetweenSetsView from '@/components/arena/BetweenSetsView';
import PreMatchView from '@/components/arena/PreMatchView';
import PostMatchView from '@/components/arena/PostMatchView';
import { PlaybackEngine } from '@/engine/playback';
import { LiveEngine, type LiveConnectionState, type MatchLifecycleEvent } from '@/engine/live';
import { generateMockData } from '@/engine/mock-data';
import { preloadFromFrames } from '@/engine/animations';
import { setJuiceEventCallback, type JuiceEvent } from '@/engine/juice';
import { useArenaStore } from '@/stores/arena';
import type { VizFrame, Engine } from '@/engine/types';

/** Read ?live=ws://... from the URL query string. */
function getLiveUrl(): string | null {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  return params.get('live') || null;
}

const CONNECTION_LABELS: Record<LiveConnectionState, { text: string; className: string }> = {
  idle: { text: '', className: '' },
  connecting: { text: 'CONNECTING', className: 'live-badge live-badge--connecting' },
  live: { text: 'LIVE', className: 'live-badge live-badge--live' },
  disconnected: { text: 'DISCONNECTED', className: 'live-badge live-badge--disconnected' },
  error: { text: 'ERROR', className: 'live-badge live-badge--error' },
};

export default function ArenaView() {
  const liveUrl = useRef(getLiveUrl());
  const isLive = liveUrl.current !== null;

  // Create the right engine type once
  const engineRef = useRef<Engine | null>(null);
  if (!engineRef.current) {
    engineRef.current = isLive ? new LiveEngine() : new PlaybackEngine();
  }
  const engine = engineRef.current;

  const [frame, setFrame] = useState<VizFrame | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const phase = useArenaStore((s) => s.phase);
  const currentMatch = useArenaStore((s) => s.currentMatch);
  const liveConnection = useArenaStore((s) => s.liveConnection);

  const onFrameUpdate = useCallback((f: VizFrame, idx: number) => {
    setFrame(f);
    setFrameIdx(idx);
    setTotalFrames(engine.totalFrames);
  }, [engine]);

  // Live mode: connect to WS server + wire lifecycle
  useEffect(() => {
    if (!isLive || !(engine instanceof LiveEngine)) return;

    const url = liveUrl.current!;
    engine.setOnConnectionChange((state) => {
      useArenaStore.getState().setLiveConnection(state);
      // On connect, start in between-sets (waiting for match_start)
      if (state === 'live') {
        useArenaStore.getState().setPhase('between-sets');
      }
    });
    engine.setOnMatchLifecycle((event: MatchLifecycleEvent) => {
      const store = useArenaStore.getState();
      if (event.type === 'match_start') {
        engine.resetForNewMatch();
        store.handleMatchStart(event.msg);
      } else if (event.type === 'match_end') {
        store.handleMatchEnd(event.msg);
      }
    });
    engine.connect(url);

    return () => {
      engine.setOnMatchLifecycle(null);
      engine.disconnect();
    };
  }, [engine, isLive]);

  // Playback mode: load replay or mock data
  useEffect(() => {
    if (isLive) return; // Live mode handles its own data

    let cancelled = false;

    function startWithFrames(data: VizFrame[]) {
      if (cancelled) return;
      preloadFromFrames(data);
      engine.loadFrames(data);
      engine.play();
    }

    fetch('/replay.json')
      .then((r) => {
        if (!r.ok) throw new Error('no replay');
        return r.json();
      })
      .then((raw: VizFrame[] | { meta?: unknown; frames: VizFrame[] }) => {
        if (cancelled) return;
        // Crank outputs {meta, stage_geometry, frames} — unwrap if needed
        const data: VizFrame[] = Array.isArray(raw) ? raw : raw.frames;
        useArenaStore.getState().setPhase('live');
        startWithFrames(data);
      })
      .catch(() => {
        // No replay file — use mock data + demo cycle
        if (cancelled) return;
        startWithFrames(generateMockData('neutral'));
        useArenaStore.getState().runDemoCycle();
      });

    return () => {
      cancelled = true;
      engine.pause();
    };
  }, [engine, isLive]);

  // Wire up juice event callback to push match events
  useEffect(() => {
    const p1Name = currentMatch?.p1.username ?? 'P1';
    const p2Name = currentMatch?.p2.username ?? 'P2';

    setJuiceEventCallback((e: JuiceEvent) => {
      const names = [p1Name, p2Name];
      useArenaStore.getState().pushEvent({
        frame: frameIdx,
        type: e.type,
        attacker: names[e.attackerIdx] ?? 'unknown',
        defender: names[e.defenderIdx] ?? 'unknown',
        attackName: e.attackName,
        damage: e.damage,
        newPercent: e.newPercent,
      });
    });

    return () => setJuiceEventCallback(null);
  }, [currentMatch, frameIdx]);

  useEffect(() => {
    return () => engine.destroy();
  }, [engine]);

  const p1Name = currentMatch?.p1.username ?? 'foxmaster-9000';
  const p2Name = currentMatch?.p2.username ?? 'waveland-wizard';
  const p1Sponsor = currentMatch?.p1.sponsor;
  const p2Sponsor = currentMatch?.p2.sponsor;

  const badge = CONNECTION_LABELS[liveConnection];

  return (
    <div className="arena-outer">
      <div className="arena-stage-wrapper crt-screen">
        <StageCanvas
          scenario="neutral"
          engine={engine}
          onFrameUpdate={onFrameUpdate}
          dimmed={phase !== 'live'}
        />
        <CRTOverlay />

        {/* Live connection indicator */}
        {isLive && badge.text && (
          <div className={badge.className}>{badge.text}</div>
        )}

        {phase === 'between-sets' && <BetweenSetsView />}
        {phase === 'pre-match' && <PreMatchView />}
        {phase === 'live' && (
          <>
            {frame && (
              <div className="arena-hud-p1">
                <PlayerPanel
                  player={frame.players[0]}
                  index={0}
                  agentName={p1Name}
                  sponsor={p1Sponsor}
                  compact
                />
              </div>
            )}
            {frame && (
              <div className="arena-hud-p2">
                <PlayerPanel
                  player={frame.players[1]}
                  index={1}
                  agentName={p2Name}
                  sponsor={p2Sponsor}
                  compact
                />
              </div>
            )}
            <MatchTicker />
          </>
        )}
        {phase === 'post-match' && <PostMatchView />}
      </div>

      {phase === 'live' && !isLive && (
        <PlaybackControls
          engine={engine}
          currentFrame={frameIdx}
          totalFrames={totalFrames}
          minimal
        />
      )}
    </div>
  );
}
