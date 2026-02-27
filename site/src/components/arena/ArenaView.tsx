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
import { generateMockData } from '@/engine/mock-data';
import { preloadFromFrames } from '@/engine/animations';
import { setJuiceEventCallback, type JuiceEvent } from '@/engine/juice';
import { useArenaStore } from '@/stores/arena';
import type { VizFrame } from '@/engine/types';

export default function ArenaView() {
  const engineRef = useRef<PlaybackEngine | null>(null);
  if (!engineRef.current) {
    engineRef.current = new PlaybackEngine();
  }
  const engine = engineRef.current;

  const [frame, setFrame] = useState<VizFrame | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const phase = useArenaStore((s) => s.phase);
  const currentMatch = useArenaStore((s) => s.currentMatch);

  const onFrameUpdate = useCallback((f: VizFrame, idx: number) => {
    setFrame(f);
    setFrameIdx(idx);
    setTotalFrames(engine.totalFrames);
  }, [engine]);

  // Load replay data (or fall back to mock) and start playback
  useEffect(() => {
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
      .then((data: VizFrame[]) => {
        if (cancelled) return;
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
  }, [engine]);

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

      {phase === 'live' && (
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
