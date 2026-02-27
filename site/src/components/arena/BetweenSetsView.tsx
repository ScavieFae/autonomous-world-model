'use client';

import { useState } from 'react';
import { useArenaStore } from '@/stores/arena';
import QuarterUpModal from '@/components/modals/QuarterUpModal';

export default function BetweenSetsView() {
  const nextMatch = useArenaStore((s) => s.nextMatch);
  const recentResults = useArenaStore((s) => s.recentResults);
  const [showQuarterUp, setShowQuarterUp] = useState(false);

  return (
    <div className="phase-overlay between-sets-overlay">
      <div className="wire-logo">WORLD OF NO JOHNS</div>
      <div className="wire-subtitle">AUTONOMOUS ARENA</div>

      {nextMatch && (
        <div className="next-match-card">
          <div className="next-match-header">NEXT MATCH</div>
          <div className="vs-preview">
            <div className="vs-preview-player p1-side">
              <div className="vs-preview-name" style={{ color: 'var(--p1)' }}>
                @{nextMatch.p1.username}
              </div>
              <div className="vs-preview-meta">{nextMatch.p1.character}</div>
              <div className="vs-preview-elo">{nextMatch.p1.elo} ELO</div>
              <div className="vs-preview-record">{nextMatch.p1.record}</div>
              {nextMatch.p1.sponsor && (
                <div className="vs-preview-sponsor">sponsored by @{nextMatch.p1.sponsor}</div>
              )}
            </div>
            <div className="vs-preview-divider">VS</div>
            <div className="vs-preview-player p2-side">
              <div className="vs-preview-name" style={{ color: 'var(--p2)' }}>
                @{nextMatch.p2.username}
              </div>
              <div className="vs-preview-meta">{nextMatch.p2.character}</div>
              <div className="vs-preview-elo">{nextMatch.p2.elo} ELO</div>
              <div className="vs-preview-record">{nextMatch.p2.record}</div>
              {nextMatch.p2.sponsor && (
                <div className="vs-preview-sponsor">sponsored by @{nextMatch.p2.sponsor}</div>
              )}
            </div>
          </div>
        </div>
      )}

      {recentResults.length > 0 && (
        <div className="recent-results">
          {recentResults.map((r, i) => (
            <div key={i} className="recent-result-line">
              <span style={{ color: 'var(--p1)' }}>{r.winner}</span>
              {' def. '}
              <span style={{ color: 'var(--dim)' }}>{r.loser}</span>
              {` (${r.stocks} stock${r.stocks !== 1 ? 's' : ''})`}
            </div>
          ))}
        </div>
      )}

      <button
        className="btn btn-primary quarter-up-cta"
        onClick={() => setShowQuarterUp(true)}
      >
        QUARTER UP
      </button>

      {showQuarterUp && (
        <QuarterUpModal onClose={() => setShowQuarterUp(false)} />
      )}
    </div>
  );
}
