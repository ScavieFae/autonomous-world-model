'use client';

import { useArenaStore } from '@/stores/arena';

export default function PreMatchView() {
  const match = useArenaStore((s) => s.currentMatch);
  if (!match) return null;

  return (
    <div className="phase-overlay pre-match-overlay">
      <div className="vs-splash">
        <div className="vs-splash-player slide-in-left">
          <div className="vs-splash-name" style={{ color: 'var(--p1)' }}>
            @{match.p1.username}
          </div>
          <div className="vs-splash-detail">
            {match.p1.character} &middot; {match.p1.elo} ELO
          </div>
          {match.p1.sponsor && (
            <div className="vs-splash-sponsor">@{match.p1.sponsor}</div>
          )}
        </div>

        <div className="vs-flash">VS</div>

        <div className="vs-splash-player slide-in-right">
          <div className="vs-splash-name" style={{ color: 'var(--p2)' }}>
            @{match.p2.username}
          </div>
          <div className="vs-splash-detail">
            {match.p2.character} &middot; {match.p2.elo} ELO
          </div>
          {match.p2.sponsor && (
            <div className="vs-splash-sponsor">@{match.p2.sponsor}</div>
          )}
        </div>
      </div>
    </div>
  );
}
