'use client';

import { useArenaStore } from '@/stores/arena';

export default function PostMatchView() {
  const result = useArenaStore((s) => s.matchResult);
  const match = useArenaStore((s) => s.currentMatch);
  if (!result) return null;

  const isP1Winner = result.winner === match?.p1.username;
  const winnerElo = isP1Winner ? match?.p1.elo : match?.p2.elo;
  const loserElo = isP1Winner ? match?.p2.elo : match?.p1.elo;

  return (
    <div className="phase-overlay post-match-overlay">
      <div className="game-text">GAME!</div>

      <div
        className="winner-name"
        style={{ color: isP1Winner ? 'var(--p1)' : 'var(--p2)' }}
      >
        @{result.winner}
      </div>
      <div className="win-detail">
        {result.stocks} stock{result.stocks !== 1 ? 's' : ''} remaining
      </div>

      <div className="elo-deltas">
        <span className="elo-winner">
          {(winnerElo ?? 0) + result.eloDelta} (+{result.eloDelta})
        </span>
        <span className="elo-sep">/</span>
        <span className="elo-loser">
          {(loserElo ?? 0) - result.eloDelta} (-{result.eloDelta})
        </span>
      </div>

      <div className="prize-split">
        POT: 0.02 SOL &middot; 50% winner / 35% sponsor / 10% protocol / 5% loser
      </div>
    </div>
  );
}
