'use client';

import { useArenaStore, type MatchEvent } from '@/stores/arena';

function formatEvent(e: MatchEvent) {
  const prefix = `[F ${e.frame}]`;
  if (e.type === 'ko') {
    return `${prefix} ${e.attacker} KO'd ${e.defender}`;
  }
  const pct = e.newPercent != null ? ` (${e.newPercent.toFixed(1)}%)` : '';
  const combo = e.type === 'combo' ? ' combo' : '';
  return `${prefix} ${e.attacker} \u25B8 ${e.attackName}${combo} \u2192 ${e.defender}${pct}`;
}

export default function MatchTicker() {
  const events = useArenaStore((s) => s.matchEvents);
  const visible = events.slice(-5);

  if (visible.length === 0) return null;

  return (
    <div className="match-ticker">
      {visible.map((e, i) => (
        <div key={`${e.frame}-${i}`} className="match-ticker-line">
          {formatEvent(e)}
        </div>
      ))}
    </div>
  );
}
