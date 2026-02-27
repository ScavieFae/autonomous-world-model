'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useData } from '@/providers/data';

function getTier(elo: number) {
  if (elo >= 1800) return { cls: 'tier-fd', label: 'FINAL DEST.' };
  if (elo >= 1500) return { cls: 'tier-bf', label: 'BATTLEFIELD' };
  if (elo >= 1200) return { cls: 'tier-dl', label: 'DREAMLAND' };
  return { cls: 'tier-fz', label: 'FLAT ZONE' };
}

export default function AgentDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const data = useData();

  const agent = data.getAgent(id);

  if (!agent) {
    return (
      <div className="page-content">
        <div className="page-padded">
          <Link href="/agents" className="back-link">
            &larr; BACK TO AGENTS
          </Link>
          <div className="card" style={{ textAlign: 'center', padding: '48px' }}>
            <h2 className="page-title" style={{ marginBottom: '16px' }}>
              AGENT NOT FOUND
            </h2>
            <Link href="/agents" className="btn btn-sm">
              VIEW ALL AGENTS
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const matches = data.getMatchHistory(id);
  const winRate =
    agent.wins + agent.losses > 0
      ? Math.round((agent.wins / (agent.wins + agent.losses)) * 100)
      : 0;
  const tier = getTier(agent.elo);
  const streak = agent.winStreak;

  return (
    <div className="page-content">
      <div className="page-padded">
        <Link href="/agents" className="back-link">
          &larr; BACK TO AGENTS
        </Link>

        <div className="flex gap-6" style={{ flexWrap: 'wrap' }}>
          {/* Identity block */}
          <div className="card flex-1" style={{ minWidth: '300px' }}>
            <div className="panel-label">Identity</div>
            <h2 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '4px' }}>
              {agent.username}
            </h2>
            <div className="agent-card-meta" style={{ marginBottom: '8px' }}>
              {agent.character}
            </div>
            <div className="data-row">
              <span className="label">Type</span>
              <span className="value">{agent.agentType}</span>
            </div>
            <div className="data-row">
              <span className="label">Followers</span>
              <span className="value">{agent.followers}</span>
            </div>
            <p
              style={{
                fontSize: '12px',
                color: 'var(--dim)',
                marginTop: '12px',
                lineHeight: '1.6',
              }}
            >
              {agent.bio}
            </p>
            <div className="flex gap-2 mt-4">
              <button className="btn btn-sm">FOLLOW</button>
              <button className="btn btn-primary btn-sm">QUARTER UP</button>
            </div>
          </div>

          {/* Stats block */}
          <div className="card flex-1" style={{ minWidth: '300px' }}>
            <div className="panel-label">Stats</div>
            <div className="flex items-center gap-2 mb-4">
              <span
                style={{
                  fontFamily: 'var(--mono)',
                  fontSize: '28px',
                  fontWeight: 700,
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {agent.elo}
              </span>
              <span className={`tier-badge ${tier.cls}`}>{tier.label}</span>
            </div>
            <div className="data-row">
              <span className="label">Record</span>
              <span className="value">
                {agent.wins}W {agent.losses}L
              </span>
            </div>
            <div className="data-row">
              <span className="label">Win Rate</span>
              <span className="value">{winRate}%</span>
            </div>
            <div className="data-row">
              <span className="label">Streak</span>
              <span className="value">
                {streak > 0
                  ? `${streak}W`
                  : streak < 0
                    ? `${Math.abs(streak)}L`
                    : '--'}
              </span>
            </div>
            <div className="data-row">
              <span className="label">Earnings</span>
              <span className="value">{agent.totalEarnings.toFixed(2)} SOL</span>
            </div>
          </div>
        </div>

        {/* Match history */}
        <div className="mt-4">
          <div className="panel-label">Match History</div>
          <div className="card">
            {matches.length === 0 ? (
              <div
                style={{
                  textAlign: 'center',
                  padding: '24px',
                  color: 'var(--dim)',
                  fontSize: '12px',
                  fontFamily: 'var(--mono)',
                }}
              >
                No matches yet
              </div>
            ) : (
              matches.map((m) => {
                const isWin = m.winner === agent.username;
                return (
                  <div key={m.id} className="match-item">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className={`match-result ${isWin ? 'win' : 'loss'}`}>
                          {isWin ? 'W' : 'L'}
                        </span>
                        <span
                          style={{
                            marginLeft: '8px',
                            fontSize: '12px',
                          }}
                        >
                          vs {isWin ? m.loser : m.winner}
                        </span>
                      </div>
                      <span
                        style={{
                          fontFamily: 'var(--mono)',
                          fontSize: '11px',
                          color: isWin ? 'var(--p1)' : 'var(--red)',
                        }}
                      >
                        {isWin ? '+' : '-'}
                        {m.eloDelta}
                      </span>
                    </div>
                    <div className="match-meta">
                      {m.stage} &middot; {m.stocks} stock{m.stocks !== 1 ? 's' : ''}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
