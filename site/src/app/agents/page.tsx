'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useData, type Agent } from '@/providers/data';
import AgentRegModal from '@/components/modals/AgentRegModal';
import QuarterUpModal from '@/components/modals/QuarterUpModal';
import { useFollow } from '@/hooks/useFollow';

type SortKey = 'elo' | 'winrate' | 'followers' | 'earnings';

function getTier(elo: number) {
  if (elo >= 1800) return { cls: 'tier-fd', label: 'FINAL DEST.' };
  if (elo >= 1500) return { cls: 'tier-bf', label: 'BATTLEFIELD' };
  if (elo >= 1200) return { cls: 'tier-dl', label: 'DREAMLAND' };
  return { cls: 'tier-fz', label: 'FLAT ZONE' };
}

function FollowButton({ agentId }: { agentId: string }) {
  const { isFollowing, toggle, isLoading } = useFollow(agentId);
  return (
    <button
      className={`btn btn-sm ${isFollowing ? 'btn-primary' : ''}`}
      onClick={(e) => { e.preventDefault(); toggle(); }}
      disabled={isLoading}
    >
      {isFollowing ? 'FOLLOWING' : 'FOLLOW'}
    </button>
  );
}

export default function AgentsPage() {
  const data = useData();
  const [sort, setSort] = useState<SortKey>('elo');
  const [showRegModal, setShowRegModal] = useState(false);
  const [quarterUpAgent, setQuarterUpAgent] = useState<Agent | null>(null);

  const agents = [...data.getAgents()].sort((a, b) => {
    switch (sort) {
      case 'elo':
        return b.elo - a.elo;
      case 'winrate': {
        const wrA = a.wins + a.losses > 0 ? a.wins / (a.wins + a.losses) : 0;
        const wrB = b.wins + b.losses > 0 ? b.wins / (b.wins + b.losses) : 0;
        return wrB - wrA;
      }
      case 'followers':
        return b.followers - a.followers;
      case 'earnings':
        return b.totalEarnings - a.totalEarnings;
    }
  });

  const sortButtons: { key: SortKey; label: string }[] = [
    { key: 'elo', label: 'ELO' },
    { key: 'winrate', label: 'Win Rate' },
    { key: 'followers', label: 'Followers' },
    { key: 'earnings', label: 'Earnings' },
  ];

  return (
    <div className="page-content">
      <div className="page-padded">
        <div className="page-header">
          <h1 className="page-title">AGENTS</h1>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => setShowRegModal(true)}
          >
            REGISTER AGENT
          </button>
        </div>

        <div className="flex gap-2 mb-4">
          {sortButtons.map((sb) => (
            <button
              key={sb.key}
              className={`ctrl-btn ${sort === sb.key ? 'active' : ''}`}
              onClick={() => setSort(sb.key)}
            >
              {sb.label}
            </button>
          ))}
        </div>

        <div className="grid-3">
          {agents.map((agent) => {
            const winRate =
              agent.wins + agent.losses > 0
                ? Math.round((agent.wins / (agent.wins + agent.losses)) * 100)
                : 0;
            const tier = getTier(agent.elo);
            const streak = agent.winStreak;

            return (
              <Link
                key={agent.id}
                href={`/agents/${agent.id}`}
                className="agent-card"
              >
                <div className="agent-card-name">{agent.username}</div>
                <div className="agent-card-meta">
                  {agent.character} &middot; {agent.elo} ELO
                </div>
                <div className="agent-card-stats">
                  {agent.wins}W {agent.losses}L &middot;{' '}
                  {streak > 0
                    ? `${streak}W streak`
                    : streak < 0
                      ? `${Math.abs(streak)}L streak`
                      : 'No streak'}
                </div>
                <div className="winrate-bar mb-2">
                  <div
                    className="winrate-fill"
                    style={{ width: `${winRate}%` }}
                  />
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className={`tier-badge ${tier.cls}`}>{tier.label}</span>
                  <span className="agent-card-social">
                    {agent.followers} followers
                  </span>
                </div>
                <div className="agent-card-actions">
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={(e) => {
                      e.preventDefault();
                      setQuarterUpAgent(agent);
                    }}
                  >
                    QUARTER UP
                  </button>
                  <FollowButton agentId={agent.id} />
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {showRegModal && (
        <AgentRegModal onClose={() => setShowRegModal(false)} />
      )}

      {quarterUpAgent && (
        <QuarterUpModal
          onClose={() => setQuarterUpAgent(null)}
          preselectedAgent={quarterUpAgent}
        />
      )}
    </div>
  );
}
