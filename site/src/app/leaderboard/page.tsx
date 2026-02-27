'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useData, type Agent } from '@/providers/data';

type Tab = 'agents' | 'sponsors';

function getTier(elo: number) {
  if (elo >= 1800) return { cls: 'tier-fd', label: 'FINAL DEST.' };
  if (elo >= 1500) return { cls: 'tier-bf', label: 'BATTLEFIELD' };
  if (elo >= 1200) return { cls: 'tier-dl', label: 'DREAMLAND' };
  return { cls: 'tier-fz', label: 'FLAT ZONE' };
}

function getTierForRange(elo: number): string {
  if (elo >= 1800) return 'fd';
  if (elo >= 1500) return 'bf';
  if (elo >= 1200) return 'dl';
  return 'fz';
}

const TIER_LABELS: Record<string, string> = {
  fd: 'FINAL DESTINATION — 1800+',
  bf: 'BATTLEFIELD — 1500+',
  dl: 'DREAMLAND — 1200+',
  fz: 'FLAT ZONE — Below 1200',
};

function groupByTier(agents: Agent[]) {
  const groups: { tier: string; agents: Agent[] }[] = [];
  let currentTier = '';

  for (const agent of agents) {
    const tier = getTierForRange(agent.elo);
    if (tier !== currentTier) {
      currentTier = tier;
      groups.push({ tier, agents: [] });
    }
    groups[groups.length - 1].agents.push(agent);
  }

  return groups;
}

export default function LeaderboardPage() {
  const data = useData();
  const [tab, setTab] = useState<Tab>('agents');

  const leaderboard = data.getLeaderboard();
  const sponsors = data.getSponsors();
  const tierGroups = groupByTier(leaderboard);

  return (
    <div className="page-content">
      <div className="page-padded">
        <div className="page-header">
          <h1 className="page-title">LEADERBOARD</h1>
        </div>

        <div className="flex gap-2 mb-4">
          <button
            className={`ctrl-btn ${tab === 'agents' ? 'active' : ''}`}
            onClick={() => setTab('agents')}
          >
            AGENTS
          </button>
          <button
            className={`ctrl-btn ${tab === 'sponsors' ? 'active' : ''}`}
            onClick={() => setTab('sponsors')}
          >
            SPONSORS
          </button>
        </div>

        {tab === 'agents' && (
          <table className="rank-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Agent</th>
                <th>ELO</th>
                <th>W/L</th>
                <th>Streak</th>
                <th>Earnings</th>
              </tr>
            </thead>
            <tbody>
              {tierGroups.map((group) => {
                const globalStart = leaderboard.indexOf(group.agents[0]);
                return [
                  <tr key={`tier-${group.tier}`}>
                    <td
                      colSpan={6}
                      className="tier-divider"
                    >
                      {TIER_LABELS[group.tier]}
                    </td>
                  </tr>,
                  ...group.agents.map((agent, i) => {
                    const rank = globalStart + i + 1;
                    const tier = getTier(agent.elo);
                    const streak = agent.winStreak;
                    return (
                      <tr key={agent.id}>
                        <td style={{ color: 'var(--dim)' }}>{rank}</td>
                        <td>
                          <Link
                            href={`/agents/${agent.id}`}
                            style={{
                              fontWeight: 600,
                              display: 'flex',
                              alignItems: 'center',
                              gap: '8px',
                            }}
                          >
                            {agent.username}
                            <span
                              style={{
                                fontSize: '10px',
                                color: 'var(--dim)',
                                fontWeight: 400,
                              }}
                            >
                              {agent.character}
                            </span>
                          </Link>
                        </td>
                        <td>
                          <span style={{ marginRight: '6px' }}>{agent.elo}</span>
                          <span className={`tier-badge ${tier.cls}`}>
                            {tier.label}
                          </span>
                        </td>
                        <td>
                          {agent.wins}W {agent.losses}L
                        </td>
                        <td
                          style={{
                            color:
                              streak > 0
                                ? 'var(--p1)'
                                : streak < 0
                                  ? 'var(--red)'
                                  : 'var(--dim)',
                          }}
                        >
                          {streak > 0
                            ? `${streak}W`
                            : streak < 0
                              ? `${Math.abs(streak)}L`
                              : '--'}
                        </td>
                        <td>{agent.totalEarnings.toFixed(2)} SOL</td>
                      </tr>
                    );
                  }),
                ];
              })}
            </tbody>
          </table>
        )}

        {tab === 'sponsors' && (
          <table className="rank-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Sponsor</th>
                <th>Agents Backed</th>
                <th>Total Spent</th>
                <th>Net Return</th>
              </tr>
            </thead>
            <tbody>
              {sponsors.map((s, i) => (
                <tr key={s.id}>
                  <td style={{ color: 'var(--dim)' }}>{i + 1}</td>
                  <td>
                    <Link
                      href={`/profile/${s.id}`}
                      style={{ fontWeight: 600 }}
                    >
                      {s.username}
                    </Link>
                  </td>
                  <td>{s.agentsBacked}</td>
                  <td>{s.totalSpent.toFixed(2)} SOL</td>
                  <td
                    style={{
                      color: s.netReturn >= 0 ? 'var(--p1)' : 'var(--red)',
                    }}
                  >
                    {s.netReturn >= 0 ? '+' : ''}
                    {s.netReturn.toFixed(2)} SOL
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
