'use client';

import { createContext, useContext, type ReactNode } from 'react';

export interface Agent {
  id: string;
  username: string;
  character: string;
  characterId: number;
  elo: number;
  wins: number;
  losses: number;
  winStreak: number;
  totalEarnings: number;
  bio: string;
  followers: number;
  agentType: string;
}

export interface MatchResult {
  id: string;
  winner: string;
  loser: string;
  winnerElo: number;
  loserElo: number;
  eloDelta: number;
  stage: string;
  stocks: number;
  sponsor: string;
  timestamp: number;
}

export interface Sponsor {
  id: string;
  username: string;
  agentsBacked: number;
  totalSpent: number;
  netReturn: number;
}

export interface DataProvider {
  getAgents: () => Agent[];
  getAgent: (id: string) => Agent | undefined;
  getMatchHistory: (agentId: string) => MatchResult[];
  getLeaderboard: () => Agent[];
  getSponsors: () => Sponsor[];
}

const DataContext = createContext<DataProvider | null>(null);

export function useData(): DataProvider {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error('useData must be used within DataContextProvider');
  return ctx;
}

// Mock data
const MOCK_AGENTS: Agent[] = [
  { id: 'fox-1', username: 'foxmaster-9000', character: 'Fox', characterId: 1, elo: 1847, wins: 142, losses: 41, winStreak: 12, totalEarnings: 2.4, bio: 'Trained on 10M frames of Mango vs Zain. Multishine or die.', followers: 142, agentType: 'mamba2-v1' },
  { id: 'marth-1', username: 'waveland-wizard', character: 'Marth', characterId: 18, elo: 1791, wins: 128, losses: 52, winStreak: 3, totalEarnings: 1.8, bio: 'Ken combo specialist. Tipper or nothing.', followers: 89, agentType: 'mamba2-v1' },
  { id: 'puff-1', username: 'puff-master', character: 'Jigglypuff', characterId: 32, elo: 1756, wins: 97, losses: 44, winStreak: -1, totalEarnings: 1.2, bio: 'Rest is best. Hungrybox fan network.', followers: 203, agentType: 'mamba2-v1' },
  { id: 'falcon-1', username: 'falcon-punch', character: 'Cpt. Falcon', characterId: 2, elo: 1623, wins: 84, losses: 71, winStreak: 2, totalEarnings: 0.8, bio: 'Show me your moves. Knee of justice enjoyer.', followers: 67, agentType: 'mamba2-v1' },
  { id: 'falco-1', username: 'shine-grinder', character: 'Falco', characterId: 22, elo: 1544, wins: 73, losses: 58, winStreak: -3, totalEarnings: 0.6, bio: 'Pillar combos all day. Westballz 2.0.', followers: 54, agentType: 'mamba2-v1' },
  { id: 'sheik-1', username: 'needle-storm', character: 'Sheik', characterId: 7, elo: 1489, wins: 61, losses: 49, winStreak: 1, totalEarnings: 0.45, bio: 'Tech chase demon. Plup club.', followers: 41, agentType: 'mamba2-v1' },
  { id: 'peach-1', username: 'float-queen', character: 'Peach', characterId: 9, elo: 1387, wins: 44, losses: 56, winStreak: -2, totalEarnings: 0.3, bio: 'Turnip RNG blessed. Armada legacy.', followers: 38, agentType: 'mamba2-v1' },
  { id: 'ic-1', username: 'desync-lord', character: 'Popo', characterId: 10, elo: 1245, wins: 32, losses: 48, winStreak: -4, totalEarnings: 0.15, bio: 'Wobbling is hype, actually. Fight me.', followers: 22, agentType: 'mamba2-v1' },
  { id: 'link-1', username: 'bomb-recovery', character: 'Link', characterId: 6, elo: 1156, wins: 18, losses: 42, winStreak: -6, totalEarnings: 0.05, bio: 'Low tier hero. Bomb jump specialist.', followers: 14, agentType: 'mamba2-v1' },
  { id: 'ganon-1', username: 'stomp-city', character: 'Ganondorf', characterId: 25, elo: 1098, wins: 12, losses: 38, winStreak: -8, totalEarnings: 0.02, bio: 'Disrespect only. Style points > stocks.', followers: 31, agentType: 'mamba2-v1' },
];

const MOCK_MATCHES: MatchResult[] = [
  { id: 'm1', winner: 'foxmaster-9000', loser: 'waveland-wizard', winnerElo: 1847, loserElo: 1791, eloDelta: 12, stage: 'Battlefield', stocks: 2, sponsor: 'alice', timestamp: Date.now() - 300_000 },
  { id: 'm2', winner: 'waveland-wizard', loser: 'foxmaster-9000', winnerElo: 1803, loserElo: 1835, eloDelta: 12, stage: 'Final Destination', stocks: 1, sponsor: 'bob', timestamp: Date.now() - 600_000 },
  { id: 'm3', winner: 'puff-master', loser: 'falcon-punch', winnerElo: 1756, loserElo: 1623, eloDelta: 8, stage: 'Dream Land', stocks: 3, sponsor: 'alice', timestamp: Date.now() - 900_000 },
  { id: 'm4', winner: 'foxmaster-9000', loser: 'shine-grinder', winnerElo: 1859, loserElo: 1532, eloDelta: 6, stage: 'Yoshis Story', stocks: 2, sponsor: 'carol', timestamp: Date.now() - 1_200_000 },
  { id: 'm5', winner: 'falcon-punch', loser: 'needle-storm', winnerElo: 1635, loserElo: 1477, eloDelta: 10, stage: 'Battlefield', stocks: 1, sponsor: 'bob', timestamp: Date.now() - 1_500_000 },
];

const MOCK_SPONSORS: Sponsor[] = [
  { id: 's1', username: 'alice', agentsBacked: 14, totalSpent: 0.82, netReturn: 0.41 },
  { id: 's2', username: 'bob', agentsBacked: 8, totalSpent: 0.34, netReturn: 0.12 },
  { id: 's3', username: 'carol', agentsBacked: 22, totalSpent: 1.10, netReturn: -0.03 },
  { id: 's4', username: 'dave', agentsBacked: 5, totalSpent: 0.21, netReturn: 0.08 },
  { id: 's5', username: 'eve', agentsBacked: 3, totalSpent: 0.09, netReturn: -0.02 },
];

const mockProvider: DataProvider = {
  getAgents: () => MOCK_AGENTS,
  getAgent: (id: string) => MOCK_AGENTS.find((a) => a.id === id),
  getMatchHistory: (agentId: string) => {
    const agent = MOCK_AGENTS.find((a) => a.id === agentId);
    if (!agent) return [];
    return MOCK_MATCHES.filter(
      (m) => m.winner === agent.username || m.loser === agent.username
    );
  },
  getLeaderboard: () => [...MOCK_AGENTS].sort((a, b) => b.elo - a.elo),
  getSponsors: () => MOCK_SPONSORS,
};

export function DataContextProvider({ children }: { children: ReactNode }) {
  return (
    <DataContext.Provider value={mockProvider}>
      {children}
    </DataContext.Provider>
  );
}
