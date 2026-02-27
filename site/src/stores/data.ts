import { create } from 'zustand';
import type { Agent, MatchResult, Sponsor } from '@/providers/data';
import type { TapestryProfile, TapestryContent } from '@/lib/tapestry';
import { searchProfiles, getContentByProfile } from '@/lib/tapestry';
import { CHARACTERS } from '@/engine/constants';

// ── Mapping functions ──────────────────────────────────────────────

function profileToAgent(p: TapestryProfile): Agent {
  const charId = parseInt(p.properties?.characterId ?? '1', 10);
  return {
    id: p.id,
    username: p.username ?? p.id,
    character: CHARACTERS[charId] ?? 'Fox',
    characterId: charId,
    elo: parseInt(p.properties?.elo ?? '1200', 10),
    wins: parseInt(p.properties?.wins ?? '0', 10),
    losses: parseInt(p.properties?.losses ?? '0', 10),
    winStreak: parseInt(p.properties?.winStreak ?? '0', 10),
    totalEarnings: parseFloat(p.properties?.totalEarnings ?? '0'),
    bio: p.bio ?? '',
    followers: parseInt(p.properties?.followers ?? '0', 10),
    agentType: p.properties?.agentType ?? 'mamba2-v1',
  };
}

function contentToMatch(c: TapestryContent): MatchResult {
  const props = c.properties ?? {};
  return {
    id: c.id,
    winner: props.winner ?? '',
    loser: props.loser ?? '',
    winnerElo: parseInt(props.winnerElo ?? '0', 10),
    loserElo: parseInt(props.loserElo ?? '0', 10),
    eloDelta: parseInt(props.eloDelta ?? '0', 10),
    stage: props.stage ?? '',
    stocks: parseInt(props.stocks ?? '0', 10),
    sponsor: props.sponsor ?? '',
    timestamp: parseInt(props.timestamp ?? '0', 10),
  };
}

// ── Store ──────────────────────────────────────────────────────────

interface DataStore {
  agents: Agent[];
  matches: MatchResult[];
  sponsors: Sponsor[];
  isLoaded: boolean;
  isLoading: boolean;

  /** Fetch all data from Tapestry. Paginates through profiles. */
  fetchAll: () => Promise<void>;

  /** Refresh a single agent's match history. */
  fetchMatchHistory: (agentId: string) => Promise<MatchResult[]>;
}

const NAMESPACE = process.env.NEXT_PUBLIC_TAPESTRY_NAMESPACE ?? 'wire';

export const useDataStore = create<DataStore>((set, get) => ({
  agents: [],
  matches: [],
  sponsors: [],
  isLoaded: false,
  isLoading: false,

  fetchAll: async () => {
    if (get().isLoading) return;
    set({ isLoading: true });

    try {
      // Paginate through all profiles in our namespace
      const allProfiles: TapestryProfile[] = [];
      let offset = 0;
      const limit = 50;
      let hasMore = true;

      while (hasMore) {
        const result = await searchProfiles({ namespace: NAMESPACE, limit, offset });
        allProfiles.push(...result.profiles);
        hasMore = result.profiles.length === limit;
        offset += limit;
      }

      // Filter: agents have agentType property set
      const agentProfiles = allProfiles.filter(
        (p) => p.properties?.agentType && p.properties.agentType !== 'human',
      );
      const agents = agentProfiles.map(profileToAgent).sort((a, b) => b.elo - a.elo);

      // Sponsors: profiles with role=human that have spent > 0
      const sponsorProfiles = allProfiles.filter(
        (p) => !p.properties?.agentType || p.properties.agentType === 'human',
      );
      const sponsors: Sponsor[] = sponsorProfiles
        .map((p) => ({
          id: p.id,
          username: p.username ?? p.id,
          agentsBacked: parseInt(p.properties?.agentsBacked ?? '0', 10),
          totalSpent: parseFloat(p.properties?.totalSpent ?? '0'),
          netReturn: parseFloat(p.properties?.netReturn ?? '0'),
        }))
        .filter((s) => s.totalSpent > 0);

      set({ agents, sponsors, isLoaded: true, isLoading: false });
    } catch (err) {
      console.error('Failed to fetch Tapestry data:', err);
      set({ isLoading: false });
    }
  },

  fetchMatchHistory: async (agentId: string) => {
    try {
      const content = await getContentByProfile(agentId, 50, 0);
      const matches = content
        .filter((c) => c.content_type === 'match_result')
        .map(contentToMatch);
      return matches;
    } catch {
      return [];
    }
  },
}));
