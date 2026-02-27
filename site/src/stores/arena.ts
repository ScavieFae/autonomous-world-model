import { create } from 'zustand';
import type { RenderMode, CharacterFillMode } from '@/engine/types';
import { CHARACTER_FILL_MODES } from '@/engine/types';
import type { LiveConnectionState } from '@/engine/live';
import type { CrankMatchStart, CrankMatchEnd } from '@/engine/live';
import { CHARACTERS } from '@/engine/constants';

export type ArenaPhase = 'between-sets' | 'pre-match' | 'live' | 'post-match';

export interface MatchPreview {
  p1: { username: string; character: string; elo: number; record: string; sponsor?: string };
  p2: { username: string; character: string; elo: number; record: string; sponsor?: string };
}

export interface MatchResult {
  winner: string;
  loser: string;
  stocks: number;
  eloDelta: number;
}

export interface MatchEvent {
  frame: number;
  type: 'hit' | 'ko' | 'combo';
  attacker: string;
  defender: string;
  attackName: string;
  damage?: number;
  newPercent?: number;
}

interface ArenaStore {
  phase: ArenaPhase;
  renderMode: RenderMode;
  characterFill: CharacterFillMode;
  liveConnection: LiveConnectionState;
  nextMatch: MatchPreview | null;
  currentMatch: MatchPreview | null;
  matchResult: MatchResult | null;
  matchEvents: MatchEvent[];
  recentResults: MatchResult[];

  setPhase: (phase: ArenaPhase) => void;
  setRenderMode: (mode: RenderMode) => void;
  cycleCharacterFill: () => void;
  setLiveConnection: (state: LiveConnectionState) => void;
  pushEvent: (event: MatchEvent) => void;
  handleMatchStart: (msg: CrankMatchStart) => void;
  handleMatchEnd: (msg: CrankMatchEnd) => void;
  runDemoCycle: () => void;
  stopDemoCycle: () => void;
}

// Mock agents for demo rotation
const AGENTS = [
  { username: 'foxmaster-9000', character: 'Fox', elo: 1847, record: '24-8' },
  { username: 'waveland-wizard', character: 'Marth', elo: 1791, record: '19-12' },
  { username: 'shine-bot-3k', character: 'Falco', elo: 1823, record: '22-10' },
  { username: 'rest-in-puff', character: 'Jigglypuff', elo: 1756, record: '16-11' },
  { username: 'tech-chase-ai', character: 'Sheik', elo: 1812, record: '21-9' },
  { username: 'stomp-lord', character: 'Cpt. Falcon', elo: 1779, record: '18-13' },
  { username: 'pill-pusher', character: 'Doc', elo: 1698, record: '12-15' },
  { username: 'chain-grab-9', character: 'Marth', elo: 1734, record: '15-14' },
  { username: 'nana-carry', character: 'Popo', elo: 1665, record: '10-16' },
  { username: 'laser-brain', character: 'Falco', elo: 1801, record: '20-11' },
];

const MOCK_RESULTS: MatchResult[] = [
  { winner: 'foxmaster-9000', loser: 'rest-in-puff', stocks: 2, eloDelta: 14 },
  { winner: 'tech-chase-ai', loser: 'pill-pusher', stocks: 3, eloDelta: 18 },
  { winner: 'shine-bot-3k', loser: 'stomp-lord', stocks: 1, eloDelta: 11 },
  { winner: 'waveland-wizard', loser: 'chain-grab-9', stocks: 2, eloDelta: 13 },
  { winner: 'laser-brain', loser: 'nana-carry', stocks: 3, eloDelta: 20 },
];

const SPONSORS = ['alice', 'bob', 'carol', undefined, 'dave', undefined, 'eve', undefined, undefined, 'frank'];

let matchRotation = 0;
let demoTimerId: ReturnType<typeof setTimeout> | null = null;
let lifecycleTimerId: ReturnType<typeof setTimeout> | null = null;

function pickNextMatch(): MatchPreview {
  const i = (matchRotation * 2) % AGENTS.length;
  const j = (matchRotation * 2 + 1) % AGENTS.length;
  matchRotation++;
  return {
    p1: { ...AGENTS[i], sponsor: SPONSORS[i] },
    p2: { ...AGENTS[j], sponsor: SPONSORS[j] },
  };
}

export const useArenaStore = create<ArenaStore>((set, get) => ({
  phase: 'between-sets',
  renderMode: 'xray',
  characterFill: 'hologram',
  liveConnection: 'idle',
  nextMatch: pickNextMatch(),
  currentMatch: null,
  matchResult: null,
  matchEvents: [],
  recentResults: MOCK_RESULTS.slice(0, 3),

  setPhase: (phase) => set({ phase }),
  setRenderMode: (renderMode) => set({ renderMode }),
  setLiveConnection: (liveConnection) => set({ liveConnection }),
  cycleCharacterFill: () => set((s) => {
    const idx = CHARACTER_FILL_MODES.indexOf(s.characterFill);
    const next = CHARACTER_FILL_MODES[(idx + 1) % CHARACTER_FILL_MODES.length];
    return { characterFill: next };
  }),

  pushEvent: (event) => set((s) => ({
    matchEvents: [...s.matchEvents.slice(-49), event],
  })),

  handleMatchStart: (msg) => {
    // Clear any pending lifecycle timer
    if (lifecycleTimerId) { clearTimeout(lifecycleTimerId); lifecycleTimerId = null; }

    const charName = (idx: number, fallbackName: string) =>
      CHARACTERS[idx] ?? fallbackName;

    const preview: MatchPreview = {
      p1: {
        username: charName(msg.p0.character, msg.p0.character_name),
        character: charName(msg.p0.character, msg.p0.character_name),
        elo: 1500,
        record: '0-0',
      },
      p2: {
        username: charName(msg.p1.character, msg.p1.character_name),
        character: charName(msg.p1.character, msg.p1.character_name),
        elo: 1500,
        record: '0-0',
      },
    };

    set({ phase: 'pre-match', currentMatch: preview, matchEvents: [] });

    // Transition to live after 3s
    lifecycleTimerId = setTimeout(() => {
      if (get().phase === 'pre-match') {
        set({ phase: 'live' });
      }
      lifecycleTimerId = null;
    }, 3000);
  },

  handleMatchEnd: (msg) => {
    // Clear any pending lifecycle timer
    if (lifecycleTimerId) { clearTimeout(lifecycleTimerId); lifecycleTimerId = null; }

    const { currentMatch, recentResults } = get();
    const winnerName = msg.winner != null
      ? (msg.winner === 0 ? currentMatch?.p1.username : currentMatch?.p2.username) ?? 'unknown'
      : 'draw';
    const loserName = msg.winner != null
      ? (msg.winner === 0 ? currentMatch?.p2.username : currentMatch?.p1.username) ?? 'unknown'
      : 'draw';
    const winnerStocks = msg.winner != null ? msg.final_stocks[msg.winner] : 0;

    const result: MatchResult = {
      winner: winnerName,
      loser: loserName,
      stocks: winnerStocks,
      eloDelta: 0,
    };

    set({
      phase: 'post-match',
      matchResult: result,
      recentResults: [result, ...recentResults].slice(0, 5),
    });

    // Transition to between-sets after 5s
    lifecycleTimerId = setTimeout(() => {
      if (get().phase === 'post-match') {
        set({
          phase: 'between-sets',
          matchResult: null,
          matchEvents: [],
        });
      }
      lifecycleTimerId = null;
    }, 5000);
  },

  runDemoCycle: () => {
    const advance = () => {
      const { phase, nextMatch } = get();

      if (phase === 'between-sets') {
        set({ phase: 'pre-match', currentMatch: nextMatch });
        demoTimerId = setTimeout(advance, 3000);
      } else if (phase === 'pre-match') {
        set({ phase: 'live', matchEvents: [] });
        // Live phase runs until mock data ends — PlaybackEngine will call onMatchEnd
        // For demo, auto-advance after the mock data duration (~6s at 360 frames / 60fps)
        demoTimerId = setTimeout(advance, 6000);
      } else if (phase === 'live') {
        const { currentMatch, recentResults } = get();
        const result: MatchResult = {
          winner: currentMatch?.p1.username ?? 'unknown',
          loser: currentMatch?.p2.username ?? 'unknown',
          stocks: 2,
          eloDelta: 8 + Math.floor(Math.random() * 16),
        };
        set({
          phase: 'post-match',
          matchResult: result,
          recentResults: [result, ...recentResults].slice(0, 5),
        });
        demoTimerId = setTimeout(advance, 5000);
      } else if (phase === 'post-match') {
        set({
          phase: 'between-sets',
          nextMatch: pickNextMatch(),
          matchResult: null,
          matchEvents: [],
        });
        demoTimerId = setTimeout(advance, 8000);
      }
    };

    // Kick off
    demoTimerId = setTimeout(advance, 8000);
  },

  stopDemoCycle: () => {
    if (demoTimerId) {
      clearTimeout(demoTimerId);
      demoTimerId = null;
    }
  },
}));
