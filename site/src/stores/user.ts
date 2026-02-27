import { create } from 'zustand';
import type { TapestryProfile } from '@/lib/tapestry';
import { findOrCreateProfile } from '@/lib/tapestry';

interface UserState {
  wallet: string | null;
  profile: TapestryProfile | null;
  isLoading: boolean;
  error: string | null;

  /** Called when wallet connects — auto-creates Tapestry profile if API key set */
  onConnect: (walletAddress: string) => Promise<void>;

  /** Called when wallet disconnects */
  onDisconnect: () => void;
}

const hasTapestry = !!process.env.NEXT_PUBLIC_TAPESTRY_API_KEY;

export const useUserStore = create<UserState>((set) => ({
  wallet: null,
  profile: null,
  isLoading: false,
  error: null,

  onConnect: async (walletAddress: string) => {
    set({ wallet: walletAddress, isLoading: true, error: null });

    if (!hasTapestry) {
      // No API key — just store the wallet address
      set({ isLoading: false });
      return;
    }

    try {
      // Truncated wallet as default username
      const username = `user-${walletAddress.slice(0, 8)}`;
      const profile = await findOrCreateProfile({
        walletAddress,
        username,
        bio: '',
        customProperties: { role: 'human' },
      });
      set({ profile, isLoading: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to load profile',
        isLoading: false,
      });
    }
  },

  onDisconnect: () => {
    set({ wallet: null, profile: null, isLoading: false, error: null });
  },
}));
