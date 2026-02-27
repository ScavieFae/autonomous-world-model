import { useState, useEffect, useCallback } from 'react';
import { useUserStore } from '@/stores/user';
import { checkFollow, follow, unfollow } from '@/lib/tapestry';

const hasTapestry = !!process.env.NEXT_PUBLIC_TAPESTRY_API_KEY;

export function useFollow(agentId: string) {
  const [isFollowing, setIsFollowing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const profile = useUserStore((s) => s.profile);
  const wallet = useUserStore((s) => s.wallet);

  // Check initial follow state
  useEffect(() => {
    if (!hasTapestry || !profile) return;

    let cancelled = false;
    checkFollow(profile.id, agentId).then((result) => {
      if (!cancelled) setIsFollowing(result);
    });
    return () => { cancelled = true; };
  }, [profile, agentId]);

  const toggle = useCallback(async () => {
    if (!wallet) return;
    if (!hasTapestry || !profile) {
      // Mock mode — just toggle locally
      setIsFollowing((prev) => !prev);
      return;
    }

    setIsLoading(true);
    try {
      if (isFollowing) {
        await unfollow(profile.id, agentId);
        setIsFollowing(false);
      } else {
        await follow(profile.id, agentId);
        setIsFollowing(true);
      }
    } catch (err) {
      console.error('Follow toggle failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, [wallet, profile, agentId, isFollowing]);

  return { isFollowing, toggle, isLoading, requiresWallet: !wallet };
}
