/**
 * Tapestry REST API client.
 * Plain fetch — no SDK dependency.
 */

const BASE_URL = process.env.NEXT_PUBLIC_TAPESTRY_BASE_URL ?? 'https://api.usetapestry.dev/v1';
const API_KEY = process.env.NEXT_PUBLIC_TAPESTRY_API_KEY ?? '';
const NAMESPACE = process.env.NEXT_PUBLIC_TAPESTRY_NAMESPACE ?? 'wire';

// ── Types ──────────────────────────────────────────────────────────

export interface TapestryProfile {
  id: string;
  namespace: string;
  created_at: string;
  username?: string;
  bio?: string;
  image?: string;
  blockchain_address?: string;
  properties: Record<string, string>;
}

export interface TapestryContent {
  id: string;
  profile_id: string;
  content_type: string;
  content: string;
  created_at: string;
  properties: Record<string, string>;
}

export interface Pagination {
  page: number;
  limit: number;
  total: number;
}

interface SearchResult {
  profiles: TapestryProfile[];
  pagination: Pagination;
}

// ── Internal helpers ───────────────────────────────────────────────

function headers(): HeadersInit {
  return {
    'Content-Type': 'application/json',
    ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
  };
}

async function tapFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, { ...init, headers: { ...headers(), ...init?.headers } });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Tapestry ${res.status}: ${text || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Profile ────────────────────────────────────────────────────────

export async function findOrCreateProfile(opts: {
  walletAddress: string;
  username: string;
  bio?: string;
  customProperties?: Record<string, string>;
}): Promise<TapestryProfile> {
  // Try to find existing profile by wallet address first
  try {
    const existing = await getProfileByWallet(opts.walletAddress);
    if (existing) return existing;
  } catch {
    // Profile not found — create a new one
  }

  return tapFetch<TapestryProfile>('/profiles', {
    method: 'POST',
    body: JSON.stringify({
      namespace: NAMESPACE,
      username: opts.username,
      bio: opts.bio ?? '',
      blockchain_address: opts.walletAddress,
      properties: opts.customProperties ?? {},
    }),
  });
}

export async function getProfile(profileId: string): Promise<TapestryProfile> {
  return tapFetch<TapestryProfile>(`/profiles/${profileId}`);
}

export async function getProfileByWallet(walletAddress: string): Promise<TapestryProfile | null> {
  try {
    const result = await searchProfiles({ namespace: NAMESPACE, limit: 1 });
    // Search through results for matching wallet
    const match = result.profiles.find(
      (p) => p.blockchain_address === walletAddress,
    );
    return match ?? null;
  } catch {
    return null;
  }
}

export async function searchProfiles(opts: {
  namespace?: string;
  limit?: number;
  offset?: number;
}): Promise<SearchResult> {
  const params = new URLSearchParams();
  params.set('namespace', opts.namespace ?? NAMESPACE);
  if (opts.limit) params.set('limit', String(opts.limit));
  if (opts.offset) params.set('offset', String(opts.offset));

  return tapFetch<SearchResult>(`/profiles?${params}`);
}

// ── Follow / Unfollow ──────────────────────────────────────────────

export async function follow(followerId: string, followeeId: string): Promise<void> {
  await tapFetch<unknown>('/connections', {
    method: 'POST',
    body: JSON.stringify({
      start_id: followerId,
      end_id: followeeId,
      connection_type: 'follow',
    }),
  });
}

export async function unfollow(followerId: string, followeeId: string): Promise<void> {
  await tapFetch<unknown>('/connections', {
    method: 'DELETE',
    body: JSON.stringify({
      start_id: followerId,
      end_id: followeeId,
      connection_type: 'follow',
    }),
  });
}

export async function checkFollow(followerId: string, followeeId: string): Promise<boolean> {
  try {
    const result = await tapFetch<{ connections: { end_id: string }[] }>(
      `/connections?start_id=${followerId}&connection_type=follow`,
    );
    return result.connections.some((c) => c.end_id === followeeId);
  } catch {
    return false;
  }
}

export async function getFollowerCount(profileId: string): Promise<number> {
  try {
    const result = await tapFetch<{ connections: unknown[]; pagination: Pagination }>(
      `/connections?end_id=${profileId}&connection_type=follow&limit=1`,
    );
    return result.pagination.total;
  } catch {
    return 0;
  }
}

// ── Content ────────────────────────────────────────────────────────

export async function createContent(opts: {
  profileId: string;
  content: string;
  contentType: string;
  customProperties?: Record<string, string>;
}): Promise<TapestryContent> {
  return tapFetch<TapestryContent>('/content', {
    method: 'POST',
    body: JSON.stringify({
      profile_id: opts.profileId,
      content: opts.content,
      content_type: opts.contentType,
      properties: opts.customProperties ?? {},
    }),
  });
}

export async function getContentByProfile(
  profileId: string,
  limit = 20,
  offset = 0,
): Promise<TapestryContent[]> {
  const result = await tapFetch<{ content: TapestryContent[] }>(
    `/content?profile_id=${profileId}&limit=${limit}&offset=${offset}`,
  );
  return result.content;
}
