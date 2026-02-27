'use client';

import dynamic from 'next/dynamic';
import type { ReactNode } from 'react';

const SolanaWalletProvider = dynamic(
  () => import('@/providers/wallet').then((m) => m.SolanaWalletProvider),
  { ssr: false },
);

export function WalletProviderWrapper({ children }: { children: ReactNode }) {
  return <SolanaWalletProvider>{children}</SolanaWalletProvider>;
}
