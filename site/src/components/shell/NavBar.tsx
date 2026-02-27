'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useWallet } from '@solana/wallet-adapter-react';
import { useWalletModal } from '@solana/wallet-adapter-react-ui';
import { useUserStore } from '@/stores/user';

const NAV_LINKS = [
  { href: '/', label: 'Arena' },
  { href: '/live', label: 'Live' },
  { href: '/agents', label: 'Agents' },
  { href: '/leaderboard', label: 'Leaderboard' },
  { href: '/model', label: 'Model' },
];

function truncateAddress(addr: string): string {
  return `${addr.slice(0, 4)}...${addr.slice(-4)}`;
}

export default function NavBar() {
  const pathname = usePathname();
  const { publicKey, connected, disconnect } = useWallet();
  const { setVisible } = useWalletModal();
  const { wallet, onConnect, onDisconnect } = useUserStore();

  // Sync wallet adapter state → user store
  useEffect(() => {
    if (connected && publicKey) {
      const addr = publicKey.toBase58();
      if (addr !== wallet) {
        onConnect(addr);
      }
    } else if (!connected && wallet) {
      onDisconnect();
    }
  }, [connected, publicKey, wallet, onConnect, onDisconnect]);

  function isActive(href: string) {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  }

  function handleWalletClick() {
    if (connected) {
      disconnect();
    } else {
      setVisible(true);
    }
  }

  return (
    <nav className="nav">
      <Link href="/" className="nav-logo">
        <span className="dot">&#x25C9;</span>
        WORLD OF NO JOHNS
      </Link>
      <div className="nav-links">
        {NAV_LINKS.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`nav-link ${isActive(link.href) ? 'active' : ''}`}
          >
            {link.label}
          </Link>
        ))}
      </div>
      <div className="nav-right">
        <button className="wallet-btn" onClick={handleWalletClick}>
          {connected && publicKey
            ? truncateAddress(publicKey.toBase58())
            : 'Connect Wallet'}
        </button>
      </div>
    </nav>
  );
}
