'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV_LINKS = [
  { href: '/', label: 'Arena' },
  { href: '/live', label: 'Live' },
  { href: '/agents', label: 'Agents' },
  { href: '/leaderboard', label: 'Leaderboard' },
  { href: '/model', label: 'Model' },
];

export default function NavBar() {
  const pathname = usePathname();

  function isActive(href: string) {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
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
        <button className="wallet-btn">Connect Wallet</button>
      </div>
    </nav>
  );
}
