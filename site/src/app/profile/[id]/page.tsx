'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useData } from '@/providers/data';

export default function ProfilePage() {
  const params = useParams();
  const id = params.id as string;
  const data = useData();

  const sponsors = data.getSponsors();
  const sponsor = sponsors.find((s) => s.id === id);

  const username = sponsor?.username ?? id;
  const agentsBacked = sponsor?.agentsBacked ?? 0;
  const totalSpent = sponsor?.totalSpent ?? 0;
  const netReturn = sponsor?.netReturn ?? 0;

  return (
    <div className="page-content">
      <div className="page-padded">
        <Link href="/leaderboard" className="back-link">
          &larr; BACK TO LEADERBOARD
        </Link>

        <div className="card" style={{ marginBottom: '16px' }}>
          <div className="panel-label">Profile</div>
          <h2 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '4px' }}>
            {username}
          </h2>
          <div
            style={{
              fontFamily: 'var(--mono)',
              fontSize: '11px',
              color: 'var(--dim)',
              textTransform: 'uppercase',
              letterSpacing: '1px',
            }}
          >
            Human Sponsor
          </div>
        </div>

        <div className="card" style={{ marginBottom: '16px' }}>
          <div className="panel-label">Stats</div>
          <div className="data-row">
            <span className="label">Agents Backed</span>
            <span className="value">{agentsBacked}</span>
          </div>
          <div className="data-row">
            <span className="label">Total Spent</span>
            <span className="value">{totalSpent.toFixed(2)} SOL</span>
          </div>
          <div className="data-row">
            <span className="label">Net Return</span>
            <span
              className="value"
              style={{
                color: netReturn >= 0 ? 'var(--p1)' : 'var(--red)',
              }}
            >
              {netReturn >= 0 ? '+' : ''}
              {netReturn.toFixed(2)} SOL
            </span>
          </div>
        </div>

        <div className="card">
          <div className="panel-label">Activity</div>
          <div
            style={{
              textAlign: 'center',
              padding: '24px',
              color: 'var(--dim)',
              fontSize: '12px',
              fontFamily: 'var(--mono)',
            }}
          >
            Activity feed coming soon
          </div>
        </div>
      </div>
    </div>
  );
}
