'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';

export default function ReplayPage() {
  const params = useParams();
  const sessionId = params.sessionId as string;

  return (
    <div className="page-content">
      <div className="page-padded">
        <Link href="/" className="back-link">
          &larr; BACK TO ARENA
        </Link>

        <h1 className="page-title" style={{ marginBottom: '24px' }}>
          REPLAY &mdash; Session {sessionId}
        </h1>

        <div className="card">
          <div
            style={{
              textAlign: 'center',
              padding: '48px',
              color: 'var(--dim)',
              fontSize: '12px',
              fontFamily: 'var(--mono)',
            }}
          >
            Replay viewer coming soon
          </div>
        </div>
      </div>
    </div>
  );
}
