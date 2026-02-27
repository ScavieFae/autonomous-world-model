'use client';

import dynamic from 'next/dynamic';

const LiveArenaView = dynamic(() => import('@/components/arena/LiveArenaView'), { ssr: false });

export default function LivePage() {
  return <LiveArenaView />;
}
