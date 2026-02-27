'use client';

import dynamic from 'next/dynamic';

const ArenaView = dynamic(() => import('@/components/arena/ArenaView'), { ssr: false });

export default function ArenaPage() {
  return <ArenaView />;
}
