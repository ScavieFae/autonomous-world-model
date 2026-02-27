'use client';

import { useEffect, useState } from 'react';

const SHORTCUTS = [
  { key: 'W', desc: 'Wire render mode' },
  { key: 'C', desc: 'Character render mode' },
  { key: 'D', desc: 'Data render mode' },
  { key: 'X', desc: 'X-Ray render mode' },
  { key: 'Space', desc: 'Play / Pause' },
  { key: '\u2190 / \u2192', desc: 'Frame step' },
  { key: '+ / -', desc: 'Speed' },
  { key: 'F', desc: 'Fullscreen' },
  { key: '?', desc: 'This overlay' },
];

export default function KeyboardShortcuts() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === '?') {
        e.preventDefault();
        setOpen((v) => !v);
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  if (!open) return null;

  return (
    <div className="shortcuts-overlay" onClick={() => setOpen(false)}>
      <div className="shortcuts-panel" onClick={(e) => e.stopPropagation()}>
        <h3>Keyboard Shortcuts</h3>
        {SHORTCUTS.map((s) => (
          <div key={s.key} className="shortcut-row">
            <span style={{ color: 'var(--dim)' }}>{s.desc}</span>
            <kbd>{s.key}</kbd>
          </div>
        ))}
      </div>
    </div>
  );
}
