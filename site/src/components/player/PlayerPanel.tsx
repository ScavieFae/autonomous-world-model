'use client';

import type { VizPlayerFrame } from '@/engine/types';
import { actionName, CHARACTERS } from '@/engine/constants';

interface PlayerPanelProps {
  player: VizPlayerFrame;
  index: number; // 0 or 1
  agentName?: string;
  sponsor?: string;
  compact?: boolean;
}

export default function PlayerPanel({ player: p, index, agentName, sponsor, compact }: PlayerPanelProps) {
  const n = index + 1;
  const charName = CHARACTERS[p.character] || `Char_${p.character}`;
  const isHigh = p.percent > 100;

  if (compact) {
    return (
      <div className="arena-hud-panel">
        <div className="flex items-center justify-between mb-2">
          <span style={{ color: `var(--p${n})`, fontWeight: 600, fontSize: 13 }}>
            {agentName ? `@${agentName}` : `P${n}`}
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dim)' }}>
            {charName}
          </span>
        </div>

        <div className={`percent percent-sm ${isHigh ? 'high' : ''}`} style={{ color: isHigh ? undefined : `var(--p${n})` }}>
          {Math.round(p.percent)}%
        </div>

        <div className="stocks-row mt-2">
          {Array.from({ length: 4 }, (_, s) => (
            <div key={s} className={`stock-pip ${s < p.stocks ? `alive p${n}` : 'dead'}`} />
          ))}
        </div>

        <div className="action-badge mt-2">
          {actionName(p.action_state)}
        </div>

        {sponsor && (
          <div style={{ marginTop: 6, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dim)' }}>
            sponsor: @{sponsor}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="arena-panel-section">
      <div className="panel-label">Player {n}</div>
      <div className={`card card-p${n}`}>
        <div className="flex items-center justify-between mb-2">
          <span style={{ color: `var(--p${n})`, fontWeight: 600, fontSize: 13 }}>
            {agentName ? `@${agentName}` : `P${n}`}
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dim)' }}>
            {charName}
          </span>
        </div>

        <div className={`percent percent-sm ${isHigh ? 'high' : ''}`} style={{ color: isHigh ? undefined : `var(--p${n})` }}>
          {Math.round(p.percent)}%
        </div>

        <div className="stocks-row mt-2">
          {Array.from({ length: 4 }, (_, s) => (
            <div key={s} className={`stock-pip ${s < p.stocks ? `alive p${n}` : 'dead'}`} />
          ))}
        </div>

        <div className="action-badge mt-2">
          {actionName(p.action_state)} ({p.action_state})
        </div>

        <div style={{ marginTop: 8 }}>
          <div className="data-row">
            <span className="label">Position</span>
            <span className="value">{p.x.toFixed(1)}, {p.y.toFixed(1)}</span>
          </div>
          <div className="data-row">
            <span className="label">Velocity</span>
            <span className="value">
              {(p.on_ground ? p.speed_ground_x : p.speed_air_x).toFixed(2)}, {p.speed_y.toFixed(2)}
            </span>
          </div>
          <div className="data-row">
            <span className="label">Facing</span>
            <span className="value">{p.facing ? 'Right' : 'Left'}</span>
          </div>
          <div className="data-row">
            <span className="label">Airborne</span>
            <span className="value">{p.on_ground ? 'No' : 'Yes'}</span>
          </div>
          <div className="data-row">
            <span className="label">Shield</span>
            <span className="value">{p.shield_strength.toFixed(1)}</span>
          </div>
          <div className="data-row">
            <span className="label">Jumps</span>
            <span className="value">{p.jumps_left}</span>
          </div>
          <div className="data-row">
            <span className="label">State age</span>
            <span className="value">{p.state_age}</span>
          </div>
          <div className="data-row">
            <span className="label">Hitlag</span>
            <span className="value">{p.hitlag || 0}</span>
          </div>
        </div>

        {sponsor && (
          <div style={{ marginTop: 8, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--dim)' }}>
            sponsor: @{sponsor}
          </div>
        )}
      </div>
    </div>
  );
}
