'use client';

import { useEffect, useState } from 'react';

interface Architecture {
  model_type: string;
  d_model?: number;
  d_inner?: number;
  d_state?: number;
  n_layers?: number;
  nheads?: number;
  headdim?: number;
  hidden_dim?: number;
  trunk_dim?: number;
}

interface Shard {
  index: number;
  offset: number;
  size: number;
}

interface WeightEntry {
  key: string;
  group: string;
  shape: number[];
  size: number;
  quantization: string;
  snr_db?: number;
}

interface ModelStats {
  format: string;
  architecture: Architecture;
  total_weight_bytes: number;
  fp32_bytes: number;
  compression_ratio: number;
  context_len: number | null;
  shard_map: { num_shards: number; shards: Shard[] };
  error_summary: {
    num_2d_weights: number;
    mean_snr_db: number;
    min_snr_db: number;
    max_snr_db: number;
  };
  weight_groups: Record<string, { count: number; total_bytes: number }>;
  weights: WeightEntry[];
}

interface LutStats {
  total_bytes: number;
  num_luts: number;
  lut_size: number;
  functions: string[];
}

interface AllStats {
  generated_at: string;
  world_model: ModelStats;
  policy: ModelStats;
  luts: LutStats;
}

function fmtBytes(b: number): string {
  if (b >= 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  if (b >= 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${b} B`;
}

function fmtSNR(snr: number): string {
  return `${snr.toFixed(1)} dB`;
}

function snrQuality(snr: number): { label: string; color: string } {
  if (snr >= 40) return { label: 'EXCELLENT', color: 'var(--p1)' };
  if (snr >= 30) return { label: 'GOOD', color: 'var(--p2)' };
  if (snr >= 20) return { label: 'OK', color: 'var(--dim)' };
  return { label: 'LOW', color: 'var(--red)' };
}

function SNRBar({ snr }: { snr: number }) {
  const pct = Math.min(100, (snr / 60) * 100);
  const q = snrQuality(snr);
  return (
    <div className="snr-bar-container">
      <div className="snr-bar-track">
        <div
          className="snr-bar-fill"
          style={{ width: `${pct}%`, background: q.color }}
        />
      </div>
      <span className="snr-bar-label" style={{ color: q.color }}>
        {fmtSNR(snr)}
      </span>
    </div>
  );
}

function ModelCard({
  title,
  stats,
  accent,
}: {
  title: string;
  stats: ModelStats;
  accent: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const arch = stats.architecture;
  const q = snrQuality(stats.error_summary.mean_snr_db);

  return (
    <div className="card" style={{ borderLeft: `3px solid ${accent}` }}>
      <div className="model-card-header">
        <div>
          <div className="panel-label">{title}</div>
          <div className="model-type-label">{arch.model_type.toUpperCase()}</div>
        </div>
        <span className="tag" style={{
          background: `${accent}22`,
          color: accent,
          border: `1px solid ${accent}33`,
        }}>
          INT8
        </span>
      </div>

      {/* Architecture */}
      <div className="model-section">
        <div className="model-section-title">Architecture</div>
        {arch.d_model != null && (
          <>
            <div className="data-row"><span className="label">d_model</span><span className="value">{arch.d_model}</span></div>
            <div className="data-row"><span className="label">d_inner</span><span className="value">{arch.d_inner}</span></div>
            <div className="data-row"><span className="label">d_state</span><span className="value">{arch.d_state}</span></div>
            <div className="data-row"><span className="label">Layers</span><span className="value">{arch.n_layers}</span></div>
            <div className="data-row"><span className="label">Heads</span><span className="value">{arch.nheads}</span></div>
            <div className="data-row"><span className="label">Head dim</span><span className="value">{arch.headdim}</span></div>
          </>
        )}
        {arch.hidden_dim != null && (
          <>
            <div className="data-row"><span className="label">Hidden dim</span><span className="value">{arch.hidden_dim}</span></div>
            <div className="data-row"><span className="label">Trunk dim</span><span className="value">{arch.trunk_dim}</span></div>
          </>
        )}
        {stats.context_len != null && (
          <div className="data-row"><span className="label">Context len</span><span className="value">{stats.context_len} frames</span></div>
        )}
      </div>

      {/* Size */}
      <div className="model-section">
        <div className="model-section-title">Size</div>
        <div className="data-row"><span className="label">FP32</span><span className="value">{fmtBytes(stats.fp32_bytes)}</span></div>
        <div className="data-row"><span className="label">INT8</span><span className="value" style={{ color: accent }}>{fmtBytes(stats.total_weight_bytes)}</span></div>
        <div className="data-row"><span className="label">Compression</span><span className="value">{stats.compression_ratio}x</span></div>
        <div className="data-row"><span className="label">Rent (est.)</span><span className="value">{(stats.total_weight_bytes * 0.00000696).toFixed(1)} SOL</span></div>
      </div>

      {/* Quality */}
      <div className="model-section">
        <div className="model-section-title">Quantization Quality</div>
        <div className="data-row">
          <span className="label">Mean SNR</span>
          <span className="value" style={{ color: q.color }}>{fmtSNR(stats.error_summary.mean_snr_db)} {q.label}</span>
        </div>
        <div className="data-row"><span className="label">Min SNR</span><SNRBar snr={stats.error_summary.min_snr_db} /></div>
        <div className="data-row"><span className="label">Max SNR</span><SNRBar snr={stats.error_summary.max_snr_db} /></div>
        <div className="data-row"><span className="label">Matrix weights</span><span className="value">{stats.error_summary.num_2d_weights}</span></div>
      </div>

      {/* Weight groups */}
      <div className="model-section">
        <div className="model-section-title">Weight Groups</div>
        {Object.entries(stats.weight_groups)
          .filter(([, g]) => g.count > 0)
          .map(([name, g]) => (
            <div className="data-row" key={name}>
              <span className="label">{name.replace('_', ' ')}</span>
              <span className="value">{g.count} ({fmtBytes(g.total_bytes)})</span>
            </div>
          ))}
      </div>

      {/* Shards */}
      <div className="model-section">
        <div className="model-section-title">Solana Shards</div>
        {stats.shard_map.shards.map((s) => (
          <div className="data-row" key={s.index}>
            <span className="label">WeightShard #{s.index}</span>
            <span className="value">{fmtBytes(s.size)}</span>
          </div>
        ))}
      </div>

      {/* Weight detail toggle */}
      <button
        className="ctrl-btn mt-2"
        style={{ width: '100%', fontSize: '10px' }}
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? 'HIDE' : 'SHOW'} ALL {stats.weights.length} WEIGHTS
      </button>

      {expanded && (
        <div className="model-weight-list mt-2">
          <table className="rank-table" style={{ fontSize: '10px' }}>
            <thead>
              <tr>
                <th>Key</th>
                <th>Shape</th>
                <th>Size</th>
                <th>SNR</th>
              </tr>
            </thead>
            <tbody>
              {stats.weights.map((w) => {
                const wq = w.snr_db != null ? snrQuality(w.snr_db) : null;
                return (
                  <tr key={w.key}>
                    <td style={{ fontFamily: 'var(--mono)', fontSize: '9px', wordBreak: 'break-all' }}>{w.key}</td>
                    <td style={{ fontFamily: 'var(--mono)' }}>{w.shape.join('x')}</td>
                    <td style={{ fontFamily: 'var(--mono)' }}>{fmtBytes(w.size)}</td>
                    <td style={{ color: wq?.color ?? 'var(--dim)', fontFamily: 'var(--mono)' }}>
                      {w.snr_db != null ? fmtSNR(w.snr_db) : '--'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function ModelDashboard() {
  const [stats, setStats] = useState<AllStats | null>(null);

  useEffect(() => {
    fetch('/model-stats.json')
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  if (!stats) {
    return (
      <div className="page-content">
        <div className="page-padded">
          <div className="page-header">
            <h1 className="page-title">MODEL</h1>
          </div>
          <div style={{ color: 'var(--dim)', fontFamily: 'var(--mono)', fontSize: '12px' }}>
            Loading model stats...
          </div>
        </div>
      </div>
    );
  }

  const totalOnchain =
    stats.world_model.total_weight_bytes +
    stats.policy.total_weight_bytes +
    stats.luts.total_bytes;

  return (
    <div className="page-content">
      <div className="page-padded">
        <div className="page-header">
          <h1 className="page-title">MODEL</h1>
          <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--dim)' }}>
            Quantized {stats.generated_at}
          </span>
        </div>

        {/* Summary row */}
        <div className="model-summary-row">
          <div className="model-stat-box">
            <div className="model-stat-value">{fmtBytes(totalOnchain)}</div>
            <div className="model-stat-label">Total Onchain</div>
          </div>
          <div className="model-stat-box">
            <div className="model-stat-value">4.0x</div>
            <div className="model-stat-label">Compression</div>
          </div>
          <div className="model-stat-box">
            <div className="model-stat-value">{fmtSNR(stats.world_model.error_summary.mean_snr_db)}</div>
            <div className="model-stat-label">World Model SNR</div>
          </div>
          <div className="model-stat-box">
            <div className="model-stat-value">{fmtSNR(stats.policy.error_summary.mean_snr_db)}</div>
            <div className="model-stat-label">Policy SNR</div>
          </div>
        </div>

        {/* LUTs card */}
        <div className="card mb-4" style={{ borderLeft: '3px solid var(--cyan)' }}>
          <div className="panel-label">Activation LUTs</div>
          <div className="model-lut-row">
            {stats.luts.functions.map((fn) => (
              <div key={fn} className="model-lut-chip">
                {fn}
              </div>
            ))}
          </div>
          <div className="data-row mt-2">
            <span className="label">Total size</span>
            <span className="value">{stats.luts.num_luts} x {stats.luts.lut_size} = {fmtBytes(stats.luts.total_bytes)}</span>
          </div>
          <div className="data-row">
            <span className="label">Solana account</span>
            <span className="value">Embedded in ModelManifest</span>
          </div>
        </div>

        {/* Model cards side by side */}
        <div className="model-cards-grid">
          <ModelCard
            title="World Model"
            stats={stats.world_model}
            accent="var(--p1)"
          />
          <ModelCard
            title="Policy"
            stats={stats.policy}
            accent="var(--p2)"
          />
        </div>

        {/* Solana account map */}
        <div className="card mt-4" style={{ borderLeft: '3px solid var(--violet)' }}>
          <div className="panel-label">Solana Account Map</div>
          <table className="rank-table">
            <thead>
              <tr>
                <th>Account</th>
                <th>Type</th>
                <th>Size</th>
                <th>Rent (est.)</th>
              </tr>
            </thead>
            <tbody>
              {stats.world_model.shard_map.shards.map((s) => (
                <tr key={`wm-${s.index}`}>
                  <td style={{ fontFamily: 'var(--mono)' }}>WeightShard (WM #{s.index})</td>
                  <td>INT8 weights</td>
                  <td style={{ fontFamily: 'var(--mono)' }}>{fmtBytes(s.size)}</td>
                  <td style={{ fontFamily: 'var(--mono)' }}>{(s.size * 0.00000696).toFixed(1)} SOL</td>
                </tr>
              ))}
              {stats.policy.shard_map.shards.map((s) => (
                <tr key={`pol-${s.index}`}>
                  <td style={{ fontFamily: 'var(--mono)' }}>WeightShard (Policy #{s.index})</td>
                  <td>INT8 weights</td>
                  <td style={{ fontFamily: 'var(--mono)' }}>{fmtBytes(s.size)}</td>
                  <td style={{ fontFamily: 'var(--mono)' }}>{(s.size * 0.00000696).toFixed(1)} SOL</td>
                </tr>
              ))}
              <tr>
                <td style={{ fontFamily: 'var(--mono)' }}>ModelManifest (WM)</td>
                <td>Architecture + scales + LUTs</td>
                <td style={{ fontFamily: 'var(--mono)' }}>~2 KB</td>
                <td style={{ fontFamily: 'var(--mono)' }}>0.01 SOL</td>
              </tr>
              <tr>
                <td style={{ fontFamily: 'var(--mono)' }}>ModelManifest (Policy)</td>
                <td>Architecture + scales</td>
                <td style={{ fontFamily: 'var(--mono)' }}>~1 KB</td>
                <td style={{ fontFamily: 'var(--mono)' }}>0.01 SOL</td>
              </tr>
              <tr>
                <td style={{ fontFamily: 'var(--mono)' }}>SessionState</td>
                <td>Per-frame game state</td>
                <td style={{ fontFamily: 'var(--mono)' }}>~1 KB</td>
                <td style={{ fontFamily: 'var(--mono)' }}>0.01 SOL</td>
              </tr>
              <tr>
                <td style={{ fontFamily: 'var(--mono)' }}>HiddenState</td>
                <td>Mamba2 recurrent state</td>
                <td style={{ fontFamily: 'var(--mono)' }}>~200 KB</td>
                <td style={{ fontFamily: 'var(--mono)' }}>1.39 SOL</td>
              </tr>
            </tbody>
          </table>
          <div className="data-row mt-2" style={{ borderTop: '1px solid var(--border)', paddingTop: '8px' }}>
            <span className="label">Total rent deposit</span>
            <span className="value" style={{ color: 'var(--violet)' }}>
              ~{((totalOnchain + 200 * 1024 + 4096) * 0.00000696).toFixed(1)} SOL
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
