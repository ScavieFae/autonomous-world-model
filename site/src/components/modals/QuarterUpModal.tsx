'use client';

import { useState } from 'react';
import { useData, type Agent } from '@/providers/data';
import { useNotificationStore } from '@/stores/notifications';

interface QuarterUpModalProps {
  onClose: () => void;
  preselectedAgent?: Agent;
}

const TIERS = [
  { label: 'Casual', fee: '0.001 SOL' },
  { label: 'Ranked', fee: '0.01 SOL' },
  { label: 'High Stakes', fee: '0.1 SOL' },
];

type Step = 'select-agent' | 'select-tier' | 'confirm' | 'result';

export default function QuarterUpModal({ onClose, preselectedAgent }: QuarterUpModalProps) {
  const data = useData();
  const notify = useNotificationStore((s) => s.notify);
  const agents = data.getAgents();

  const [step, setStep] = useState<Step>(preselectedAgent ? 'select-tier' : 'select-agent');
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(preselectedAgent ?? null);
  const [selectedTier, setSelectedTier] = useState(1);

  function handleSelectAgent(agent: Agent) {
    setSelectedAgent(agent);
    setStep('select-tier');
  }

  function handleSelectTier(idx: number) {
    setSelectedTier(idx);
    setStep('confirm');
  }

  function handleConfirm() {
    setStep('result');
    notify(`QUARTERED UP for @${selectedAgent?.username} · ${TIERS[selectedTier].fee}`, 'success');
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">Quarter Up</div>

        {step === 'select-agent' && (
          <>
            <div className="modal-step">Step 1 / 4 — Select Agent</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 300, overflowY: 'auto' }}>
              {agents.map((agent) => (
                <button
                  key={agent.id}
                  className="btn"
                  style={{ textAlign: 'left', justifyContent: 'flex-start' }}
                  onClick={() => handleSelectAgent(agent)}
                >
                  @{agent.username} · {agent.character} · ELO {agent.elo}
                </button>
              ))}
            </div>
          </>
        )}

        {step === 'select-tier' && (
          <>
            <div className="modal-step">Step 2 / 4 — Select Tier</div>
            <div style={{ marginBottom: 12, fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--dim)' }}>
              Sponsoring @{selectedAgent?.username}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {TIERS.map((tier, idx) => (
                <button
                  key={tier.label}
                  className="btn"
                  onClick={() => handleSelectTier(idx)}
                >
                  {tier.label} — {tier.fee}
                </button>
              ))}
            </div>
          </>
        )}

        {step === 'confirm' && (
          <>
            <div className="modal-step">Step 3 / 4 — Confirm</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 2 }}>
              <div>Agent: <span style={{ color: 'var(--p1)' }}>@{selectedAgent?.username}</span></div>
              <div>Tier: {TIERS[selectedTier].label}</div>
              <div>Entry Fee: {TIERS[selectedTier].fee}</div>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <button className="btn btn-primary" onClick={handleConfirm}>
                Confirm Quarter Up
              </button>
              <button className="btn" onClick={onClose}>Cancel</button>
            </div>
          </>
        )}

        {step === 'result' && (
          <>
            <div className="modal-step">Step 4 / 4 — Done</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--p1)', marginBottom: 16 }}>
              QUARTERED UP!
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--dim)', lineHeight: 2 }}>
              <div>@{selectedAgent?.username} is entering the queue</div>
              <div>Entry fee: {TIERS[selectedTier].fee}</div>
            </div>
            <button className="btn" onClick={onClose} style={{ marginTop: 16 }}>
              Close
            </button>
          </>
        )}
      </div>
    </div>
  );
}
