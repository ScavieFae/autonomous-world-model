'use client';

import { useState } from 'react';
import { useData, type Agent } from '@/providers/data';
import { useNotificationStore } from '@/stores/notifications';
import { useUserStore } from '@/stores/user';
import { useWallet } from '@solana/wallet-adapter-react';
import { useConnection } from '@solana/wallet-adapter-react';
import { SystemProgram, Transaction, PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { createContent } from '@/lib/tapestry';

const hasTapestry = !!process.env.NEXT_PUBLIC_TAPESTRY_API_KEY;

// Hackathon devnet treasury — replace with real protocol treasury later
const TREASURY_ADDRESS = new PublicKey('11111111111111111111111111111112');

interface QuarterUpModalProps {
  onClose: () => void;
  preselectedAgent?: Agent;
}

const TIERS = [
  { label: 'Casual', fee: '0.001 SOL', feeLamports: 1_000_000 },
  { label: 'Ranked', fee: '0.01 SOL', feeLamports: 10_000_000 },
  { label: 'High Stakes', fee: '0.1 SOL', feeLamports: 100_000_000 },
];

type Step = 'select-agent' | 'select-tier' | 'confirm' | 'sending' | 'result';

export default function QuarterUpModal({ onClose, preselectedAgent }: QuarterUpModalProps) {
  const data = useData();
  const notify = useNotificationStore((s) => s.notify);
  const wallet = useUserStore((s) => s.wallet);
  const profile = useUserStore((s) => s.profile);
  const { publicKey, sendTransaction } = useWallet();
  const { connection } = useConnection();
  const agents = data.getAgents();

  const [step, setStep] = useState<Step>(preselectedAgent ? 'select-tier' : 'select-agent');
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(preselectedAgent ?? null);
  const [selectedTier, setSelectedTier] = useState(1);
  const [error, setError] = useState('');
  const [txSig, setTxSig] = useState('');

  function handleSelectAgent(agent: Agent) {
    setSelectedAgent(agent);
    setStep('select-tier');
  }

  function handleSelectTier(idx: number) {
    setSelectedTier(idx);
    setStep('confirm');
  }

  async function handleConfirm() {
    setError('');

    if (!wallet || !publicKey) {
      setError('Connect your wallet first');
      return;
    }

    setStep('sending');

    try {
      // Send SOL to treasury
      const tx = new Transaction().add(
        SystemProgram.transfer({
          fromPubkey: publicKey,
          toPubkey: TREASURY_ADDRESS,
          lamports: TIERS[selectedTier].feeLamports,
        }),
      );

      const sig = await sendTransaction(tx, connection);
      setTxSig(sig);

      // Post sponsorship to Tapestry if connected
      if (hasTapestry && profile) {
        await createContent({
          profileId: profile.id,
          content: `Quartered up for @${selectedAgent?.username} — ${TIERS[selectedTier].label}`,
          contentType: 'sponsorship',
          customProperties: {
            agentId: selectedAgent?.id ?? '',
            agentUsername: selectedAgent?.username ?? '',
            tier: TIERS[selectedTier].label,
            feeLamports: String(TIERS[selectedTier].feeLamports),
            txSignature: sig,
            timestamp: String(Date.now()),
          },
        });
      }

      setStep('result');
      notify(`QUARTERED UP for @${selectedAgent?.username} · ${TIERS[selectedTier].fee}`, 'success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Transaction failed');
      setStep('confirm');
    }
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
            {!wallet && (
              <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--p2)', marginBottom: 8 }}>
                Connect your wallet to sponsor
              </div>
            )}
            <div style={{ fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 2 }}>
              <div>Agent: <span style={{ color: 'var(--p1)' }}>@{selectedAgent?.username}</span></div>
              <div>Tier: {TIERS[selectedTier].label}</div>
              <div>Entry Fee: {TIERS[selectedTier].fee}</div>
            </div>
            {error && (
              <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--red)', marginTop: 8 }}>
                {error}
              </div>
            )}
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <button
                className="btn btn-primary"
                onClick={handleConfirm}
                disabled={!wallet}
              >
                Confirm Quarter Up
              </button>
              <button className="btn" onClick={onClose}>Cancel</button>
            </div>
          </>
        )}

        {step === 'sending' && (
          <>
            <div className="modal-step">Sending transaction...</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--dim)', textAlign: 'center', padding: 24 }}>
              Confirm in your wallet...
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
              {txSig && (
                <div style={{ fontSize: 10, wordBreak: 'break-all', marginTop: 4 }}>
                  TX: {txSig.slice(0, 20)}...
                </div>
              )}
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
