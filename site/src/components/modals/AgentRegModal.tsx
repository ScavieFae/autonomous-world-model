'use client';

import { useState } from 'react';
import { CHARACTERS } from '@/engine/constants';
import { useNotificationStore } from '@/stores/notifications';
import { useUserStore } from '@/stores/user';
import { findOrCreateProfile } from '@/lib/tapestry';
import { useDataStore } from '@/stores/data';

const hasTapestry = !!process.env.NEXT_PUBLIC_TAPESTRY_API_KEY;

interface AgentRegModalProps {
  onClose: () => void;
}

export default function AgentRegModal({ onClose }: AgentRegModalProps) {
  const notify = useNotificationStore((s) => s.notify);
  const wallet = useUserStore((s) => s.wallet);
  const fetchAll = useDataStore((s) => s.fetchAll);

  const [username, setUsername] = useState('');
  const [character, setCharacter] = useState('1');
  const [bio, setBio] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit() {
    if (!username.trim()) return;
    setError('');

    if (!wallet) {
      setError('Connect your wallet first');
      return;
    }

    if (!hasTapestry) {
      // Mock mode — just notify
      notify(`Agent @${username.trim()} registered!`, 'success');
      onClose();
      return;
    }

    setIsSubmitting(true);
    try {
      await findOrCreateProfile({
        walletAddress: wallet,
        username: username.trim(),
        bio,
        customProperties: {
          characterId: character,
          agentType: 'mamba2-v1',
          elo: '1200',
          wins: '0',
          losses: '0',
          winStreak: '0',
          totalEarnings: '0',
          followers: '0',
        },
      });

      // Refresh the data store
      await fetchAll();

      notify(`Agent @${username.trim()} registered!`, 'success');
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">Register Agent</div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {!wallet && (
            <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--p2)' }}>
              Connect your wallet to register an agent
            </div>
          )}

          <div>
            <label className="input-label">Username</label>
            <input
              className="input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="my-agent-name"
              disabled={isSubmitting}
            />
          </div>

          <div>
            <label className="input-label">Character</label>
            <select
              className="input"
              value={character}
              onChange={(e) => setCharacter(e.target.value)}
              disabled={isSubmitting}
            >
              {Object.entries(CHARACTERS).map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="input-label">Bio</label>
            <input
              className="input"
              value={bio}
              onChange={(e) => setBio(e.target.value)}
              placeholder="Trained on 10M frames..."
              disabled={isSubmitting}
            />
          </div>

          {error && (
            <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--red)' }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isSubmitting || !username.trim()}
            >
              {isSubmitting ? 'Registering...' : 'Register'}
            </button>
            <button className="btn" onClick={onClose} disabled={isSubmitting}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
