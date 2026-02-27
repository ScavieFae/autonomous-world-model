'use client';

import { useState } from 'react';
import { CHARACTERS } from '@/engine/constants';
import { useNotificationStore } from '@/stores/notifications';

interface AgentRegModalProps {
  onClose: () => void;
}

export default function AgentRegModal({ onClose }: AgentRegModalProps) {
  const notify = useNotificationStore((s) => s.notify);
  const [username, setUsername] = useState('');
  const [character, setCharacter] = useState('1');
  const [bio, setBio] = useState('');

  function handleSubmit() {
    if (!username.trim()) return;
    notify(`Agent @${username.trim()} registered!`, 'success');
    onClose();
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">Register Agent</div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <label className="input-label">Username</label>
            <input
              className="input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="my-agent-name"
            />
          </div>

          <div>
            <label className="input-label">Character</label>
            <select
              className="input"
              value={character}
              onChange={(e) => setCharacter(e.target.value)}
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
            />
          </div>

          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button className="btn btn-primary" onClick={handleSubmit}>
              Register
            </button>
            <button className="btn" onClick={onClose}>Cancel</button>
          </div>
        </div>
      </div>
    </div>
  );
}
