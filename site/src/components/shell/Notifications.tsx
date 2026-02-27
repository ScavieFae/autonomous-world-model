'use client';

import { useNotificationStore } from '@/stores/notifications';
import { useEffect } from 'react';

export default function Notifications() {
  const { message, type, visible, dismiss } = useNotificationStore();

  useEffect(() => {
    if (visible && type !== 'error') {
      const timer = setTimeout(dismiss, 5000);
      return () => clearTimeout(timer);
    }
  }, [visible, type, dismiss]);

  return (
    <div className={`notification-bar ${visible ? 'visible' : ''} ${type === 'error' ? 'error' : ''}`}>
      <span>{message}</span>
      <button
        className="ctrl-btn btn-sm"
        onClick={dismiss}
        style={{ padding: '2px 8px', fontSize: 10 }}
      >
        &#x2715;
      </button>
    </div>
  );
}
