import { create } from 'zustand';

interface NotificationState {
  message: string;
  type: 'info' | 'error' | 'success';
  visible: boolean;
  notify: (message: string, type?: 'info' | 'error' | 'success') => void;
  dismiss: () => void;
}

export const useNotificationStore = create<NotificationState>((set) => ({
  message: '',
  type: 'info',
  visible: false,
  notify: (message, type = 'info') => set({ message, type, visible: true }),
  dismiss: () => set({ visible: false }),
}));
