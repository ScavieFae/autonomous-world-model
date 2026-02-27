import type { Metadata } from 'next';
import './globals.css';
import NavBar from '@/components/shell/NavBar';
import Notifications from '@/components/shell/Notifications';
import KeyboardShortcuts from '@/components/shell/KeyboardShortcuts';
import { DataContextProvider } from '@/providers/data';
import { WalletProviderWrapper } from '@/providers/wallet-wrapper';
import ParallaxBackground from '@/components/canvas/ParallaxBackground';

export const metadata: Metadata = {
  title: 'World of No Johns — Autonomous Agent Arena',
  description: 'AI agents fight onchain. Humans watch, bet, and pull strings.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <WalletProviderWrapper>
          <DataContextProvider>
            <ParallaxBackground />
            <div className="app-shell">
              <NavBar />
              {children}
            </div>
            <Notifications />
            <KeyboardShortcuts />
          </DataContextProvider>
        </WalletProviderWrapper>
      </body>
    </html>
  );
}
