import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Turbopack (default in Next.js 16) — resolve aliases for Node builtins
  turbopack: {
    resolveAlias: {
      crypto: { browser: 'crypto-browserify' },
      stream: { browser: 'stream-browserify' },
      buffer: { browser: 'buffer' },
      http: { browser: 'stream-http' },
      https: { browser: 'https-browserify' },
      zlib: { browser: 'browserify-zlib' },
      url: { browser: 'url' },
    },
  },
  // Webpack fallback (for --webpack builds)
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      crypto: false,
      stream: false,
      buffer: false,
      http: false,
      https: false,
      zlib: false,
      url: false,
    };
    return config;
  },
};

export default nextConfig;
