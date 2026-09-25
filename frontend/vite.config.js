import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    allowedHosts: ['.devtunnels.ms'], // Vite rejects unknown Host headers, which blocks the forwarded-port URL
    // The browser only ever talks to this server, so one forwarded port (5173) is enough for a phone.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
