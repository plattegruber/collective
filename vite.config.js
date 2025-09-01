import { defineConfig } from 'vite';

export default defineConfig({
  base: '/collective/',
  publicDir: 'public',
  server: {
    host: '0.0.0.0',
    port: 3000,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    target: 'es2022',
    rollupOptions: {
      external: ['three'],
    },
  },
});